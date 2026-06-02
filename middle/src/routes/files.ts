import { Router } from 'express'
import { getEngine } from '../engine.js'
import { v4 as uuidv4 } from 'uuid'
import { existsSync, mkdirSync, createWriteStream, promises as fsPromises } from 'fs'
import { join } from 'path'
import { sanitizeFilename, sanitizeForLog } from '../security/validation.js'
import { lookup } from 'dns'
import { promisify } from 'util'

const lookupAsync = promisify(lookup)

const router = Router()
const UPLOAD_DIR = process.env.UPLOAD_DIR || './data/uploads'
const MAX_FILE_SIZE = 50 * 1024 * 1024
const USER_QUOTA_LIMIT = 500 * 1024 * 1024

// Allowed top-level MIME types. Server-side sniff MUST match
// one of these (or the octet-stream wildcard). Trusting the
// client's x-file-type header alone is a finding (H5).
const ALLOWED_MIME_TYPES = new Set([
  'application/json',
  'text/plain',
  'text/csv',
  'image/png',
  'image/jpeg',
  'image/gif',
  'application/pdf',
  'application/octet-stream',
])

if (!existsSync(UPLOAD_DIR)) {
  mkdirSync(UPLOAD_DIR, { recursive: true })
}

interface FileRecord {
  id: string
  name: string
  size: number
  mimeType: string
  uploadedAt: string
  userId: string
}

const fileRecords = new Map<string, FileRecord>()
const userQuota = new Map<string, number>()

/**
 * Server-side MIME sniff. Reads the first 16 bytes of the buffer
 * and matches against known magic bytes. Returns the detected
 * MIME type, or 'application/octet-stream' as a fallback.
 *
 * The pre-remediation code trusted req.headers['x-file-type']
 * as authoritative. The fix sniffs the content and rejects
 * uploads where the client-declared type disagrees with the
 * sniffed type.
 */
function sniffMime(buf: Buffer): string {
  if (buf.length < 4) return 'application/octet-stream'
  // PNG: 89 50 4E 47
  if (buf[0] === 0x89 && buf[1] === 0x50 && buf[2] === 0x4e && buf[3] === 0x47) return 'image/png'
  // JPEG: FF D8 FF
  if (buf[0] === 0xff && buf[1] === 0xd8 && buf[2] === 0xff) return 'image/jpeg'
  // GIF: 47 49 46 38
  if (buf[0] === 0x47 && buf[1] === 0x49 && buf[2] === 0x46 && buf[3] === 0x38) return 'image/gif'
  // PDF: 25 50 44 46
  if (buf[0] === 0x25 && buf[1] === 0x50 && buf[2] === 0x44 && buf[3] === 0x46) return 'application/pdf'
  // JSON: starts with { or [ (after optional whitespace)
  const head = buf.toString('utf8', 0, Math.min(16, buf.length)).trimStart()
  if (head.startsWith('{') || head.startsWith('[')) return 'application/json'
  // Plain text: all bytes are printable ASCII or common whitespace
  let printable = 0
  for (let i = 0; i < Math.min(512, buf.length); i++) {
    const b = buf[i]
    if ((b >= 0x20 && b < 0x7f) || b === 0x09 || b === 0x0a || b === 0x0d) printable++
  }
  if (printable / Math.min(512, buf.length) > 0.95) return 'text/plain'
  return 'application/octet-stream'
}

function declaredMimeFromHeader(raw: string | string[] | undefined): string {
  if (!raw) return 'application/octet-stream'
  const v = Array.isArray(raw) ? raw[0] : raw
  return v.split(';')[0].trim().toLowerCase()
}

router.post('/upload', async (req, res) => {
  const userId = req.user?.id
  if (!userId) {
    res.status(401).json({ error: 'Unauthorized' })
    return
  }

  const contentLength = parseInt(req.headers['content-length'] || '0', 10)
  if (contentLength > MAX_FILE_SIZE) {
    res.status(413).json({ error: 'File too large (max 50MB)' })
    return
  }

  const currentQuota = userQuota.get(userId) || 0
  if (currentQuota + contentLength > USER_QUOTA_LIMIT) {
    res.status(413).json({ error: 'Quota exceeded' })
    return
  }

  // Sanitize the filename at the boundary. The client may
  // supply x-file-name with path-traversal characters; the
  // fix rejects path separators and parent-directory refs.
  const rawName = String(req.headers['x-file-name'] || `upload-${uuidv4()}`)
  const safeName = sanitizeFilename(rawName)
  if (!safeName) {
    res.status(400).json({ error: 'Invalid filename' })
    return
  }

  const declaredMime = declaredMimeFromHeader(req.headers['x-file-type']) ||
    declaredMimeFromHeader(req.headers['content-type'])

  // Stream upload to a temp file. The pre-remediation code
  // buffered the whole upload in memory (chunks: Buffer[]),
  // which is a memory-exhaustion vector. The fix streams to
  // a temp file, sniffs the first bytes, and renames the
  // temp file to the final id on success.
  const id = uuidv4()
  const tempPath = join(UPLOAD_DIR, `.tmp-${id}`)
  const finalPath = join(UPLOAD_DIR, id)

  let uploaded = 0
  let aborted = false
  const sniffBuf: Buffer[] = []
  let sniffed = false

  await new Promise<void>((resolve) => {
    const out = createWriteStream(tempPath)
    req.on('data', (chunk: Buffer) => {
      if (aborted) return
      uploaded += chunk.length
      if (uploaded > MAX_FILE_SIZE) {
        aborted = true
        try { out.destroy() } catch { /* ignore */ }
        fsPromises.unlink(tempPath).catch(() => undefined)
        if (!res.headersSent) {
          res.status(413).json({ error: 'File too large (max 50MB)' })
        }
        req.destroy()
        resolve()
        return
      }
      if (!sniffed && sniffBuf.length < 1 && chunk.length > 0) {
        sniffBuf.push(chunk.slice(0, Math.min(16, chunk.length)))
        if (sniffBuf[0].length >= 16 || chunk.length >= 16) sniffed = true
      }
      out.write(chunk)
    })
    req.on('end', () => {
      if (aborted) {
        resolve()
        return
      }
      out.end(() => resolve())
    })
    req.on('error', () => {
      aborted = true
      try { out.destroy() } catch { /* ignore */ }
      fsPromises.unlink(tempPath).catch(() => undefined)
      resolve()
    })
  })

  if (aborted) {
    if (!res.headersSent) res.status(413).json({ error: 'File too large' })
    return
  }

  // Server-side MIME sniff (read the temp file)
  let sniffedMime = 'application/octet-stream'
  try {
    const fd = await fsPromises.open(tempPath, 'r')
    const sniffBuffer = Buffer.alloc(16)
    await fd.read(sniffBuffer, 0, 16, 0)
    await fd.close()
    sniffedMime = sniffMime(sniffBuffer)
  } catch {
    await fsPromises.unlink(tempPath).catch(() => undefined)
    res.status(500).json({ error: 'Could not read uploaded file' })
    return
  }

  // Reject if declared MIME is not in allowlist
  const declaredBase = declaredMime.split(';')[0].trim()
  const sniffedBase = sniffedMime.split(';')[0].trim()
  // Accept if either matches an allowed type and the other is
  // either the same or octet-stream (a permissive client).
  const declaredOk = ALLOWED_MIME_TYPES.has(declaredBase)
  const sniffedOk = ALLOWED_MIME_TYPES.has(sniffedBase)
  if (!declaredOk && !sniffedOk) {
    await fsPromises.unlink(tempPath).catch(() => undefined)
    res.status(415).json({ error: 'File type not allowed' })
    return
  }
  // If both are specified and they disagree, reject (MIME
  // mismatch attack).
  if (declaredBase !== 'application/octet-stream' && sniffedBase !== 'application/octet-stream' &&
      declaredBase !== sniffedBase) {
    await fsPromises.unlink(tempPath).catch(() => undefined)
    res.status(415).json({ error: 'MIME type mismatch between declared and content' })
    return
  }
  const finalMime = sniffedOk ? sniffedBase : declaredBase

  // Quota check (re-verify with actual size)
  const stats = await fsPromises.stat(tempPath)
  const freshQuota = userQuota.get(userId) || 0
  if (freshQuota + stats.size > USER_QUOTA_LIMIT) {
    await fsPromises.unlink(tempPath).catch(() => undefined)
    res.status(413).json({ error: 'Quota exceeded' })
    return
  }

  // Rename temp → final
  await fsPromises.rename(tempPath, finalPath)

  const record: FileRecord = {
    id,
    name: safeName,
    size: stats.size,
    mimeType: finalMime,
    uploadedAt: new Date().toISOString(),
    userId,
  }

  fileRecords.set(id, record)
  userQuota.set(userId, freshQuota + stats.size)
  getEngine().appendActivity(
    'file.upload',
    true,
    JSON.stringify({
      fileId: id,
      name: sanitizeForLog(safeName),
      size: stats.size,
      mimeType: finalMime,
    }),
  )

  res.json({ success: true, file: record })
})

router.get('/files', (req, res) => {
  const records = Array.from(fileRecords.values())
    .filter(r => r.userId === req.user?.id || req.user?.role === 'admin')
  res.json({ files: records })
})

router.get('/files/:id', async (req, res) => {
  const record = fileRecords.get(req.params.id)
  if (!record) {
    res.status(404).json({ error: 'File not found' })
    return
  }
  if (record.userId !== req.user?.id && req.user?.role !== 'admin') {
    res.status(403).json({ error: 'Forbidden' })
    return
  }
  const filePath = join(UPLOAD_DIR, record.id)
  try {
    await fsPromises.access(filePath)
  } catch {
    res.status(404).json({ error: 'File data not found' })
    return
  }
  const data = await fsPromises.readFile(filePath)
  res.setHeader('Content-Type', record.mimeType)
  // Re-sanitize for the Content-Disposition header (defense
  // in depth — record.name was sanitized on upload, but be
  // paranoid)
  const safeName = sanitizeFilename(record.name)
  res.setHeader('Content-Disposition', `attachment; filename="${safeName.replace(/"/g, '')}"`)
  res.send(data)
})

router.delete('/files/:id', async (req, res) => {
  const record = fileRecords.get(req.params.id)
  if (!record) {
    res.status(404).json({ error: 'File not found' })
    return
  }
  if (record.userId !== req.user?.id && req.user?.role !== 'admin') {
    res.status(403).json({ error: 'Forbidden' })
    return
  }
  const filePath = join(UPLOAD_DIR, record.id)
  try { await fsPromises.unlink(filePath) } catch { /* ignore missing file */ }
  const freed = record.size
  fileRecords.delete(req.params.id)
  const currentQuota = userQuota.get(record.userId) || 0
  userQuota.set(record.userId, Math.max(0, currentQuota - freed))
  getEngine().appendActivity(
    'file.delete',
    true,
    JSON.stringify({ fileId: record.id, name: sanitizeForLog(record.name) }),
  )
  res.json({ success: true })
})

export default router
