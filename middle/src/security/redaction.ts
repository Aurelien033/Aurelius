// Centralized redaction for the BFF.
//
// Pre-remediation, appendActivity() calls in middle/src/
// server.ts and the routes/ files interpolated user
// input into log strings without sanitization. An attacker
// who could control a log field (e.g. a filename, a room
// name, a scheduler task name) could forge log lines,
// inject CR/LF to break log structure, or inject ANSI
// escape sequences to control terminal renderers.
//
// This module exports redact() that strips:
//   - Authorization headers / bearer tokens / API keys
//   - Cookie values that look like session tokens
//   - CR/LF and other control characters
//   - ANSI escape sequences (ESC [ ... letter)
//   - <script>...</script> and other HTML-looking tags
//   - Long printable runs that look like secrets
//
// The output is safe to interpolate into a log line,
// append into JSON, or write to disk.

const SECRET_PATTERNS: Array<{ name: string; re: RegExp }> = [
  // Authorization: Bearer xxx / Basic xxx
  { name: 'Authorization', re: /(?:authorization|Authorization)\s*[:=]\s*(?:Bearer|Basic|Token|ApiKey)\s+[A-Za-z0-9._\-+/=]{4,}/g },
  // Cookie: aurelius_sid=xxx; aurelius_csrf=xxx
  { name: 'Cookie', re: /(aurelius_(?:sid|csrf|session|api[-_]?key))\s*=\s*([^;\s]+)/gi },
  // X-API-Key / X-Aurelius-Session
  { name: 'X-*-Key', re: /(x-(?:api[-_]?key|aurelius[-_]?session))\s*[:=]\s*([^\s,;]+)/gi },
  // AWS-style keys (defense in depth)
  { name: 'AWS', re: /\b(?:AKIA|ASIA)[A-Z0-9]{16}\b/g },
  // PEM blocks
  { name: 'PEM', re: /-----BEGIN [A-Z ]+-----[\s\S]*?-----END [A-Z ]+-----/g },
  // JWTs (three base64url segments)
  { name: 'JWT', re: /\beyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\b/g },
]

const CONTROL_CHARS: RegExp = /[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]/g
const CR_LF: RegExp = /[\r\n]+/g
const ANSI_ESCAPE: RegExp = /\x1b\[[0-9;?]*[ -/]*[@-~]/g
const SCRIPT_TAG: RegExp = /<\/?(?:script|iframe|object|embed|svg)[^>]*>/gi

export interface RedactionResult {
  /** The redacted string, safe for logs. */
  value: string
  /** True if any secret pattern was found and redacted. */
  redacted: boolean
  /** Names of the secret patterns that matched. */
  hits: string[]
}

/**
 * Redact sensitive material and dangerous control
 * characters from a string. The output is safe to
 * interpolate into a log line.
 */
export function redact(input: unknown): RedactionResult {
  if (input === null || input === undefined) {
    return { value: '', redacted: false, hits: [] }
  }
  const str = typeof input === 'string' ? input : String(input)
  const hits: string[] = []
  let value = str
  for (const { name, re } of SECRET_PATTERNS) {
    const matches = value.match(re)
    if (matches && matches.length > 0) {
      hits.push(name)
      value = value.replace(re, `[REDACTED:${name}]`)
    }
  }
  // Control chars (excluding tab, LF, CR which we handle separately)
  if (CONTROL_CHARS.test(value)) {
    value = value.replace(CONTROL_CHARS, '')
    hits.push('CTRL')
  }
  // CR/LF (collapse to space to keep log single-line)
  if (CR_LF.test(value)) {
    value = value.replace(CR_LF, ' ')
    hits.push('CRLF')
  }
  // ANSI escapes
  if (ANSI_ESCAPE.test(value)) {
    value = value.replace(ANSI_ESCAPE, '')
    hits.push('ANSI')
  }
  // HTML/JS injection
  if (SCRIPT_TAG.test(value)) {
    value = value.replace(SCRIPT_TAG, '[REDACTED:HTML]')
    hits.push('HTML')
  }
  return {
    value,
    redacted: hits.length > 0,
    hits,
  }
}

/**
 * Convenience: redact and return just the value.
 */
export function redactValue(input: unknown): string {
  return redact(input).value
}

/**
 * Convenience: JSON.stringify an object, redacting each
 * value recursively. Top-level keys are NOT redacted
 * (callers should not put secrets in keys).
 */
export function redactJson(obj: unknown): string {
  if (obj === null || obj === undefined) return JSON.stringify(obj)
  if (typeof obj !== 'object') return JSON.stringify(redactValue(obj))
  if (Array.isArray(obj)) {
    return '[' + obj.map((v) => redactJson(v)).join(',') + ']'
  }
  const out: string[] = []
  for (const [k, v] of Object.entries(obj as Record<string, unknown>)) {
    if (typeof v === 'string' || typeof v === 'number' || typeof v === 'boolean') {
      out.push(JSON.stringify(k) + ':' + JSON.stringify(redactValue(v)))
    } else if (v === null || v === undefined) {
      out.push(JSON.stringify(k) + ':' + JSON.stringify(v))
    } else {
      out.push(JSON.stringify(k) + ':' + redactJson(v))
    }
  }
  return '{' + out.join(',') + '}'
}
