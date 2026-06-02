import { Router, type Request, type Response } from 'express';
import { randomUUID as uuidv4 } from 'crypto';
import { getEngine } from '../engine.js';

const router = Router();

interface Document {
  id: string; // UUID v4
  filename: string;
  content: string;
  chunkCount: number;
  uploadedAt: number;
  source: string;
  tenant: string;
  ownerId: string;
  deleted: boolean; // tombstone for non-enumerable deletion
}

// Per-tenant storage. Top-level key is tenant, inner key is doc id.
// Replacing the process-global `documents` array with a per-tenant
// map closes the cross-tenant read vulnerability (H4) and makes
// per-tenant quotas cheap to enforce.
const tenantStores = new Map<string, Map<string, Document>>();
const MAX_DOCS_PER_TENANT = 500;
const MAX_DOC_BYTES = 5 * 1024 * 1024; // 5 MB

function getTenantStore(tenant: string): Map<string, Document> {
  let store = tenantStores.get(tenant);
  if (!store) {
    store = new Map();
    tenantStores.set(tenant, store);
  }
  return store;
}

function tenantFor(req: Request): string {
  // In a real deployment this would be derived from the auth
  // principal. For now, the auth middleware sets req.user.tenant.
  return req.user?.tenant || 'default';
}

function ownerIdFor(req: Request): string {
  return req.user?.id || 'anonymous';
}

function isAdmin(req: Request): boolean {
  return req.user?.role === 'admin';
}

function publicShape(d: Document) {
  return {
    id: d.id,
    filename: d.filename,
    chunkCount: d.chunkCount,
    uploadedAt: d.uploadedAt,
    source: d.source,
  };
}

router.get('/documents', (req: Request, res: Response) => {
  const tenant = tenantFor(req);
  const store = getTenantStore(tenant);
  const docs = Array.from(store.values())
    .filter((d) => !d.deleted)
    .map(publicShape);
  res.json({ documents: docs, total: docs.length });
});

router.get('/documents/:id', (req: Request, res: Response) => {
  const tenant = tenantFor(req);
  const store = getTenantStore(tenant);
  const doc = store.get(String(req.params.id));
  if (!doc || doc.deleted) return res.status(404).json({ error: 'Document not found' });
  res.json({ document: publicShape(doc) });
});

router.post('/documents', (req: Request, res: Response) => {
  const tenant = tenantFor(req);
  const ownerId = ownerIdFor(req);
  const { filename, content, source } = req.body || {};
  if (!filename || !content) {
    res.status(400).json({ error: 'filename and content required' });
    return;
  }
  if (typeof content !== 'string') {
    res.status(400).json({ error: 'content must be a string' });
    return;
  }
  if (content.length > MAX_DOC_BYTES) {
    res.status(413).json({ error: `Document exceeds ${MAX_DOC_BYTES} bytes` });
    return;
  }

  const store = getTenantStore(tenant);
  // Quota check (tombstones don't count; only live docs)
  const liveCount = Array.from(store.values()).filter((d) => !d.deleted).length;
  if (liveCount >= MAX_DOCS_PER_TENANT) {
    res.status(413).json({ error: 'Tenant document quota exceeded' });
    return;
  }

  const id = uuidv4();
  const chunks = content.match(/[\s\S]{1,512}/g) || [];
  const doc: Document = {
    id,
    filename: String(filename),
    content,
    chunkCount: chunks.length,
    uploadedAt: Date.now(),
    source: source || 'upload',
    tenant,
    ownerId,
    deleted: false,
  };

  store.set(id, doc);
  const engine = getEngine();
  engine.appendActivity(
    'rag.upload',
    true,
    JSON.stringify({ docId: id, tenant, ownerId, filename: String(filename), chunks: chunks.length }),
  );

  for (let i = 0; i < chunks.length; i++) {
    engine.addMemoryEntry('rag', `[${id}:${i}] ${chunks[i]}`);
  }

  res.status(201).json({ document: publicShape(doc) });
});

router.delete('/documents/:id', (req: Request, res: Response) => {
  const tenant = tenantFor(req);
  const ownerId = ownerIdFor(req);
  const store = getTenantStore(tenant);
  const doc = store.get(String(req.params.id));
  if (!doc || doc.deleted) return res.status(404).json({ error: 'Document not found' });
  if (doc.ownerId !== ownerId && !isAdmin(req)) {
    res.status(403).json({ error: 'Forbidden' });
    return;
  }
  if (!doc || doc.deleted) return res.status(404).json({ error: 'Document not found' });
  // Tombstone: mark deleted but keep the record for audit.
  // This prevents re-enumeration of recently-deleted IDs.
  doc.deleted = true;
  // Optional: actually evict after a grace period (out of scope
  // for this PR; tombstone is sufficient to close the read path)
  res.json({ ok: true });
});

router.get('/search', (req: Request, res: Response) => {
  const tenant = tenantFor(req);
  const ownerId = ownerIdFor(req);
  const query = String(req.query.q || '');
  if (!query) {
    res.status(400).json({ error: 'query parameter q required' });
    return;
  }
  if (query.length > 1024) {
    res.status(400).json({ error: 'query too long' });
    return;
  }

  const store = getTenantStore(tenant);
  const results = Array.from(store.values())
    .filter((d) => !d.deleted)
    .filter((d) => d.ownerId === ownerId || isAdmin(req))
    .map((d) => {
      const idx = d.content.toLowerCase().indexOf(query.toLowerCase());
      if (idx === -1) return null;
      const start = Math.max(0, idx - 100);
      const end = Math.min(d.content.length, idx + query.length + 100);
      return {
        documentId: d.id,
        filename: d.filename,
        snippet: d.content.slice(start, end),
        relevance: 1.0,
      };
    })
    .filter(Boolean);

  res.json({ query, results, total: results.length });
});

export default router;
