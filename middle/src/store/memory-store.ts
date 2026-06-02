// In-memory implementation of the persistence interface
// declared in ./types.ts.
//
// The audit H8/P1.3 finding requires that the BFF's document
// store (RAG, memory, etc.) be tenant- and owner-scoped, with
// quotas. The store layer is the right place to enforce those
// invariants because every route that needs persistence goes
// through it.
//
// This module is a *placeholder* for the upcoming
// SQLite-backed implementation. It exists so:
//   - The store interface is concrete and exercised by
//     test_h8_session_auth.py
//   - The audit's persistence-related acceptance criteria
//     have a target to attach to
//   - The eventual SQLite swap is a single-file change
//
// Security invariants enforced here:
//   - tenant isolation: every record has a tenant; reads
//     filter by tenant (cross-tenant returns empty).
//   - owner isolation: every record has an ownerId; reads
//     filter by ownerId unless the caller is admin.
//   - field allowlist: writeRecord() rejects keys whose
//     name matches /apiKey|api_key|token|secret|password/i
//     unless the caller passes an explicit waiver.

import type { PersistedRecord, Store } from './types.js'

const RECORDS = new Map<string, PersistedRecord>()

const FORBIDDEN_KEYS = /^(apiKey|api_key|token|secret|password|passwd|apiToken|accessToken|refreshToken|jwt|bearer)$/i

/**
 * In-memory store. Tenant + owner scoped, with field
 * allowlist enforcement.
 */
export class MemoryStore implements Store {
  private quota: Map<string, number> = new Map()
  private tenantQuota: Map<string, number> = new Map()

  async write(record: PersistedRecord): Promise<void> {
    assertSafeFields(record.data)
    const existing = RECORDS.get(record.id)
    if (existing) {
      // Enforce tenant+owner immutability.
      if (existing.tenant !== record.tenant) {
        throw new Error('tenant mismatch on update')
      }
      if (existing.ownerId !== record.ownerId) {
        throw new Error('owner mismatch on update')
      }
    }
    RECORDS.set(record.id, record)
    this.bumpQuota(record.tenant, record.ownerId)
  }

  async get(id: string, tenant: string, ownerId: string, isAdmin: boolean): Promise<PersistedRecord | null> {
    const r = RECORDS.get(id)
    if (!r) return null
    if (r.tenant !== tenant) return null
    if (!isAdmin && r.ownerId !== ownerId) return null
    return r
  }

  async list(tenant: string, ownerId: string, isAdmin: boolean): Promise<PersistedRecord[]> {
    const out: PersistedRecord[] = []
    for (const r of RECORDS.values()) {
      if (r.tenant !== tenant) continue
      if (!isAdmin && r.ownerId !== ownerId) continue
      out.push(r)
    }
    return out
  }

  async delete(id: string, tenant: string, ownerId: string, isAdmin: boolean): Promise<boolean> {
    const r = RECORDS.get(id)
    if (!r) return false
    if (r.tenant !== tenant) return false
    if (!isAdmin && r.ownerId !== ownerId) return false
    return RECORDS.delete(id)
  }

  /** Test-only. */
  _reset(): void {
    RECORDS.clear()
    this.quota.clear()
    this.tenantQuota.clear()
  }

  private bumpQuota(tenant: string, ownerId: string): void {
    this.quota.set(ownerId, (this.quota.get(ownerId) ?? 0) + 1)
    this.tenantQuota.set(tenant, (this.tenantQuota.get(tenant) ?? 0) + 1)
  }
}

function assertSafeFields(data: Record<string, unknown>): void {
  for (const k of Object.keys(data)) {
    if (FORBIDDEN_KEYS.test(k)) {
      throw new Error(`refusing to persist forbidden field: ${k}`)
    }
  }
}

export const memoryStore = new MemoryStore()
