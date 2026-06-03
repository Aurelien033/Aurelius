import Database from 'better-sqlite3'
import { mkdirSync } from 'fs'
import { dirname } from 'path'
import type {
  BffAgentRecord,
  BffStore,
  ConversationRecord,
  CronTaskRecord,
  FileRecord,
  TraceRecord,
  TraceStep,
  UserRecord,
} from './types.js'

const SCHEMA = `
CREATE TABLE IF NOT EXISTS users (
  id TEXT PRIMARY KEY,
  username TEXT NOT NULL UNIQUE,
  role TEXT NOT NULL,
  api_keys_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS user_sessions (
  token TEXT PRIMARY KEY,
  expires_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS invite_tokens (
  token TEXT PRIMARY KEY,
  role TEXT NOT NULL,
  used INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS bff_agents (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  role TEXT NOT NULL,
  capabilities_json TEXT NOT NULL,
  state TEXT NOT NULL,
  created INTEGER NOT NULL,
  last_heartbeat INTEGER NOT NULL,
  metrics_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS files (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  size INTEGER NOT NULL,
  mime_type TEXT NOT NULL,
  uploaded_at TEXT NOT NULL,
  user_id TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS user_quota (
  user_id TEXT PRIMARY KEY,
  bytes_used INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS traces (
  id TEXT PRIMARY KEY,
  agent_id TEXT NOT NULL,
  agent_name TEXT NOT NULL,
  task TEXT NOT NULL,
  status TEXT NOT NULL,
  steps_json TEXT NOT NULL,
  started_at INTEGER NOT NULL,
  completed_at INTEGER,
  total_duration INTEGER,
  token_count INTEGER
);
CREATE TABLE IF NOT EXISTS scheduler_tasks (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  cron TEXT NOT NULL,
  command TEXT NOT NULL,
  enabled INTEGER NOT NULL,
  last_run TEXT,
  last_success INTEGER,
  next_run TEXT,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chat_conversations (
  id TEXT PRIMARY KEY,
  agent TEXT NOT NULL,
  messages_json TEXT NOT NULL
);
`

function parseUser(row: {
  id: string
  username: string
  role: string
  api_keys_json: string
  created_at: string
}): UserRecord {
  return {
    id: row.id,
    username: row.username,
    role: row.role as UserRecord['role'],
    apiKeys: JSON.parse(row.api_keys_json) as string[],
    createdAt: row.created_at,
  }
}

export class SqliteBffStore implements BffStore {
  readonly persistent = true
  private readonly db: Database.Database

  constructor(dbPath: string) {
    mkdirSync(dirname(dbPath), { recursive: true })
    this.db = new Database(dbPath)
    this.db.pragma('journal_mode = WAL')
    this.db.exec(SCHEMA)
  }

  listUsers(): UserRecord[] {
    const rows = this.db.prepare('SELECT * FROM users').all() as Array<{
      id: string
      username: string
      role: string
      api_keys_json: string
      created_at: string
    }>
    return rows.map(parseUser)
  }

  getUser(id: string): UserRecord | undefined {
    const row = this.db.prepare('SELECT * FROM users WHERE id = ?').get(id) as
      | {
          id: string
          username: string
          role: string
          api_keys_json: string
          created_at: string
        }
      | undefined
    return row ? parseUser(row) : undefined
  }

  findUserByApiKey(apiKey: string): UserRecord | undefined {
    for (const user of this.listUsers()) {
      if (user.apiKeys.includes(apiKey)) return user
    }
    return undefined
  }

  findUserByUsername(username: string): UserRecord | undefined {
    const row = this.db.prepare('SELECT * FROM users WHERE username = ?').get(username) as
      | {
          id: string
          username: string
          role: string
          api_keys_json: string
          created_at: string
        }
      | undefined
    return row ? parseUser(row) : undefined
  }

  saveUser(user: UserRecord): void {
    this.db
      .prepare(
        `INSERT INTO users (id, username, role, api_keys_json, created_at)
         VALUES (?, ?, ?, ?, ?)
         ON CONFLICT(id) DO UPDATE SET
           username = excluded.username,
           role = excluded.role,
           api_keys_json = excluded.api_keys_json,
           created_at = excluded.created_at`,
      )
      .run(user.id, user.username, user.role, JSON.stringify(user.apiKeys), user.createdAt)
  }

  addUserApiKey(userId: string, apiKey: string): void {
    const user = this.getUser(userId)
    if (!user) return
    if (!user.apiKeys.includes(apiKey)) {
      user.apiKeys.push(apiKey)
      this.saveUser(user)
    }
  }

  removeUserApiKey(userId: string, prefix: string): string | undefined {
    const user = this.getUser(userId)
    if (!user) return undefined
    const idx = user.apiKeys.findIndex((k) => k.startsWith(prefix))
    if (idx === -1) return undefined
    const removed = user.apiKeys.splice(idx, 1)[0]
    this.saveUser(user)
    return removed
  }

  setSession(token: string, expiresAt: number): void {
    this.db
      .prepare(
        `INSERT INTO user_sessions (token, expires_at) VALUES (?, ?)
         ON CONFLICT(token) DO UPDATE SET expires_at = excluded.expires_at`,
      )
      .run(token, expiresAt)
  }

  deleteExpiredSessions(now: number): void {
    this.db.prepare('DELETE FROM user_sessions WHERE expires_at < ?').run(now)
  }

  markInviteUsed(token: string): void {
    this.db.prepare('UPDATE invite_tokens SET used = 1 WHERE token = ?').run(token)
  }

  listBffAgents(): BffAgentRecord[] {
    const rows = this.db.prepare('SELECT * FROM bff_agents').all() as Array<{
      id: string
      name: string
      role: string
      capabilities_json: string
      state: string
      created: number
      last_heartbeat: number
      metrics_json: string
    }>
    return rows.map((row) => ({
      id: row.id,
      name: row.name,
      role: row.role,
      capabilities: JSON.parse(row.capabilities_json) as string[],
      state: row.state as BffAgentRecord['state'],
      created: row.created,
      lastHeartbeat: row.last_heartbeat,
      metrics: JSON.parse(row.metrics_json) as BffAgentRecord['metrics'],
    }))
  }

  getBffAgent(id: string): BffAgentRecord | undefined {
    return this.listBffAgents().find((a) => a.id === id)
  }

  saveBffAgent(agent: BffAgentRecord): void {
    this.db
      .prepare(
        `INSERT INTO bff_agents (
          id, name, role, capabilities_json, state, created, last_heartbeat, metrics_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
          name = excluded.name,
          role = excluded.role,
          capabilities_json = excluded.capabilities_json,
          state = excluded.state,
          last_heartbeat = excluded.last_heartbeat,
          metrics_json = excluded.metrics_json`,
      )
      .run(
        agent.id,
        agent.name,
        agent.role,
        JSON.stringify(agent.capabilities),
        agent.state,
        agent.created,
        agent.lastHeartbeat,
        JSON.stringify(agent.metrics),
      )
  }

  deleteBffAgent(id: string): void {
    this.db.prepare('DELETE FROM bff_agents WHERE id = ?').run(id)
  }

  listFiles(userId: string, isAdmin: boolean): FileRecord[] {
    const rows = isAdmin
      ? (this.db.prepare('SELECT * FROM files').all() as Array<Record<string, unknown>>)
      : (this.db.prepare('SELECT * FROM files WHERE user_id = ?').all(userId) as Array<
          Record<string, unknown>
        >)
    return rows.map((row) => ({
      id: String(row.id),
      name: String(row.name),
      size: Number(row.size),
      mimeType: String(row.mime_type),
      uploadedAt: String(row.uploaded_at),
      userId: String(row.user_id),
    }))
  }

  getFile(id: string): FileRecord | undefined {
    const row = this.db.prepare('SELECT * FROM files WHERE id = ?').get(id) as
      | Record<string, unknown>
      | undefined
    if (!row) return undefined
    return {
      id: String(row.id),
      name: String(row.name),
      size: Number(row.size),
      mimeType: String(row.mime_type),
      uploadedAt: String(row.uploaded_at),
      userId: String(row.user_id),
    }
  }

  saveFile(record: FileRecord): void {
    this.db
      .prepare(
        `INSERT INTO files (id, name, size, mime_type, uploaded_at, user_id)
         VALUES (?, ?, ?, ?, ?, ?)
         ON CONFLICT(id) DO UPDATE SET
           name = excluded.name,
           size = excluded.size,
           mime_type = excluded.mime_type,
           uploaded_at = excluded.uploaded_at,
           user_id = excluded.user_id`,
      )
      .run(record.id, record.name, record.size, record.mimeType, record.uploadedAt, record.userId)
  }

  deleteFile(id: string): void {
    this.db.prepare('DELETE FROM files WHERE id = ?').run(id)
  }

  getUserQuota(userId: string): number {
    const row = this.db.prepare('SELECT bytes_used FROM user_quota WHERE user_id = ?').get(userId) as
      | { bytes_used: number }
      | undefined
    return row?.bytes_used ?? 0
  }

  addUserQuota(userId: string, delta: number): number {
    const current = this.getUserQuota(userId)
    const next = current + delta
    this.db
      .prepare(
        `INSERT INTO user_quota (user_id, bytes_used) VALUES (?, ?)
         ON CONFLICT(user_id) DO UPDATE SET bytes_used = excluded.bytes_used`,
      )
      .run(userId, next)
    return next
  }

  listTraces(filters: { agentId?: string; status?: string; limit?: number }): TraceRecord[] {
    let sql = 'SELECT * FROM traces WHERE 1=1'
    const params: Array<string | number> = []
    if (filters.agentId) {
      sql += ' AND agent_id = ?'
      params.push(filters.agentId)
    }
    if (filters.status) {
      sql += ' AND status = ?'
      params.push(filters.status)
    }
    sql += ' ORDER BY started_at DESC'
    if (filters.limit) {
      sql += ' LIMIT ?'
      params.push(filters.limit)
    }
    const rows = this.db.prepare(sql).all(...params) as Array<Record<string, unknown>>
    return rows.map((row) => this.rowToTrace(row))
  }

  getTrace(id: string): TraceRecord | undefined {
    const row = this.db.prepare('SELECT * FROM traces WHERE id = ?').get(id) as
      | Record<string, unknown>
      | undefined
    return row ? this.rowToTrace(row) : undefined
  }

  saveTrace(trace: TraceRecord): void {
    this.db
      .prepare(
        `INSERT INTO traces (
          id, agent_id, agent_name, task, status, steps_json, started_at,
          completed_at, total_duration, token_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
          agent_id = excluded.agent_id,
          agent_name = excluded.agent_name,
          task = excluded.task,
          status = excluded.status,
          steps_json = excluded.steps_json,
          started_at = excluded.started_at,
          completed_at = excluded.completed_at,
          total_duration = excluded.total_duration,
          token_count = excluded.token_count`,
      )
      .run(
        trace.id,
        trace.agentId,
        trace.agentName,
        trace.task,
        trace.status,
        JSON.stringify(trace.steps),
        trace.startedAt,
        trace.completedAt ?? null,
        trace.totalDuration ?? null,
        trace.tokenCount ?? null,
      )
  }

  trimTraces(maxCount: number): void {
    const count = this.db.prepare('SELECT COUNT(*) AS c FROM traces').get() as { c: number }
    if (count.c <= maxCount) return
    const excess = count.c - maxCount
    this.db
      .prepare(
        `DELETE FROM traces WHERE id IN (
          SELECT id FROM traces ORDER BY started_at ASC LIMIT ?
        )`,
      )
      .run(excess)
  }

  listSchedulerTasks(): CronTaskRecord[] {
    const rows = this.db.prepare('SELECT * FROM scheduler_tasks').all() as Array<
      Record<string, unknown>
    >
    return rows.map((row) => ({
      id: String(row.id),
      name: String(row.name),
      cron: String(row.cron),
      command: String(row.command),
      enabled: Boolean(row.enabled),
      lastRun: row.last_run ? String(row.last_run) : null,
      lastSuccess: row.last_success === null || row.last_success === undefined
        ? null
        : Boolean(row.last_success),
      nextRun: row.next_run ? String(row.next_run) : null,
      createdAt: String(row.created_at),
    }))
  }

  getSchedulerTask(id: string): CronTaskRecord | undefined {
    return this.listSchedulerTasks().find((t) => t.id === id)
  }

  saveSchedulerTask(task: CronTaskRecord): void {
    this.db
      .prepare(
        `INSERT INTO scheduler_tasks (
          id, name, cron, command, enabled, last_run, last_success, next_run, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
          name = excluded.name,
          cron = excluded.cron,
          command = excluded.command,
          enabled = excluded.enabled,
          last_run = excluded.last_run,
          last_success = excluded.last_success,
          next_run = excluded.next_run`,
      )
      .run(
        task.id,
        task.name,
        task.cron,
        task.command,
        task.enabled ? 1 : 0,
        task.lastRun,
        task.lastSuccess === null ? null : task.lastSuccess ? 1 : 0,
        task.nextRun,
        task.createdAt,
      )
  }

  deleteSchedulerTask(id: string): void {
    this.db.prepare('DELETE FROM scheduler_tasks WHERE id = ?').run(id)
  }

  getConversation(id: string): ConversationRecord | undefined {
    const row = this.db.prepare('SELECT * FROM chat_conversations WHERE id = ?').get(id) as
      | { agent: string; messages_json: string }
      | undefined
    if (!row) return undefined
    return {
      agent: row.agent,
      messages: JSON.parse(row.messages_json) as ConversationRecord['messages'],
    }
  }

  saveConversation(id: string, conv: ConversationRecord): void {
    this.db
      .prepare(
        `INSERT INTO chat_conversations (id, agent, messages_json)
         VALUES (?, ?, ?)
         ON CONFLICT(id) DO UPDATE SET
           agent = excluded.agent,
           messages_json = excluded.messages_json`,
      )
      .run(id, conv.agent, JSON.stringify(conv.messages))
  }

  listConversations(limit = 10000): Array<{ id: string; conv: ConversationRecord }> {
    const rows = this.db
      .prepare('SELECT id, agent, messages_json FROM chat_conversations ORDER BY rowid DESC LIMIT ?')
      .all(limit) as Array<{ id: string; agent: string; messages_json: string }>
    return rows.map((row) => ({
      id: row.id,
      conv: {
        agent: row.agent,
        messages: JSON.parse(row.messages_json) as ConversationRecord['messages'],
      },
    }))
  }

  trimConversations(maxCount: number): void {
    const count = this.db.prepare('SELECT COUNT(*) AS c FROM chat_conversations').get() as {
      c: number
    }
    if (count.c <= maxCount) return
    const excess = count.c - maxCount
    this.db
      .prepare(
        `DELETE FROM chat_conversations WHERE id IN (
          SELECT id FROM chat_conversations ORDER BY rowid ASC LIMIT ?
        )`,
      )
      .run(excess)
  }

  close(): void {
    this.db.close()
  }

  private rowToTrace(row: Record<string, unknown>): TraceRecord {
    const steps = JSON.parse(String(row.steps_json)) as TraceStep[]
    return {
      id: String(row.id),
      agentId: String(row.agent_id),
      agentName: String(row.agent_name),
      task: String(row.task),
      status: String(row.status) as TraceRecord['status'],
      steps,
      startedAt: Number(row.started_at),
      completedAt: row.completed_at != null ? Number(row.completed_at) : undefined,
      totalDuration: row.total_duration != null ? Number(row.total_duration) : undefined,
      stepCount: steps.length,
      tokenCount: row.token_count != null ? Number(row.token_count) : undefined,
    }
  }
}
