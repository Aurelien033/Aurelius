import type {
  BffAgentRecord,
  BffStore,
  ConversationRecord,
  CronTaskRecord,
  FileRecord,
  TraceRecord,
  UserRecord,
} from './types.js'

export class MemoryBffStore implements BffStore {
  readonly persistent = false

  private users = new Map<string, UserRecord>()
  private userSessions = new Map<string, { token: string; expiresAt: number }>()
  private inviteTokens = new Map<string, { role: 'admin' | 'user'; used: boolean }>()
  private bffAgents = new Map<string, BffAgentRecord>()
  private fileRecords = new Map<string, FileRecord>()
  private userQuota = new Map<string, number>()
  private traces = new Map<string, TraceRecord>()
  private schedulerTasks = new Map<string, CronTaskRecord>()
  private conversations = new Map<string, ConversationRecord>()

  listUsers(): UserRecord[] {
    return Array.from(this.users.values())
  }

  getUser(id: string): UserRecord | undefined {
    return this.users.get(id)
  }

  findUserByApiKey(apiKey: string): UserRecord | undefined {
    for (const user of this.users.values()) {
      if (user.apiKeys.includes(apiKey)) return user
    }
    return undefined
  }

  findUserByUsername(username: string): UserRecord | undefined {
    for (const user of this.users.values()) {
      if (user.username === username) return user
    }
    return undefined
  }

  saveUser(user: UserRecord): void {
    this.users.set(user.id, user)
  }

  addUserApiKey(userId: string, apiKey: string): void {
    const user = this.users.get(userId)
    if (user) user.apiKeys.push(apiKey)
  }

  removeUserApiKey(userId: string, prefix: string): string | undefined {
    const user = this.users.get(userId)
    if (!user) return undefined
    const idx = user.apiKeys.findIndex((k) => k.startsWith(prefix))
    if (idx === -1) return undefined
    return user.apiKeys.splice(idx, 1)[0]
  }

  setSession(token: string, expiresAt: number): void {
    this.userSessions.set(token, { token, expiresAt })
  }

  deleteExpiredSessions(now: number): void {
    for (const [token, session] of this.userSessions) {
      if (session.expiresAt < now) this.userSessions.delete(token)
    }
  }

  markInviteUsed(token: string): void {
    const invite = this.inviteTokens.get(token)
    if (invite) invite.used = true
  }

  listBffAgents(): BffAgentRecord[] {
    return Array.from(this.bffAgents.values())
  }

  getBffAgent(id: string): BffAgentRecord | undefined {
    return this.bffAgents.get(id)
  }

  saveBffAgent(agent: BffAgentRecord): void {
    this.bffAgents.set(agent.id, agent)
  }

  deleteBffAgent(id: string): void {
    this.bffAgents.delete(id)
  }

  listFiles(userId: string, isAdmin: boolean): FileRecord[] {
    return Array.from(this.fileRecords.values()).filter(
      (r) => isAdmin || r.userId === userId,
    )
  }

  getFile(id: string): FileRecord | undefined {
    return this.fileRecords.get(id)
  }

  saveFile(record: FileRecord): void {
    this.fileRecords.set(record.id, record)
  }

  deleteFile(id: string): void {
    this.fileRecords.delete(id)
  }

  getUserQuota(userId: string): number {
    return this.userQuota.get(userId) || 0
  }

  addUserQuota(userId: string, delta: number): number {
    const next = (this.userQuota.get(userId) || 0) + delta
    this.userQuota.set(userId, next)
    return next
  }

  listTraces(filters: { agentId?: string; status?: string; limit?: number }): TraceRecord[] {
    let result = Array.from(this.traces.values())
    if (filters.agentId) result = result.filter((t) => t.agentId === filters.agentId)
    if (filters.status) result = result.filter((t) => t.status === filters.status)
    result.sort((a, b) => b.startedAt - a.startedAt)
    return filters.limit ? result.slice(0, filters.limit) : result
  }

  getTrace(id: string): TraceRecord | undefined {
    return this.traces.get(id)
  }

  saveTrace(trace: TraceRecord): void {
    this.traces.set(trace.id, trace)
  }

  trimTraces(maxCount: number): void {
    if (this.traces.size <= maxCount) return
    const sorted = Array.from(this.traces.entries()).sort((a, b) => a[1].startedAt - b[1].startedAt)
    while (this.traces.size > maxCount && sorted.length > 0) {
      const [id] = sorted.shift()!
      this.traces.delete(id)
    }
  }

  listSchedulerTasks(): CronTaskRecord[] {
    return Array.from(this.schedulerTasks.values())
  }

  getSchedulerTask(id: string): CronTaskRecord | undefined {
    return this.schedulerTasks.get(id)
  }

  saveSchedulerTask(task: CronTaskRecord): void {
    this.schedulerTasks.set(task.id, task)
  }

  deleteSchedulerTask(id: string): void {
    this.schedulerTasks.delete(id)
  }

  getConversation(id: string): ConversationRecord | undefined {
    return this.conversations.get(id)
  }

  saveConversation(id: string, conv: ConversationRecord): void {
    this.conversations.set(id, conv)
  }

  listConversations(limit = 10000): Array<{ id: string; conv: ConversationRecord }> {
    return Array.from(this.conversations.entries())
      .slice(-limit)
      .map(([id, conv]) => ({ id, conv }))
  }

  trimConversations(maxCount: number): void {
    if (this.conversations.size <= maxCount) return
    const keys = Array.from(this.conversations.keys())
    while (this.conversations.size > maxCount && keys.length > 0) {
      this.conversations.delete(keys.shift()!)
    }
  }

  close(): void {
    // no-op
  }
}
