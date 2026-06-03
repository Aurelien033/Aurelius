export interface UserRecord {
  id: string
  username: string
  role: 'admin' | 'user'
  apiKeys: string[]
  createdAt: string
}

export interface SessionRecord {
  token: string
  expiresAt: number
}

export interface BffAgentRecord {
  id: string
  name: string
  role: string
  capabilities: string[]
  state: 'idle' | 'active' | 'busy' | 'error' | 'terminated'
  created: number
  lastHeartbeat: number
  metrics: { tasksCompleted: number; avgLatencyMs: number; errorRate: number }
}

export interface FileRecord {
  id: string
  name: string
  size: number
  mimeType: string
  uploadedAt: string
  userId: string
}

export interface TraceStep {
  id: string
  type: 'thought' | 'tool_call' | 'tool_result' | 'action' | 'error' | 'observation'
  timestamp: number
  content: string
  metadata?: Record<string, unknown>
  duration?: number
}

export interface TraceRecord {
  id: string
  agentId: string
  agentName: string
  task: string
  status: 'running' | 'completed' | 'failed' | 'truncated'
  steps: TraceStep[]
  startedAt: number
  completedAt?: number
  totalDuration?: number
  stepCount: number
  tokenCount?: number
}

export interface CronTaskRecord {
  id: string
  name: string
  cron: string
  command: string
  enabled: boolean
  lastRun: string | null
  lastSuccess: boolean | null
  nextRun: string | null
  createdAt: string
}

export interface ConversationRecord {
  messages: Array<{ role: string; content: string }>
  agent: string
}

export interface BffStore {
  readonly persistent: boolean

  // Users & auth
  listUsers(): UserRecord[]
  getUser(id: string): UserRecord | undefined
  findUserByApiKey(apiKey: string): UserRecord | undefined
  findUserByUsername(username: string): UserRecord | undefined
  saveUser(user: UserRecord): void
  addUserApiKey(userId: string, apiKey: string): void
  removeUserApiKey(userId: string, prefix: string): string | undefined

  setSession(token: string, expiresAt: number): void
  deleteExpiredSessions(now: number): void

  markInviteUsed(token: string): void

  // BFF agents (route registry — distinct from data-engine agents)
  listBffAgents(): BffAgentRecord[]
  getBffAgent(id: string): BffAgentRecord | undefined
  saveBffAgent(agent: BffAgentRecord): void
  deleteBffAgent(id: string): void

  // Files
  listFiles(userId: string, isAdmin: boolean): FileRecord[]
  getFile(id: string): FileRecord | undefined
  saveFile(record: FileRecord): void
  deleteFile(id: string): void
  getUserQuota(userId: string): number
  addUserQuota(userId: string, delta: number): number

  // Traces
  listTraces(filters: { agentId?: string; status?: string; limit?: number }): TraceRecord[]
  getTrace(id: string): TraceRecord | undefined
  saveTrace(trace: TraceRecord): void
  trimTraces(maxCount: number): void

  // Scheduler
  listSchedulerTasks(): CronTaskRecord[]
  getSchedulerTask(id: string): CronTaskRecord | undefined
  saveSchedulerTask(task: CronTaskRecord): void
  deleteSchedulerTask(id: string): void

  // Chat
  getConversation(id: string): ConversationRecord | undefined
  saveConversation(id: string, conv: ConversationRecord): void
  listConversations(limit?: number): Array<{ id: string; conv: ConversationRecord }>
  trimConversations(maxCount: number): void

  close(): void
}
