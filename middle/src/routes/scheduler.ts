import { Router } from 'express'
import { requireScope } from '../middleware/auth.js'

interface CronTask {
  id: string
  name: string
  cron: string
  command: string
  enabled: boolean
  lastRun: string | null
  lastSuccess: boolean | null
  nextRun: string | null
  createdAt: string
  createdBy: string
  updatedAt: string
  lastRunAt: string | null
  runCount: number
}

// ── Scheduler persistence store ───────────────────────────────────────────────

export interface SchedulerStore {
  load(): Promise<CronTask[]>
  save(tasks: CronTask[]): Promise<void>
  list(): Promise<CronTask[]>
  upsert(task: CronTask): Promise<void>
  delete(id: string): Promise<void>
}

export class MemorySchedulerStore implements SchedulerStore {
  private _tasks = new Map<string, CronTask>()

  async load(): Promise<CronTask[]> {
    return Array.from(this._tasks.values())
  }

  async save(tasks: CronTask[]): Promise<void> {
    this._tasks.clear()
    for (const t of tasks) this._tasks.set(t.id, t)
  }

  async list(): Promise<CronTask[]> {
    return Array.from(this._tasks.values())
  }

  async upsert(task: CronTask): Promise<void> {
    this._tasks.set(task.id, task)
  }

  async delete(id: string): Promise<void> {
    this._tasks.delete(id)
  }
}

// SqliteSchedulerStore is a named export for optional persistence.
// When DATABASE_URL is set, callers should instantiate this and pass it to the router.
export class SqliteSchedulerStore implements SchedulerStore {
  constructor(private readonly _url: string) {
    if (!_url) throw new Error('SqliteSchedulerStore requires a DATABASE_URL')
  }

  async load(): Promise<CronTask[]> {
    // Stub: real implementation would query sqlite/postgres via DATABASE_URL.
    console.warn('[scheduler] SqliteSchedulerStore.load() — stub, returning empty')
    return []
  }

  async save(tasks: CronTask[]): Promise<void> {
    console.warn('[scheduler] SqliteSchedulerStore.save() — stub, noop', tasks.length)
  }

  async list(): Promise<CronTask[]> { return this.load() }

  async upsert(task: CronTask): Promise<void> {
    console.warn('[scheduler] SqliteSchedulerStore.upsert() — stub', task.id)
  }

  async delete(id: string): Promise<void> {
    console.warn('[scheduler] SqliteSchedulerStore.delete() — stub', id)
  }
}

// ── Router factory ────────────────────────────────────────────────────────────

export function createSchedulerRouter(store: SchedulerStore = new MemorySchedulerStore()): Router {
  const router = Router()
  const tasks = new Map<string, CronTask>()
  const intervals = new Map<string, ReturnType<typeof setInterval>>()

  if (!process.env.DATABASE_URL) {
    console.warn('[scheduler] DATABASE_URL not set — scheduler state is ephemeral (in-memory only)')
  }

  // Reconcile in-memory state from the store on startup.
  store.load().then((persisted) => {
    for (const task of persisted) {
      tasks.set(task.id, task)
      if (task.enabled) scheduleTask(task.id, task)
    }
  }).catch((err) => {
    console.error('[scheduler] failed to load persisted tasks:', err)
  })

  function parseCron(expr: string): number | null {
    const parts = expr.trim().split(/\s+/)
    if (parts.length !== 5) return null
    const every = parts[0] === '*' && parts[1] === '*' && parts[2] === '*' && parts[3] === '*' && parts[4] === '*'
    if (every) return 60000
    const minutes = parseInt(parts[0], 10)
    if (!isNaN(minutes) && minutes > 0 && parts[1] === '*' && parts[2] === '*' && parts[3] === '*' && parts[4] === '*') {
      return minutes * 60000
    }
    const hours = parseInt(parts[1], 10)
    if (parts[0] === '*' && !isNaN(hours) && hours > 0 && parts[2] === '*' && parts[3] === '*' && parts[4] === '*') {
      return hours * 3600000
    }
    return null
  }

  function scheduleTask(id: string, task: CronTask) {
    clearInterval(intervals.get(id))
    const ms = parseCron(task.cron)
    if (!ms || ms < 60000) return
    const interval = setInterval(async () => {
      if (!task.enabled) return
      task.lastRun = new Date().toISOString()
      task.lastRunAt = task.lastRun
      task.runCount = (task.runCount || 0) + 1
      task.updatedAt = task.lastRun
      try {
        const res = await fetch(`http://localhost:${process.env.MIDDLE_PORT || 3001}/api/command`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ command: task.command }),
        })
        const data = await res.json()
        task.lastSuccess = data.success
      } catch {
        task.lastSuccess = false
      }
      await store.upsert(task)
    }, ms)
    intervals.set(id, interval)
  }

  router.get('/', (_req, res) => {
    res.json({ tasks: Array.from(tasks.values()) })
  })

  router.post('/', requireScope('scheduler:admin'), async (req, res) => {
    const { name, cron, command } = req.body || {}
    if (!name || !cron || !command) {
      res.status(400).json({ error: 'Name, cron, and command required' })
      return
    }

    const id = `task-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`
    const now = new Date().toISOString()
    const task: CronTask = {
      id, name, cron, command, enabled: true,
      lastRun: null, lastSuccess: null, nextRun: null,
      createdAt: now, createdBy: req.user?.id || 'unknown',
      updatedAt: now, lastRunAt: null, runCount: 0,
    }

    tasks.set(id, task)
    await store.upsert(task)
    scheduleTask(id, task)
    res.json({ success: true, task })
  })

  router.delete('/:id', requireScope('scheduler:admin'), async (req, res) => {
    const id = String(req.params.id)
    clearInterval(intervals.get(id))
    intervals.delete(id)
    tasks.delete(id)
    await store.delete(id)
    res.json({ success: true })
  })

  router.post('/:id/toggle', requireScope('scheduler:admin'), async (req, res) => {
    const id = String(req.params.id)
    const task = tasks.get(id)
    if (!task) {
      res.status(404).json({ error: 'Task not found' })
      return
    }
    task.enabled = !task.enabled
    task.updatedAt = new Date().toISOString()
    if (task.enabled) scheduleTask(task.id, task)
    else clearInterval(intervals.get(task.id))
    await store.upsert(task)
    res.json({ success: true, enabled: task.enabled })
  })

  // Replay endpoint: reload from store and reconcile in-memory state.
  // Used for testing persistence survival across simulated restarts.
  router.post('/replay', requireScope('scheduler:admin'), async (_req, res) => {
    for (const [, interval] of intervals) clearInterval(interval)
    intervals.clear()
    tasks.clear()
    const persisted = await store.load()
    for (const task of persisted) {
      tasks.set(task.id, task)
      if (task.enabled) scheduleTask(task.id, task)
    }
    res.json({ success: true, reloaded: persisted.length })
  })

  return router
}

// Default export uses in-memory store.
export default createSchedulerRouter()
