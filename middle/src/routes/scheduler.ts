import { Router } from 'express'
import { v4 as uuidv4 } from 'uuid'
import { requireScope } from '../middleware/auth.js'
import { sanitizeForLog } from '../security/validation.js'

/**
 * Typed command envelope. Pre-remediation, scheduled tasks
 * stored `{ name, cron, command: string }` and the scheduler
 * dispatched the raw string to /api/command. The fix replaces
 * the free-form string with a typed envelope (commandType +
 * typed params), records the approval metadata, enforces
 * max-runs, and enforces an expiry. The H3 fix also adds a
 * global emergency-disable switch.
 */

export type CommandType =
  | 'aurelius.query'
  | 'aurelius.summarize'
  | 'aurelius.notify'
  | 'aurelius.custom'  // requires admin approval

interface CommandEnvelope {
  type: CommandType
  params: Record<string, unknown>
  scope?: string
}

interface ScheduledTask {
  id: string
  name: string
  cron: string
  command: CommandEnvelope
  enabled: boolean
  lastRun: string | null
  lastSuccess: boolean | null
  runCount: number
  nextRun: string | null
  createdAt: string
  createdBy: string
  approvedBy: string | null
  approvedAt: string | null
  maxRuns: number
  expiresAt: string
}

const router = Router()
const tasks = new Map<string, ScheduledTask>()
const intervals = new Map<string, ReturnType<typeof setInterval>>()

// Global emergency disable. Defaults to true (normal operation).
// Setting AURELIUS_SCHEDULER_DISABLED=1 at boot disables all
// scheduling; AURELIUS_SCHEDULER_DISABLED=0 re-enables. The
// runtime toggle (POST /disable) requires admin scope.
let schedulerEnabled = process.env.AURELIUS_SCHEDULER_DISABLED !== '1'

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

function validateCommandEnvelope(input: unknown): CommandEnvelope | null {
  if (!input || typeof input !== 'object') return null
  const o = input as Record<string, unknown>
  const type = o.type
  if (typeof type !== 'string') return null
  const allowed: CommandType[] = ['aurelius.query', 'aurelius.summarize', 'aurelius.notify', 'aurelius.custom']
  if (!allowed.includes(type as CommandType)) return null
  const params = o.params
  if (!params || typeof params !== 'object' || Array.isArray(params)) return null
  return { type: type as CommandType, params: params as Record<string, unknown> }
}

function isExpired(task: ScheduledTask): boolean {
  return Date.now() > new Date(task.expiresAt).getTime()
}

function reachedMaxRuns(task: ScheduledTask): boolean {
  return task.maxRuns > 0 && task.runCount >= task.maxRuns
}

router.get('/', requireScope('scheduler:admin'), (_req, res) => {
  res.json({ tasks: Array.from(tasks.values()), enabled: schedulerEnabled })
})

router.post('/', requireScope('scheduler:admin'), (req, res) => {
  if (!schedulerEnabled) {
    res.status(503).json({ error: 'Scheduler is disabled (emergency stop)' })
    return
  }
  const { name, cron, command, maxRuns, expiresInHours, approve } = req.body || {}
  if (!name || !cron || !command) {
    res.status(400).json({ error: 'name, cron, and command (envelope) required' })
    return
  }
  const envelope = validateCommandEnvelope(command)
  if (!envelope) {
    res.status(400).json({
      error: 'command must be a typed envelope: { type: "aurelius.query|summarize|notify|custom", params: {...} }',
    })
    return
  }
  if (envelope.type === 'aurelius.custom' && !approve) {
    res.status(403).json({ error: 'aurelius.custom requires explicit approval (approve: true)' })
    return
  }
  const cap = typeof maxRuns === 'number' && maxRuns > 0 ? Math.min(maxRuns, 10000) : 100
  const ttl = typeof expiresInHours === 'number' && expiresInHours > 0
    ? Math.min(expiresInHours, 24 * 30) // cap at 30 days
    : 24
  const id = `task-${uuidv4()}`
  const task: ScheduledTask = {
    id,
    name: String(name).slice(0, 200),
    cron: String(cron).slice(0, 100),
    command: envelope,
    enabled: true,
    lastRun: null,
    lastSuccess: null,
    runCount: 0,
    nextRun: null,
    createdAt: new Date().toISOString(),
    createdBy: req.user?.id || 'unknown',
    approvedBy: envelope.type === 'aurelius.custom' ? (req.user?.id || 'unknown') : null,
    approvedAt: envelope.type === 'aurelius.custom' ? new Date().toISOString() : null,
    maxRuns: cap,
    expiresAt: new Date(Date.now() + ttl * 3600 * 1000).toISOString(),
  }

  tasks.set(id, task)
  scheduleTask(id, task)
  res.json({ success: true, task })
})

router.delete('/:id', requireScope('scheduler:admin'), (req, res) => {
  const id = String(req.params.id)
  clearInterval(intervals.get(id))
  intervals.delete(id)
  tasks.delete(id)
  res.json({ success: true })
})

router.post('/:id/toggle', requireScope('scheduler:admin'), (req, res) => {
  const id = String(req.params.id)
  const task = tasks.get(id)
  if (!task) {
    res.status(404).json({ error: 'Task not found' })
    return
  }
  task.enabled = !task.enabled
  if (task.enabled) scheduleTask(task.id, task)
  else clearInterval(intervals.get(task.id))
  res.json({ success: true, enabled: task.enabled })
})

// Emergency disable. Halts all scheduled tasks immediately.
router.post('/disable', requireScope('scheduler:admin'), (_req, res) => {
  schedulerEnabled = false
  for (const [id, interval] of intervals) {
    clearInterval(interval)
  }
  intervals.clear()
  for (const task of tasks.values()) {
    task.enabled = false
  }
  res.json({ success: true, enabled: false })
})

router.post('/enable', requireScope('scheduler:admin'), (_req, res) => {
  schedulerEnabled = true
  for (const task of tasks.values()) {
    if (task.enabled) scheduleTask(task.id, task)
  }
  res.json({ success: true, enabled: true })
})

function scheduleTask(id: string, task: ScheduledTask) {
  clearInterval(intervals.get(id))
  if (!schedulerEnabled) return
  if (isExpired(task) || reachedMaxRuns(task)) {
    task.enabled = false
    return
  }
  const ms = parseCron(task.cron)
  if (!ms || ms < 60000) return
  const interval = setInterval(async () => {
    if (!schedulerEnabled) return
    if (!task.enabled) return
    if (isExpired(task) || reachedMaxRuns(task)) {
      task.enabled = false
      clearInterval(interval)
      return
    }
    task.lastRun = new Date().toISOString()
    task.runCount++
    try {
      // The callback URL is a fixed loopback constant, NOT a
      // configurable port. The pre-remediation code used
      // process.env.MIDDLE_PORT, which an attacker who can
      // influence the env could redirect.
      const res = await fetch('http://127.0.0.1:3001/api/command', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          commandType: task.command.type,
          params: task.command.params,
          taskId: task.id,
        }),
      })
      const data = await res.json()
      task.lastSuccess = data.success === true
    } catch {
      task.lastSuccess = false
    }
  }, ms)
  intervals.set(id, interval)
}

export default router
