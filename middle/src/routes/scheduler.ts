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
}

const ALLOWED_COMMANDS = /^(status|health|ping|list-agents|list-skills|list-memory)$/i
const router = Router()
const tasks = new Map<string, CronTask>()
const intervals = new Map<string, ReturnType<typeof setInterval>>()

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

router.get('/', (_req, res) => {
  res.json({ tasks: Array.from(tasks.values()) })
})

router.post('/', requireScope('scheduler:admin'), (req, res) => {
  const { name, cron, command } = req.body || {}
  if (!name || !cron || !command) {
    res.status(400).json({ error: 'Name, cron, and command required' })
    return
  }
  if (!ALLOWED_COMMANDS.test(command)) {
    res.status(400).json({ error: `Command not allowed: ${command}. Allowed: status, health, ping, list-agents, list-skills, list-memory` })
    return
  }

  const id = `task-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`
  const task: CronTask = {
    id, name, cron, command, enabled: true,
    lastRun: null, lastSuccess: null, nextRun: null,
    createdAt: new Date().toISOString(),
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

function scheduleTask(id: string, task: CronTask) {
  clearInterval(intervals.get(id))
  const ms = parseCron(task.cron)
  if (!ms || ms < 60000) return
  const interval = setInterval(() => {
    if (!task.enabled) return
    task.lastRun = new Date().toISOString()
    task.lastSuccess = true
  }, ms)
  intervals.set(id, interval)
}

export default router
