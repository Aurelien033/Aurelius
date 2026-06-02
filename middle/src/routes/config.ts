import { Router } from 'express'
import { config as appConfig, normalizeChatBackend } from '../config.js'
import { getEngine } from '../engine.js'
import { requireScope } from '../middleware/auth.js'

const router = Router()

/**
 * H1 follow-on: protected config keys that must not be mutable
 * at runtime. The pre-remediation handler accepted any key and
 * forwarded it to engine.setConfig, including security-critical
 * values like chat.upstream_url, cors.allowed_origin, and the
 * service key. The fix rejects any mutation that touches these
 * keys, even with the `config:write` scope.
 */
const PROTECTED_CONFIG_KEYS = new Set([
  'chat.upstream_url',
  'chat.vllm_upstream_url',
  'chat.agentic_upstream_url',
  'aurelius.service_key',
  'aurelius.ws_token_secret',
  'aurelius.session_secret',
  'cors.allowed_origin',
  'ws.origin_allowlist',
  'auth.required',
  'auth.api_key',
  'aurelius.api_key',
  // H6 + H7 follow-on: the code-execution mode is a
  // security-critical control. Mutating it at runtime
  // would let an admin turn a fail-closed sandbox off
  // without going through the boot-time env var.
  'sandbox.execution_mode',
  'sandbox.isolated_backend',
  // H7 follow-on: the metrics key (a shared secret)
  // and rate-limit-disable flags must not be mutable.
  'metrics.key',
  'rate_limit.disabled',
  'rate_limit.disabled_keys',
  'network.allowed_origin',
  'network.cors_origin',
])

function isProtectedKey(key: string): boolean {
  return PROTECTED_CONFIG_KEYS.has(key) || key.endsWith('.api_key') || key.endsWith('.secret')
}

router.get('/', (_req, res) => {
  const engine = getEngine()
  const config = engine.getAllConfig()
  res.json({ config })
})

router.post('/', requireScope('config:write'), (req, res) => {
  const engine = getEngine()
  const { config } = req.body || {}
  if (!config || typeof config !== 'object') {
    res.status(400).json({ error: 'Config object required' })
    return
  }
  for (const [key, value] of Object.entries(config)) {
    if (isProtectedKey(key)) {
      res.status(403).json({ error: `Config key "${key}" is protected and cannot be modified at runtime` })
      return
    }
    if (key === 'chat.backend') {
      engine.setConfig(key, normalizeChatBackend(String(value), appConfig.defaultChatBackend))
    } else {
      engine.setConfig(key, String(value))
    }
  }
  engine.appendActivity('config.update', true, 'Configuration updated')
  res.json({ success: true, config: engine.getAllConfig() })
})

router.get('/:key', (req, res) => {
  const engine = getEngine()
  const key = String(req.params.key)
  if (isProtectedKey(key)) {
    res.status(403).json({ error: `Config key "${key}" is protected` })
    return
  }
  const value = engine.getConfig(key)
  if (value === null) {
    res.status(404).json({ error: 'Config key not found' })
    return
  }
  res.json({ key, value })
})

router.put('/:key', requireScope('config:write'), (req, res) => {
  const engine = getEngine()
  const key = String(req.params.key)
  if (isProtectedKey(key)) {
    res.status(403).json({ error: `Config key "${key}" is protected and cannot be modified at runtime` })
    return
  }
  const { value } = req.body || {}
  if (value === undefined) {
    res.status(400).json({ error: 'Value required' })
    return
  }
  if (key === 'chat.backend') {
    engine.setConfig(key, normalizeChatBackend(String(value), appConfig.defaultChatBackend))
  } else {
    engine.setConfig(key, String(value))
  }
  engine.appendActivity('config.update', true, JSON.stringify({ key }))
  res.json({ success: true })
})

export default router
