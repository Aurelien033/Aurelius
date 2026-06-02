import { Router, type Request, type Response, type NextFunction } from 'express'
import { config } from '../config.js'

const router = Router()

/**
 * C4: /api/models is user-triggered (called by the frontend with the
 * user's session). The pre-remediation handler always forwarded
 * `config.serviceApiKey`, collapsing every user into a single shared
 * upstream identity. The fix forwards the caller's own Authorization
 * header to upstream; the service principal key is reserved for the
 * `/internal/models` route mounted at the app root (which is
 * admin-only via requireAdmin).
 */
function forwardCallerAuth(req: Request): Record<string, string> {
  const headers: Record<string, string> = { 'Content-Type': 'application/json' }
  const auth = req.headers['authorization']
  if (typeof auth === 'string' && auth.length > 0) {
    headers['Authorization'] = auth
  }
  // X-API-Key is the legacy header; the BFF now issues a session
  // bearer, so we no longer echo the raw API key upstream.
  return headers
}

router.get('/', async (req, res) => {
  const headers = forwardCallerAuth(req)
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), 10000)
  try {
    const upstreamRes = await fetch(`${config.upstreamUrl}/v1/models`, { headers, signal: controller.signal })
    clearTimeout(timer)
    const data = await upstreamRes.json()
    res.status(upstreamRes.status).json(data)
  } catch (error: unknown) {
    clearTimeout(timer)
    const message = error instanceof Error ? error.message : 'Upstream unavailable'
    res.status(502).json({ error: 'Upstream unavailable', message })
  }
})

export default router

/**
 * Internal-only models route. Mounted by server.ts behind
 * requireAdmin. Uses the service principal key because the caller
 * is the operator, not a user.
 */
export const internalModelsRouter = Router()

internalModelsRouter.get('/', (_req: Request, res: Response, next: NextFunction) => {
  const headers: Record<string, string> = { 'Content-Type': 'application/json' }
  if (config.serviceApiKey) {
    headers['Authorization'] = `Bearer ${config.serviceApiKey}`
  }
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), 10000)
  fetch(`${config.upstreamUrl}/v1/models`, { headers, signal: controller.signal })
    .then((upstreamRes) => upstreamRes.json().then((data) => {
      clearTimeout(timer)
      res.status(upstreamRes.status).json(data)
    }))
    .catch((error: unknown) => {
      clearTimeout(timer)
      const message = error instanceof Error ? error.message : 'Upstream unavailable'
      res.status(502).json({ error: 'Upstream unavailable', message })
    })
    .catch(next)
})
