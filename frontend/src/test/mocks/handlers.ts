import { http, HttpResponse } from 'msw'

const BASE = '/api'

// ── Negative-path handlers (off by default; activate per-test via server.use) ─

export const errorHandlers = {
  registerForbidden: http.post(`${BASE}/auth/register`, () =>
    HttpResponse.json(
      { error: 'Forbidden', message: 'Public registration is disabled' },
      { status: 403 }
    )
  ),
  commandUnauthorized: http.post(`${BASE}/command`, () =>
    HttpResponse.json({ error: 'Unauthorized' }, { status: 401 })
  ),
  commandTooLarge: http.post(`${BASE}/command`, () =>
    HttpResponse.json({ error: 'Request body too large' }, { status: 413 })
  ),
  commandRateLimited: http.post(`${BASE}/command`, () =>
    HttpResponse.json({ error: 'Too Many Requests' }, { status: 429 })
  ),
}

export const handlers = [
  http.get(`${BASE}/v1/models`, () =>
    HttpResponse.json({
      object: 'list',
      data: [{ id: 'aurelius', object: 'model', created: Date.now(), owned_by: 'aurelius' }],
    })
  ),

  http.get(`${BASE}/health`, () =>
    HttpResponse.json({ status: 'ok', uptime: 1234, version: '0.1.0', memory: { rss: 256 * 1024 * 1024 } })
  ),

  http.post(`${BASE}/chat/completions`, () =>
    HttpResponse.json({
      id: 'chatcmpl-test',
      object: 'chat.completion',
      created: Date.now(),
      model: 'aurelius',
      choices: [{ index: 0, message: { role: 'assistant', content: 'Test response' }, finish_reason: 'stop' }],
      usage: { prompt_tokens: 10, completion_tokens: 3, total_tokens: 13 },
    })
  ),

  http.post(`${BASE}/v1/chat/completions`, () =>
    HttpResponse.json({
      id: 'chatcmpl-test',
      object: 'chat.completion',
      created: Date.now(),
      model: 'aurelius',
      choices: [{ index: 0, message: { role: 'assistant', content: 'Test response' }, finish_reason: 'stop' }],
      usage: { prompt_tokens: 10, completion_tokens: 3, total_tokens: 13 },
    })
  ),
]
