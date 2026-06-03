import { beforeAll, describe, expect, it } from 'vitest'
import { buildApp } from '../src/server.js'
import { registerApiKey } from '../src/middleware/auth.js'
import { commandRequestHeaders } from '../src/routes/scheduler.js'
import { invokeApp } from './request-app.js'

const app = buildApp()

beforeAll(() => {
  registerApiKey('test-readonly-key', {
    id: 'reader',
    role: 'user',
    scopes: ['activity:read'],
  })
  registerApiKey('test-eval-read-key', {
    id: 'eval-reader',
    role: 'user',
    scopes: ['eval:read'],
  })
  registerApiKey('test-scheduler-read-key', {
    id: 'sched-reader',
    role: 'user',
    scopes: ['scheduler:read'],
  })
  registerApiKey('test-user-key', {
    id: 'basic-user',
    role: 'user',
    scopes: ['read'],
  })
})

describe('Scope matrix enforcement', () => {
  it('read-only activity key can read activity but not write', async () => {
    const readRes = await invokeApp(app, {
      method: 'GET',
      path: '/api/activity',
      headers: { 'X-API-Key': 'test-readonly-key' },
    })
    expect(readRes.status).toBe(200)

    const writeRes = await invokeApp(app, {
      method: 'POST',
      path: '/api/activity',
      headers: {
        'X-API-Key': 'test-readonly-key',
        'Content-Type': 'application/json',
      },
      body: { command: 'scope-test', success: true },
    })
    expect(writeRes.status).toBe(403)
  })

  it('eval:read key cannot POST evaluation results', async () => {
    const writeRes = await invokeApp(app, {
      method: 'POST',
      path: '/api/evaluation/results',
      headers: {
        'X-API-Key': 'test-eval-read-key',
        'Content-Type': 'application/json',
      },
      body: { benchmark: 'scope-test', model: 'm', score: 1 },
    })
    expect(writeRes.status).toBe(403)
  })

  it('scheduler:read key can list tasks but not create them', async () => {
    const listRes = await invokeApp(app, {
      method: 'GET',
      path: '/api/scheduler',
      headers: { 'X-API-Key': 'test-scheduler-read-key' },
    })
    expect(listRes.status).toBe(200)

    const createRes = await invokeApp(app, {
      method: 'POST',
      path: '/api/scheduler',
      headers: {
        'X-API-Key': 'test-scheduler-read-key',
        'Content-Type': 'application/json',
      },
      body: { name: 'blocked', cron: '* * * * *', command: 'echo nope' },
    })
    expect(createRes.status).toBe(403)
  })

  it('non-admin key cannot mint API keys', async () => {
    const res = await invokeApp(app, {
      method: 'POST',
      path: '/api/auth/keys/generate',
      headers: {
        'X-API-Key': 'test-user-key',
        'Content-Type': 'application/json',
      },
    })
    expect(res.status).toBe(403)
  })

  it('admin wildcard key still passes scoped routes', async () => {
    const res = await invokeApp(app, {
      method: 'GET',
      path: '/api/stats',
      headers: { 'X-API-Key': 'test-admin-key' },
    })
    expect(res.status).toBe(200)
  })
})

describe('Scheduler command dispatch headers (C-08)', () => {
  it('includes X-API-Key when AURELIUS_API_KEY is set', () => {
    process.env.AURELIUS_API_KEY = 'test-admin-key'
    const headers = commandRequestHeaders()
    expect(headers['X-API-Key']).toBe('test-admin-key')
    expect(headers['Content-Type']).toBe('application/json')
  })

  it('omits X-API-Key when AURELIUS_API_KEY is unset', () => {
    delete process.env.AURELIUS_API_KEY
    const headers = commandRequestHeaders()
    expect(headers['X-API-Key']).toBeUndefined()
    process.env.AURELIUS_API_KEY = 'test-admin-key'
  })
})
