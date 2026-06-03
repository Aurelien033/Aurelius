import { describe, expect, it, afterEach } from 'vitest'
import { mkdtempSync, rmSync } from 'node:fs'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import { SqliteBffStore } from '../src/store/sqlite-store.js'

describe('SqliteBffStore persistence (H-25)', () => {
  const tempDirs: string[] = []

  afterEach(() => {
    while (tempDirs.length > 0) {
      rmSync(tempDirs.pop()!, { recursive: true, force: true })
    }
  })

  function tempDbPath(): string {
    const dir = mkdtempSync(join(tmpdir(), 'middle-bff-'))
    tempDirs.push(dir)
    return join(dir, 'middle.sqlite')
  }

  it('persists users and traces across close/reopen', () => {
    const dbPath = tempDbPath()

    const store1 = new SqliteBffStore(dbPath)
    store1.saveUser({
      id: 'user-test',
      username: 'persist-user',
      role: 'user',
      apiKeys: ['ak-test123'],
      createdAt: new Date().toISOString(),
    })
    store1.saveTrace({
      id: 'trace-abc',
      agentId: 'agent-1',
      agentName: 'Persist',
      task: 'verify sqlite',
      status: 'running',
      steps: [],
      startedAt: Date.now(),
      stepCount: 0,
    })
    store1.saveSchedulerTask({
      id: 'task-1',
      name: 'nightly',
      cron: '* * * * *',
      command: 'echo hi',
      enabled: true,
      lastRun: null,
      lastSuccess: null,
      nextRun: null,
      createdAt: new Date().toISOString(),
    })
    store1.close()

    const store2 = new SqliteBffStore(dbPath)
    expect(store2.findUserByUsername('persist-user')?.id).toBe('user-test')
    expect(store2.getTrace('trace-abc')?.task).toBe('verify sqlite')
    expect(store2.listSchedulerTasks()).toHaveLength(1)
    store2.close()
  })
})
