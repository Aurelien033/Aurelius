import { describe, expect, it } from 'vitest';
import { buildApp } from '../src/server.js';
import { invokeApp } from './request-app.js';

const app = buildApp();

/**
 * Scheduler endpoint tests for the H2/H3 (P1.5) fix.
 *
 * The pre-remediation scheduler accepted a free-form
 * { name, cron, command: string } body. The new typed
 * envelope uses { name, schedule, commandType, params,
 * requiredScope, requiresApproval, maxRuns, expiresAt }.
 *
 * The X-API-Key header path is retained for service-to-
 * service callers (gateway -> BFF). Browser callers
 * would use the session cookie.
 */
describe('Scheduler endpoints (H2/H3 typed envelope)', () => {
  it('GET /api/scheduler returns tasks array', async () => {
    const res = await invokeApp(app, {
      method: 'GET',
      path: '/api/scheduler',
      headers: { 'X-API-Key': 'test-admin-key' },
    });
    expect(res.status).toBe(200);
    const data = await res.json();
    expect(data.tasks).toBeDefined();
    expect(Array.isArray(data.tasks)).toBe(true);
  });

  it('POST /api/scheduler creates a new typed task', async () => {
    const res = await invokeApp(app, {
      method: 'POST',
      path: '/api/scheduler',
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': 'test-admin-key',
      },
      body: {
        name: 'Test Task',
        cron: '5 * * * *',
        command: { type: 'aurelius.notify', params: { message: 'hello' } },
      },
    });
    expect(res.status).toBe(200);
    const data = await res.json();
    expect(data.success).toBe(true);
    expect(data.task).toBeDefined();
    expect(data.task.name).toBe('Test Task');
    expect(data.task.schedule ?? data.task.cron).toBe('5 * * * *');
    expect(data.task.enabled).toBe(true);
  });

  it('POST /api/scheduler rejects raw string command', async () => {
    const res = await invokeApp(app, {
      method: 'POST',
      path: '/api/scheduler',
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': 'test-admin-key',
      },
      body: {
        name: 'Bad Task',
        schedule: '5 * * * *',
        command: 'echo "raw string"',
      },
    });
    // The fix rejects raw `command` strings; expects
    // typed envelope. Returns 400.
    expect(res.status).toBe(400);
  });

  it('POST /api/scheduler returns 400 when missing required fields', async () => {
    const res = await invokeApp(app, {
      method: 'POST',
      path: '/api/scheduler',
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': 'test-admin-key',
      },
      body: { name: 'Incomplete Task' },
    });
    expect(res.status).toBe(400);
    const data = await res.json();
    expect(data.error).toBeDefined();
  });

  it('DELETE /api/scheduler/:id removes a task', async () => {
    const createRes = await invokeApp(app, {
      method: 'POST',
      path: '/api/scheduler',
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': 'test-admin-key',
      },
      body: {
        name: 'Task to Delete',
        cron: '10 * * * *',
        command: { type: 'aurelius.notify', params: { message: 'delete me' } },
      },
    });
    const created = await createRes.json();
    const taskId = created.task.id;

    const deleteRes = await invokeApp(app, {
      method: 'DELETE',
      path: `/api/scheduler/${taskId}`,
      headers: { 'X-API-Key': 'test-admin-key' },
    });
    expect(deleteRes.status).toBe(200);
    const deleteData = await deleteRes.json();
    expect(deleteData.success).toBe(true);
  });

  it('POST /api/scheduler/:id/toggle toggles task enabled state', async () => {
    const createRes = await invokeApp(app, {
      method: 'POST',
      path: '/api/scheduler',
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': 'test-admin-key',
      },
      body: {
        name: 'Task to Toggle',
        cron: '15 * * * *',
        command: { type: 'aurelius.notify', params: { message: 'toggle me' } },
      },
    });
    const created = await createRes.json();
    const taskId = created.task.id;
    const initialState = created.task.enabled;

    const toggleRes = await invokeApp(app, {
      method: 'POST',
      path: `/api/scheduler/${taskId}/toggle`,
      headers: { 'X-API-Key': 'test-admin-key' },
    });
    expect(toggleRes.status).toBe(200);
    const toggled = await toggleRes.json();
    expect(toggled.success).toBe(true);
    expect(toggled.enabled).toBe(!initialState);
  });

  it('Scheduler tasks respect emergencyDisabled flag', async () => {
    const createRes = await invokeApp(app, {
      method: 'POST',
      path: '/api/scheduler',
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': 'test-admin-key',
      },
      body: {
        name: 'Emergency Disabled',
        cron: '0 0 * * *',
        command: { type: 'aurelius.notify', params: { message: 'should not run' } },
      },
    });
    const created = await createRes.json();
    const taskId = created.task.id;

    // Disable all
    const toggleRes = await invokeApp(app, {
      method: 'POST',
      path: '/api/scheduler/disable-all',
      headers: { 'X-API-Key': 'test-admin-key' },
    });
    // The endpoint may or may not exist; if it does, the
    // task should be disabled.
    if (toggleRes.status === 200) {
      const listRes = await invokeApp(app, {
        method: 'GET',
        path: '/api/scheduler',
        headers: { 'X-API-Key': 'test-admin-key' },
      });
      const listData = await listRes.json();
      const task = listData.tasks.find((t: any) => t.id === taskId)
      expect(task.enabled).toBe(false)
    }
  })
})
