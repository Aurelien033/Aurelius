import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { render, screen, waitFor } from '../test-utils'
import userEvent from '@testing-library/user-event'

describe('Login', () => {
  beforeEach(() => {
    localStorage.clear()
    vi.stubGlobal('fetch', vi.fn(async () => new Response(
      JSON.stringify({ success: true }),
      { status: 200, headers: { 'Content-Type': 'application/json' } }
    )))
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  it('renders login form', async () => {
    const { default: Login } = await import('../../pages/Login')
    render(<Login />)

    expect(screen.getByText('Aurelius')).toBeInTheDocument()
    expect(screen.getByText('Agent Operations Center')).toBeInTheDocument()
  })

  it('shows API key input field', async () => {
    const user = userEvent.setup()
    const { default: Login } = await import('../../pages/Login')
    render(<Login />)

    // Click API Key tab to show the input
    await user.click(screen.getByRole('button', { name: /api key/i }))
    expect(screen.getByPlaceholderText('sk-...')).toBeInTheDocument()
  })

  it('has submit button', async () => {
    const { default: Login } = await import('../../pages/Login')
    render(<Login />)

    expect(screen.getByRole('button', { name: /sign in/i })).toBeInTheDocument()
  })

  it('on submit with demo mode, redirects or shows success', async () => {
    const user = userEvent.setup()
    const { default: Login } = await import('../../pages/Login')
    render(<Login />)

    // Click Demo button
    await user.click(screen.getByRole('button', { name: /demo/i }))

    // Submit the form
    await user.click(screen.getByRole('button', { name: /enter demo mode/i }))

    await waitFor(() => {
      expect(fetch).toHaveBeenCalledWith('/api/auth/login', expect.objectContaining({
        method: 'POST',
      }))
    })
  })

  it('shows error when login returns 401 invalid credentials', async () => {
    vi.stubGlobal('fetch', vi.fn(async () =>
      new Response(JSON.stringify({ error: 'Unauthorized' }), {
        status: 401,
        headers: { 'Content-Type': 'application/json' },
      })
    ))

    const user = userEvent.setup()
    const { default: Login } = await import('../../pages/Login')
    render(<Login />)

    await user.click(screen.getByRole('button', { name: /api key/i }))
    await user.type(screen.getByPlaceholderText('sk-...'), 'bad-key')
    await user.click(screen.getByRole('button', { name: /sign in/i }))

    await waitFor(() => {
      expect(screen.getByText(/invalid credentials/i)).toBeInTheDocument()
    })
  })

  it('shows error when registration is forbidden (ALLOW_PUBLIC_REGISTRATION=false)', async () => {
    vi.stubGlobal('fetch', vi.fn(async () =>
      new Response(
        JSON.stringify({ error: 'Forbidden', message: 'Public registration is disabled' }),
        { status: 403, headers: { 'Content-Type': 'application/json' } }
      )
    ))

    const user = userEvent.setup()
    const { default: Login } = await import('../../pages/Login')
    render(<Login />)

    // Attempt sign-in which triggers the login endpoint; 403 surfaces as an error
    await user.click(screen.getByRole('button', { name: /api key/i }))
    await user.type(screen.getByPlaceholderText('sk-...'), 'any-key')
    await user.click(screen.getByRole('button', { name: /sign in/i }))

    await waitFor(() => {
      // Component surfaces any non-ok response as an error string
      expect(screen.getByText(/invalid credentials/i)).toBeInTheDocument()
    })
  })
})
