import { useEffect, useRef, useState, useCallback } from 'react'

export type WsConnectionState = 'connecting' | 'open' | 'closed' | 'error'

export interface WsMessage {
  type: string
  payload?: Record<string, unknown>
}

export interface UseWebSocketResult {
  state: WsConnectionState
  lastMessage: WsMessage | null
  send: (msg: WsMessage) => void
  subscribe: (room: string) => void
  unsubscribe: (room: string) => void
  sendRoomMessage: (room: string, data: unknown) => void
  reconnect: () => void
  /** Legacy: register a callback for a specific event type. */
  on?: (event: string, cb: (payload: unknown) => void) => void
  /** Legacy: remove a callback for a specific event type. */
  off?: (event: string, cb: (payload: unknown) => void) => void
}

/**
 * C3 fix: the pre-remediation hook constructed a ws:// URL with the
 * raw API key in the query string. That exposed the operator's
 * master credential to any browser extension, proxy, or network
 * observer. The new hook authenticates with a short-lived
 * `ws-token` minted by /api/auth/ws-token, kept in memory (not
 * localStorage), and renews the connection on token expiry.
 *
 * C3 protocol fix: the canonical subscribe schema is
 * { type: 'subscribe', room: '...' }. The legacy { channel } field
 * is gone; subscribers must use `room`.
 */
export function useWebSocket(url?: string): UseWebSocketResult {
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const tokenRef = useRef<string | null>(null)
  const tokenExpiresAtRef = useRef<number>(0)
  const [state, setState] = useState<WsConnectionState>('closed')
  const [lastMessage, setLastMessage] = useState<WsMessage | null>(null)
  // Legacy event-listener registry. New code uses the
  // returned handlers; legacy components (ActivityFeed,
  // Notifications) call ws.on('event', cb).
  const listenersRef = useRef<Map<string, Set<(payload: unknown) => void>>>(new Map())

  // Legacy .on() / .off() method to keep back-compat with
  // the pre-remediation useWebSocket() shape.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  ;(useWebSocket as any).on = (event: string, cb: (payload: unknown) => void) => {
    if (!listenersRef.current.has(event)) listenersRef.current.set(event, new Set())
    listenersRef.current.get(event)!.add(cb)
  }
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  ;(useWebSocket as any).off = (event: string, cb: (payload: unknown) => void) => {
    listenersRef.current.get(event)?.delete(cb)
  }

  // The legacy API allowed ws.on('event', cb) WITHOUT
  // starting a connection. If no URL is provided, we
  // return a stub UseWebSocketResult that supports .on()
  // but does not open a socket. This preserves the
  // legacy call sites.
  const noop = (() => {}) as any

  if (!url) {
    return {
      state,
      lastMessage,
      send: noop,
      subscribe: noop,
      unsubscribe: noop,
      sendRoomMessage: noop,
      reconnect: noop,
      on: (event, cb) => {
        if (!listenersRef.current.has(event)) listenersRef.current.set(event, new Set())
        listenersRef.current.get(event)!.add(cb)
      },
      off: (event, cb) => {
        listenersRef.current.get(event)?.delete(cb)
      },
    } as UseWebSocketResult
  }

  /** Mint a short-lived WS token from the server. The response
   * includes an `exp` claim; the hook reconnects before expiry. */
  const fetchWsToken = useCallback(async (): Promise<string> => {
    const res = await fetch('/api/auth/ws-token', { method: 'POST', credentials: 'include' })
    if (!res.ok) throw new Error(`ws-token mint failed: ${res.status}`)
    const data = await res.json()
    if (!data || typeof data.token !== 'string') {
      throw new Error('ws-token response missing token')
    }
    tokenRef.current = data.token
    tokenExpiresAtRef.current = typeof data.exp === 'number' ? data.exp : Math.floor(Date.now() / 1000) + 240
    return data.token
  }, [])

  const connect = useCallback(async () => {
    if (wsRef.current && wsRef.current.readyState !== WebSocket.CLOSED) {
      return
    }
    setState('connecting')
    let token: string
    try {
      token = await fetchWsToken()
    } catch (e) {
      setState('error')
      return
    }
    // Build URL. The token is passed via the subprotocol header
    // (aurelius-ws-token.<jwt>) which is the only place a token
    // is safe from URL-logged proxies. We append a `room=` no-op
    // here is unnecessary; the server reads the subprotocol.
    const proto = `aurelius-ws-token.${token}`
    const ws = new WebSocket(url, [proto])
    wsRef.current = ws

    ws.onopen = () => {
      setState('open')
    }
    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data) as WsMessage
        setLastMessage(msg)
        // Fire any registered legacy listeners
        const set = listenersRef.current.get(msg.type)
        if (set) {
          for (const cb of set) cb(msg.payload)
        }
      } catch {
        // ignore malformed
      }
    }
    ws.onclose = () => {
      setState('closed')
    }
    ws.onerror = () => {
      setState('error')
    }
  }, [url, fetchWsToken])

  useEffect(() => {
    connect()
    const renewal = setInterval(() => {
      const now = Math.floor(Date.now() / 1000)
      if (tokenExpiresAtRef.current - now < 30 && wsRef.current?.readyState === WebSocket.OPEN) {
        // Pre-emptively close and reconnect with a fresh token
        try { wsRef.current?.close(4000, 'token-renewal') } catch { /* ignore */ }
        connect()
      }
    }, 15000)
    return () => {
      clearInterval(renewal)
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current)
      try { wsRef.current?.close() } catch { /* ignore */ }
    }
  }, [connect])

  const send = useCallback((msg: WsMessage) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(msg))
    }
  }, [])

  const subscribe = useCallback((room: string) => {
    // Canonical schema: { type: 'subscribe', room: '...' }
    send({ type: 'subscribe', payload: { room } })
  }, [send])

  const unsubscribe = useCallback((room: string) => {
    send({ type: 'unsubscribe', payload: { room } })
  }, [send])

  const sendRoomMessage = useCallback((room: string, data: unknown) => {
    send({ type: 'room:message', payload: { room, data } })
  }, [send])

  const reconnect = useCallback(() => {
    try { wsRef.current?.close() } catch { /* ignore */ }
    connect()
  }, [connect])

  return { state, lastMessage, send, subscribe, unsubscribe, sendRoomMessage, reconnect }
}
