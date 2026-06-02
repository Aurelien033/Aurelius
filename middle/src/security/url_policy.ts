// URL policy for upstream calls.
//
// Pre-remediation, every upstream fetch in the BFF used a free-form
// URL string from config or environment variables. The threat is
// SSRF (forcing the BFF to call an internal service) and
// credential exfiltration (redirecting the BFF's bearer to an
// attacker host). The fix centralizes URL validation here; every
// upstream call must go through validateUpstreamUrl.

import { URL } from 'url'
import { lookup } from 'dns'
import { promisify } from 'util'
import { isIP } from 'net'

const lookupAsync = promisify(lookup)

// Cloud metadata endpoints. These MUST be rejected even in dev.
const BLOCKED_HOSTNAMES = new Set([
  'metadata.google.internal',
  'metadata',
  '169.254.169.254',
])

// Loopback and private CIDRs that must be rejected in production.
// In dev, loopback is allowed (the BFF calls the gateway on
// localhost).
const PRIVATE_IPV4_RANGES: Array<[RegExp, number]> = [
  [/^10\./, 8],                // 10.0.0.0/8
  [/^172\.(1[6-9]|2\d|3[01])\./, 12],  // 172.16.0.0/12
  [/^192\.168\./, 16],         // 192.168.0.0/16
  [/^127\./, 8],               // 127.0.0.0/8
  [/^0\./, 8],                 // 0.0.0.0/8
  [/^169\.254\./, 16],         // 169.254.0.0/16 (link-local, metadata)
  [/^100\.(6[4-9]|[7-9]\d|1[0-1]\d|12[0-7])\./, 10], // 100.64.0.0/10 CGNAT
  [/^192\.0\.0\./, 24],
  [/^192\.0\.2\./, 24],
  [/^198\.18\./, 15],
  [/^198\.51\.100\./, 24],
  [/^203\.0\.113\./, 24],
  [/^224\./, 4],               // multicast
  [/^240\./, 4],               // reserved
]

const PRIVATE_IPV6_RANGES = [
  /^::1$/,                     // ::1 loopback
  /^fc[0-9a-f]{2}:/i,         // fc00::/7 unique-local
  /^fe[89ab][0-9a-f]:/i,      // fe80::/10 link-local
]

export type UrlPolicyError =
  | 'invalid-url'
  | 'unsupported-protocol'
  | 'metadata-blocked'
  | 'private-ip-blocked'
  | 'userinfo-blocked'
  | 'hostname-blocked'
  | 'dns-resolves-to-blocked'

export interface UrlPolicyOptions {
  /** Allow loopback/private addresses. Default: false in prod. */
  allowLoopback?: boolean
  /** Allow http:// (not just https://). Default: false in prod. */
  allowHttp?: boolean
  /** Optional explicit list of allowed hostnames. */
  allowedHostnames?: string[]
  /** Optional explicit list of allowed hostname suffixes. */
  allowedHostnameSuffixes?: string[]
}

export interface UrlPolicyResult {
  ok: boolean
  reason?: UrlPolicyError
  url?: URL
  resolvedIp?: string
}

/**
 * Validate a URL for use as an upstream call target.
 *
 * @param input The URL string to validate
 * @param options Policy options (dev vs prod, allowlists)
 * @returns {ok, reason?, url?, resolvedIp?}
 */
export async function validateUpstreamUrl(
  input: string,
  options: UrlPolicyOptions = {},
): Promise<UrlPolicyResult> {
  if (typeof input !== 'string' || input.length === 0) {
    return { ok: false, reason: 'invalid-url' }
  }

  // Reject userinfo URLs (https://attacker:pw@victim)
  // The presence of '@' before the path/host separator is the
  // marker. Use a regex that catches the most common patterns.
  if (/@/.test(input.split('?')[0].split('#')[0])) {
    return { ok: false, reason: 'userinfo-blocked' }
  }

  let url: URL
  try {
    url = new URL(input)
  } catch {
    return { ok: false, reason: 'invalid-url' }
  }

  // Protocol check
  const allowHttp = options.allowHttp ?? false
  if (url.protocol === 'http:' && !allowHttp) {
    return { ok: false, reason: 'unsupported-protocol' }
  }
  if (url.protocol !== 'http:' && url.protocol !== 'https:') {
    return { ok: false, reason: 'unsupported-protocol' }
  }

  // Hostname blocklist
  const hostname = url.hostname.toLowerCase()
  if (BLOCKED_HOSTNAMES.has(hostname)) {
    return { ok: false, reason: 'hostname-blocked' }
  }

  // Hostname allowlist (if configured)
  if (options.allowedHostnames && options.allowedHostnames.length > 0) {
    const allowed = options.allowedHostnames.some(
      (h) => h.toLowerCase() === hostname,
    )
    if (!allowed) {
      return { ok: false, reason: 'hostname-blocked' }
    }
  }
  if (options.allowedHostnameSuffixes && options.allowedHostnameSuffixes.length > 0) {
    const allowed = options.allowedHostnameSuffixes.some(
      (s) => hostname.endsWith(s.toLowerCase()),
    )
    if (!allowed) {
      return { ok: false, reason: 'hostname-blocked' }
    }
  }

  // Resolve hostname to IP and check metadata / private ranges
  let resolvedIp: string | undefined
  if (isIP(hostname)) {
    resolvedIp = hostname
  } else {
    try {
      const result = await lookupAsync(hostname, { all: false })
      resolvedIp = result.address
    } catch {
      return { ok: false, reason: 'invalid-url' }
    }
  }

  if (resolvedIp) {
    // Metadata IP
    if (resolvedIp === '169.254.169.254') {
      return { ok: false, reason: 'metadata-blocked', resolvedIp }
    }
    // Private IP check
    const allowLoopback = options.allowLoopback ?? false
    if (!allowLoopback && isPrivateIp(resolvedIp)) {
      return { ok: false, reason: 'private-ip-blocked', resolvedIp }
    }
    // Loopback check (in case allowLoopback=false but we still
    // want to allow explicit localhost in dev)
    if (allowLoopback && (
      resolvedIp === '127.0.0.1' || resolvedIp === '::1' || resolvedIp.startsWith('127.')
    )) {
      // allowed
    } else if (isPrivateIp(resolvedIp)) {
      return { ok: false, reason: 'private-ip-blocked', resolvedIp }
    }
  }

  return { ok: true, url, resolvedIp }
}

function isPrivateIp(ip: string): boolean {
  if (isIP(ip) === 4) {
    for (const [pattern] of PRIVATE_IPV4_RANGES) {
      if (pattern.test(ip)) return true
    }
    return false
  }
  if (isIP(ip) === 6) {
    for (const pattern of PRIVATE_IPV6_RANGES) {
      if (pattern.test(ip)) return true
    }
    return false
  }
  return false
}

/**
 * Synchronous variant: validates the URL string without DNS
 * resolution. Use this when you've already resolved the host or
 * when the URL is a literal IP.
 */
export function validateUpstreamUrlSync(
  input: string,
  options: UrlPolicyOptions = {},
): UrlPolicyResult {
  if (typeof input !== 'string' || input.length === 0) {
    return { ok: false, reason: 'invalid-url' }
  }
  if (/@/.test(input.split('?')[0].split('#')[0])) {
    return { ok: false, reason: 'userinfo-blocked' }
  }
  let url: URL
  try {
    url = new URL(input)
  } catch {
    return { ok: false, reason: 'invalid-url' }
  }
  const allowHttp = options.allowHttp ?? false
  if (url.protocol === 'http:' && !allowHttp) {
    return { ok: false, reason: 'unsupported-protocol' }
  }
  if (url.protocol !== 'http:' && url.protocol !== 'https:') {
    return { ok: false, reason: 'unsupported-protocol' }
  }
  const hostname = url.hostname.toLowerCase()
  if (BLOCKED_HOSTNAMES.has(hostname)) {
    return { ok: false, reason: 'hostname-blocked' }
  }
  if (isIP(hostname) === 4 && isPrivateIp(hostname)) {
    if (!options.allowLoopback) {
      return { ok: false, reason: 'private-ip-blocked' }
    }
  }
  if (isIP(hostname) === 6 && isPrivateIp(hostname)) {
    if (!options.allowLoopback) {
      return { ok: false, reason: 'private-ip-blocked' }
    }
  }
  return { ok: true, url, resolvedIp: isIP(hostname) ? hostname : undefined }
}
