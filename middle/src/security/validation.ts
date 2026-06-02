// Central validation helpers for BFF inputs.
//
// This module exports typed guards and sanitizers for common input
// shapes. Routes use these instead of ad-hoc checks.

/**
 * Sanitize a string for inclusion in a log line. Strips CR/LF
 * and other control characters that could enable log injection
 * (CWE-117). The result is always a single-line string.
 */
export function sanitizeForLog(input: unknown): string {
  if (input === null || input === undefined) return ''
  const s = typeof input === 'string' ? input : String(input)
  // Strip CR, LF, and other control characters except space
  return s.replace(/[\x00-\x1f\x7f]/g, '?').slice(0, 256)
}

/**
 * Sanitize a filename. Strips path separators, parent-directory
 * references, and control characters. The result is a safe
 * filename suitable for use as a file system entry.
 */
export function sanitizeFilename(input: string): string {
  if (typeof input !== 'string') return ''
  // Strip path components (\\, /) and parent references
  let name = input
    .replace(/[\\\/]/g, '_')
    .replace(/\.\.+/g, '_')
    .replace(/[\x00-\x1f\x7f]/g, '')
  // Cap length
  if (name.length > 255) name = name.slice(0, 255)
  return name
}

/**
 * Validate that a value is a positive integer within [min, max].
 * Returns the value or null if invalid.
 */
export function validatePositiveInt(
  input: unknown,
  min: number,
  max: number,
): number | null {
  if (typeof input !== 'number' || !Number.isFinite(input)) return null
  if (!Number.isInteger(input)) return null
  if (input < min || input > max) return null
  return input
}

/**
 * Validate that a value is a non-empty string within [min, max].
 */
export function validateString(
  input: unknown,
  min: number,
  max: number,
): string | null {
  if (typeof input !== 'string') return null
  if (input.length < min || input.length > max) return null
  return input
}

/**
 * Validate a UUID v4 string.
 */
export function isUuid(input: unknown): boolean {
  if (typeof input !== 'string') return false
  return /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(input)
}
