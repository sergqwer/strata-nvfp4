/** Record ids. crypto.randomUUID is a secure-context API, but this UI runs on plain http LAN
 * addresses too, where it does not exist — fall back to a random id (getRandomValues is not
 * secure-context-gated). Ids only need uniqueness, not RFC 4122 shape. */
export function newId(): string {
  return crypto.randomUUID?.()
    ?? 'p-' + Date.now().toString(36) + '-' + Array.from(crypto.getRandomValues(new Uint8Array(8)), b => b.toString(16).padStart(2, '0')).join('');
}
