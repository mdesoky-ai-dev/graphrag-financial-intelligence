/**
 * Typed API client for the GraphRAG backend.
 *
 * One function per endpoint. Wraps fetch with:
 *   - JSON serialization on the way out
 *   - Typed deserialization on the way in
 *   - Structured error reporting (so the UI can show a clean message)
 *
 * The base URL comes from VITE_API_BASE_URL (.env). At build time Vite inlines
 * that value so the deployed bundle hits the right backend without code changes.
 */

import type { AskRequest, AskResponse, ApiError } from '../types/api';

const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000';

/**
 * Custom error class so callers can branch on `.status` and `.detail`.
 * Plain Error would lose the structured detail from FastAPI's 422 responses.
 */
export class ApiRequestError extends Error implements ApiError {
  status: number;
  detail: string | unknown;

  constructor(status: number, detail: string | unknown) {
    super(typeof detail === 'string' ? detail : 'Request failed');
    this.status = status;
    this.detail = detail;
  }
}

/**
 * POST /ask — submit a question, get a grounded answer + diagnostics.
 *
 * Throws ApiRequestError on non-2xx responses; throws plain Error on network
 * failures (e.g., backend not running).
 */
export async function ask(request: AskRequest): Promise<AskResponse> {
  let response: Response;
  try {
    response = await fetch(`${BASE_URL}/ask`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
  } catch (e) {
    // Network-level failure: server unreachable, CORS preflight failure, etc.
    throw new Error(
      `Could not reach the API at ${BASE_URL}. Is the backend running?`
    );
  }

  if (!response.ok) {
    let detail: unknown = 'Unknown error';
    try {
      const body = await response.json();
      detail = body?.detail ?? body;
    } catch {
      detail = await response.text();
    }
    throw new ApiRequestError(response.status, detail);
  }

  return (await response.json()) as AskResponse;
}
