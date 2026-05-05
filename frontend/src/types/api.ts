/**
 * Type definitions mirroring the backend's Pydantic schemas in app/schemas.py.
 *
 * Keep these in sync with the backend. If you change a Pydantic model there,
 * update the matching interface here. (A future improvement would be to
 * auto-generate these from the OpenAPI schema served at /openapi.json.)
 */

export interface AskRequest {
  question: string;
}

export interface GraphStepInfo {
  pattern: string;                          // e.g. "risks_for_company"
  params: Record<string, unknown>;          // e.g. { company: "Apple Inc.", category: "supply_chain" }
}

export interface PlanInfo {
  graph_steps: GraphStepInfo[];
  run_vector: boolean;
  vector_top_k: number;
  notes: string[];
}

export interface FusedChunkInfo {
  chunk_id: string;
  rrf_score: number;
  sources: string[];                        // e.g. ["graph", "vector"] or just ["graph"]
  graph_rank: number | null;
  vector_rank: number | null;
}

export interface AskResponse {
  question: string;
  answer: string;                           // markdown with [chunk_id] citations
  cited_chunk_ids: string[];
  elapsed_seconds: number;
  plan: PlanInfo;
  graph_hits_count: number;
  vector_hits_count: number;
  fused_chunks: FusedChunkInfo[];
  chunks_fed_to_synthesis: number;
}

/** Shape returned when a request fails. Useful for typed error handling. */
export interface ApiError {
  detail: string | unknown;
  status?: number;
}
