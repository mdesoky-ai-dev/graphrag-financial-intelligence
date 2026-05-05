import { useState } from 'react';
import { ChevronDown, ChevronRight, Database, Search, GitMerge, Layers } from 'lucide-react';
import type { AskResponse, FusedChunkInfo } from '../types/api';

interface DiagnosticsPanelProps {
  result: AskResponse;
}

/**
 * Showcase panel — exposes the inner workings of the retrieval pipeline.
 * This is what makes the demo distinctive: most RAG UIs hide retrieval;
 * ours surfaces the planner's decisions, graph/vector signals, and the
 * RRF fusion ranking with provenance.
 */
export function DiagnosticsPanel({ result }: DiagnosticsPanelProps) {
  const [expanded, setExpanded] = useState(true);

  return (
    <section className="mx-auto max-w-5xl px-6 pb-12">
      <button
        type="button"
        onClick={() => setExpanded((v) => !v)}
        className="flex w-full items-center gap-2 rounded-lg border border-slate-200 bg-white px-4 py-3 text-left text-sm font-semibold text-slate-900 shadow-sm transition hover:bg-slate-50"
      >
        {expanded ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
        Retrieval Diagnostics
        <span className="ml-2 text-xs font-normal text-slate-500">
          plan · {result.graph_hits_count} graph hits · {result.vector_hits_count} vector hits ·
          {' '}{result.fused_chunks.length} fused
        </span>
      </button>

      {expanded && (
        <div className="mt-3 space-y-4">
          <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
            <PlanCard result={result} />
            <StatCard
              icon={<Database className="h-4 w-4" />}
              label="Graph Retriever"
              primary={result.graph_hits_count.toString()}
              secondary="edges from Neo4j"
              accent="emerald"
            />
            <StatCard
              icon={<Search className="h-4 w-4" />}
              label="Vector Retriever"
              primary={result.vector_hits_count.toString()}
              secondary="chunks from Pinecone"
              accent="sky"
            />
          </div>

          <FusedChunksTable chunks={result.fused_chunks} />
        </div>
      )}
    </section>
  );
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function PlanCard({ result }: { result: AskResponse }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
      <div className="flex items-center gap-2 text-xs font-medium text-slate-500">
        <Layers className="h-4 w-4" />
        PLAN
      </div>
      <div className="mt-2 space-y-1.5">
        {result.plan.graph_steps.map((step, i) => (
          <div key={i} className="text-xs">
            <span className="font-mono font-semibold text-indigo-700">{step.pattern}</span>
            <div className="mt-0.5 ml-2 text-slate-500">
              {Object.entries(step.params).map(([k, v]) => (
                <div key={k}>
                  <span className="text-slate-400">{k}=</span>
                  <span className="text-slate-700">{String(v ?? '—')}</span>
                </div>
              ))}
            </div>
          </div>
        ))}
        <div className="border-t border-slate-100 pt-1.5 text-xs text-slate-500">
          <div>vector: top {result.plan.vector_top_k}</div>
        </div>
      </div>
    </div>
  );
}

function StatCard({
  icon,
  label,
  primary,
  secondary,
  accent,
}: {
  icon: JSX.Element;
  label: string;
  primary: string;
  secondary: string;
  accent: 'emerald' | 'sky';
}) {
  const accentClasses = accent === 'emerald'
    ? 'text-emerald-700 bg-emerald-50'
    : 'text-sky-700 bg-sky-50';
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
      <div className={`inline-flex items-center gap-2 rounded-md px-2 py-1 text-xs font-medium ${accentClasses}`}>
        {icon}
        {label}
      </div>
      <div className="mt-2 text-2xl font-bold text-slate-900">{primary}</div>
      <div className="text-xs text-slate-500">{secondary}</div>
    </div>
  );
}

function FusedChunksTable({ chunks }: { chunks: FusedChunkInfo[] }) {
  return (
    <div className="overflow-hidden rounded-lg border border-slate-200 bg-white shadow-sm">
      <div className="flex items-center gap-2 border-b border-slate-200 bg-slate-50 px-4 py-2.5">
        <GitMerge className="h-4 w-4 text-slate-500" />
        <span className="text-xs font-semibold uppercase tracking-wide text-slate-500">
          Fused Chunks (Reciprocal Rank Fusion)
        </span>
        <span className="ml-auto text-xs text-slate-500">{chunks.length} ranked</span>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="border-b border-slate-200 bg-slate-50/50 text-xs uppercase tracking-wide text-slate-500">
            <tr>
              <th className="px-4 py-2 text-left font-medium">#</th>
              <th className="px-4 py-2 text-left font-medium">Chunk</th>
              <th className="px-4 py-2 text-right font-medium">RRF Score</th>
              <th className="px-4 py-2 text-center font-medium">Graph Rank</th>
              <th className="px-4 py-2 text-center font-medium">Vector Rank</th>
              <th className="px-4 py-2 text-left font-medium">Sources</th>
            </tr>
          </thead>
          <tbody>
            {chunks.map((chunk, i) => (
              <tr
                key={chunk.chunk_id}
                id={`chunk-${chunk.chunk_id}`}
                className="border-b border-slate-100 last:border-0 transition-shadow"
              >
                <td className="px-4 py-2 text-slate-400 tabular-nums">{i + 1}</td>
                <td className="px-4 py-2 font-mono text-xs text-slate-700">
                  {chunk.chunk_id.replace('apple-10k-fy25_', '')}
                </td>
                <td className="px-4 py-2 text-right tabular-nums text-slate-900">
                  {chunk.rrf_score.toFixed(4)}
                </td>
                <td className="px-4 py-2 text-center tabular-nums text-slate-600">
                  {chunk.graph_rank ?? '—'}
                </td>
                <td className="px-4 py-2 text-center tabular-nums text-slate-600">
                  {chunk.vector_rank ?? '—'}
                </td>
                <td className="px-4 py-2">
                  <div className="flex gap-1.5">
                    {chunk.sources.includes('graph') && (
                      <span className="inline-flex items-center rounded-md bg-emerald-50 px-1.5 py-0.5 text-xs font-medium text-emerald-700">
                        graph
                      </span>
                    )}
                    {chunk.sources.includes('vector') && (
                      <span className="inline-flex items-center rounded-md bg-sky-50 px-1.5 py-0.5 text-xs font-medium text-sky-700">
                        vector
                      </span>
                    )}
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
