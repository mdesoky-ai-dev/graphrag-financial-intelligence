import { Network } from 'lucide-react';

/**
 * Top-of-page header. Sticky so it stays visible as the user scrolls
 * through long answers and the diagnostics table.
 */
export function Header() {
  return (
    <header className="sticky top-0 z-10 border-b border-slate-200 bg-white/80 backdrop-blur">
      <div className="mx-auto max-w-5xl px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-indigo-600 text-white">
            <Network className="h-5 w-5" />
          </div>
          <div>
            <h1 className="text-lg font-semibold text-slate-900 leading-tight">
              GraphRAG Financial Intelligence
            </h1>
            <p className="text-xs text-slate-500 leading-tight">
              Hybrid graph + vector retrieval over SEC filings
            </p>
          </div>
        </div>
        <div className="hidden sm:flex items-center gap-2 text-xs text-slate-500">
          <span className="inline-flex h-2 w-2 rounded-full bg-emerald-500" />
          Apple Inc. 10-K FY25
        </div>
      </div>
    </header>
  );
}
