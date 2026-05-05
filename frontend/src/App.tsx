import { useState } from 'react';
import { AlertCircle } from 'lucide-react';
import { Header } from './components/Header';
import { QueryBox } from './components/QueryBox';
import { AnswerPanel } from './components/AnswerPanel';
import { DiagnosticsPanel } from './components/DiagnosticsPanel';
import { ask, ApiRequestError } from './api/client';
import type { AskResponse } from './types/api';

/**
 * Top-level application. Owns the request state machine:
 *   - idle (initial) — no question asked yet, show empty state
 *   - loading — request in flight
 *   - success — answer displayed
 *   - error — request failed, show message
 */
export default function App() {
  const [result, setResult] = useState<AskResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const askQuestion = async (question: string) => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await ask({ question });
      setResult(response);
    } catch (e) {
      const msg = e instanceof ApiRequestError
        ? `[${e.status}] ${typeof e.detail === 'string' ? e.detail : JSON.stringify(e.detail)}`
        : e instanceof Error
        ? e.message
        : 'Unknown error';
      setError(msg);
      setResult(null);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen">
      <Header />
      <QueryBox onSubmit={askQuestion} isLoading={isLoading} />

      {error && (
        <div className="mx-auto max-w-5xl px-6 pb-4">
          <div className="flex items-start gap-2 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800">
            <AlertCircle className="h-5 w-5 flex-shrink-0 mt-0.5" />
            <div>
              <strong className="font-semibold">Request failed</strong>
              <div className="mt-0.5 font-mono text-xs">{error}</div>
            </div>
          </div>
        </div>
      )}

      {result && !error && (
        <>
          <AnswerPanel result={result} />
          <DiagnosticsPanel result={result} />
        </>
      )}

      {!result && !error && !isLoading && (
        <EmptyState />
      )}
    </div>
  );
}

function EmptyState() {
  return (
    <div className="mx-auto max-w-5xl px-6 py-16 text-center">
      <div className="rounded-xl border-2 border-dashed border-slate-200 bg-white/50 px-6 py-12">
        <h3 className="text-base font-semibold text-slate-900">
          Ask a question to see hybrid retrieval in action
        </h3>
        <p className="mt-2 text-sm text-slate-500">
          The system fuses graph-based reasoning over a knowledge graph of risks,
          companies, and geographies with semantic search over chunked filings.
          Every answer is cited and the retrieval signals are exposed below.
        </p>
      </div>
    </div>
  );
}
