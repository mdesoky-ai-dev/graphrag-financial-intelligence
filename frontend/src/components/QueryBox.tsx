import { Sparkles, Send, Loader2 } from 'lucide-react';
import { useState, FormEvent } from 'react';

interface QueryBoxProps {
  onSubmit: (question: string) => void;
  isLoading: boolean;
}

const SUGGESTED_QUESTIONS = [
  'What does Apple say about supply chain risks?',
  'What risks does Apple face related to China?',
  'What competitive risks does Apple face?',
  'What macroeconomic risks does Apple identify?',
];

/**
 * Question input. Submits on Enter or button click. Shows suggested
 * questions as quick-fill chips below the input.
 */
export function QueryBox({ onSubmit, isLoading }: QueryBoxProps) {
  const [value, setValue] = useState('');

  const handleFormSubmit = (e: FormEvent) => {
    e.preventDefault();
    const q = value.trim();
    if (q.length < 3 || isLoading) return;
    onSubmit(q);
  };

  const useSuggestion = (q: string) => {
    setValue(q);
    onSubmit(q);
  };

  return (
    <section className="mx-auto max-w-5xl px-6 pt-8 pb-4">
      <form onSubmit={handleFormSubmit} className="relative">
        <input
          type="text"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder="Ask a question about Apple's 10-K filing..."
          disabled={isLoading}
          className="w-full rounded-xl border border-slate-300 bg-white px-5 py-4 pr-14 text-base text-slate-900 placeholder:text-slate-400 shadow-sm focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-200 disabled:bg-slate-50 disabled:text-slate-400"
        />
        <button
          type="submit"
          disabled={value.trim().length < 3 || isLoading}
          className="absolute right-2 top-1/2 -translate-y-1/2 inline-flex items-center justify-center rounded-lg bg-indigo-600 p-2.5 text-white transition hover:bg-indigo-700 disabled:bg-slate-300"
          aria-label="Submit question"
        >
          {isLoading ? <Loader2 className="h-5 w-5 animate-spin" /> : <Send className="h-5 w-5" />}
        </button>
      </form>

      <div className="mt-3 flex flex-wrap items-center gap-2 text-sm">
        <span className="inline-flex items-center gap-1 text-slate-500">
          <Sparkles className="h-3.5 w-3.5" />
          Try:
        </span>
        {SUGGESTED_QUESTIONS.map((q) => (
          <button
            key={q}
            type="button"
            disabled={isLoading}
            onClick={() => useSuggestion(q)}
            className="rounded-full border border-slate-200 bg-white px-3 py-1 text-xs text-slate-600 transition hover:border-indigo-300 hover:text-indigo-700 disabled:opacity-50"
          >
            {q}
          </button>
        ))}
      </div>
    </section>
  );
}
