import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Clock, FileText } from 'lucide-react';
import type { AskResponse } from '../types/api';

interface AnswerPanelProps {
  result: AskResponse;
}

/**
 * Custom renderer that turns the [chunk_id] tokens in the markdown answer
 * into clickable pills that scroll the diagnostics table to the matching row.
 *
 * The synthesizer's prompt instructs Claude to cite as [apple-10k-fy25_chunk_NNNN],
 * so we look for that pattern in the rendered text and replace it inline.
 */
const CITATION_REGEX = /\[(apple-10k-fy25_chunk_\d{4})\]/g;

function renderTextWithCitations(text: string) {
  const parts: (string | JSX.Element)[] = [];
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  // Reset regex state since /g regexes carry state across calls.
  CITATION_REGEX.lastIndex = 0;

  while ((match = CITATION_REGEX.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parts.push(text.slice(lastIndex, match.index));
    }
    const chunkId = match[1];
    parts.push(
      <a
        key={`${chunkId}-${match.index}`}
        href={`#chunk-${chunkId}`}
        className="citation-link"
        onClick={(e) => {
          e.preventDefault();
          const target = document.getElementById(`chunk-${chunkId}`);
          if (target) {
            target.scrollIntoView({ behavior: 'smooth', block: 'center' });
            target.classList.add('ring-2', 'ring-indigo-400');
            setTimeout(() => target.classList.remove('ring-2', 'ring-indigo-400'), 1500);
          }
        }}
      >
        {chunkId.replace('apple-10k-fy25_', '')}
      </a>
    );
    lastIndex = match.index + match[0].length;
  }

  if (lastIndex < text.length) {
    parts.push(text.slice(lastIndex));
  }
  return parts;
}

export function AnswerPanel({ result }: AnswerPanelProps) {
  return (
    <section className="mx-auto max-w-5xl px-6 py-6">
      <div className="rounded-xl border border-slate-200 bg-white shadow-sm">
        <div className="flex items-center justify-between border-b border-slate-200 px-6 py-3">
          <h2 className="text-sm font-semibold text-slate-900">Answer</h2>
          <div className="flex items-center gap-4 text-xs text-slate-500">
            <span className="inline-flex items-center gap-1">
              <Clock className="h-3.5 w-3.5" />
              {result.elapsed_seconds.toFixed(1)}s
            </span>
            <span className="inline-flex items-center gap-1">
              <FileText className="h-3.5 w-3.5" />
              {result.cited_chunk_ids.length} citations
            </span>
          </div>
        </div>

        <article className="prose-sm prose-slate max-w-none px-6 py-5 text-slate-700 leading-relaxed">
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={{
              // Style headings, paragraphs, lists with Tailwind so we don't
              // need the @tailwindcss/typography plugin.
              h1: ({ children }) => <h1 className="text-xl font-bold text-slate-900 mb-3">{children}</h1>,
              h2: ({ children }) => <h2 className="text-lg font-semibold text-slate-900 mt-4 mb-2">{children}</h2>,
              h3: ({ children }) => <h3 className="text-base font-semibold text-slate-900 mt-3 mb-2">{children}</h3>,
              p: ({ children }) => {
                // Apply citation-pill rendering to every text node in this paragraph.
                const processed = Array.isArray(children)
                  ? children.flatMap((c) =>
                      typeof c === 'string' ? renderTextWithCitations(c) : c
                    )
                  : typeof children === 'string'
                  ? renderTextWithCitations(children)
                  : children;
                return <p className="mb-3 last:mb-0">{processed}</p>;
              },
              ul: ({ children }) => <ul className="list-disc pl-5 mb-3 space-y-1">{children}</ul>,
              li: ({ children }) => <li>{children}</li>,
              strong: ({ children }) => <strong className="font-semibold text-slate-900">{children}</strong>,
            }}
          >
            {result.answer}
          </ReactMarkdown>
        </article>
      </div>
    </section>
  );
}
