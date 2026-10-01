import { useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { CornerDownLeft, Loader2, Search, Square, TriangleAlert } from 'lucide-react';
import DashboardPanel from '../components/DashboardPanel';
import EmptyState from '../components/EmptyState';
import { describeError, formatTs, loadStored, saveStored, useFetch } from '../lib/api';
import { EXAMPLE_QUESTIONS, describeToolCall, streamSearch, tokenizeAnswer } from '../lib/search';
import { severityBadge } from '../lib/severity';
import { chipClass, inputClass } from '../lib/ui';

const HISTORY_KEY = 'sentinel.search.history';
const MAX_HISTORY = 8;

const eventLink = (id) => `/events?event=${id}`;

const Answer = ({ text }) => (
  <p className="text-sm text-white/85 outfit leading-relaxed whitespace-pre-wrap">
    {tokenizeAnswer(text).map((t, i) => {
      if (t.kind === 'cite') {
        return (
          <Link key={i} to={eventLink(t.id)} className="mono text-[11px] text-cyan-300 hover:text-cyan-100 underline underline-offset-2">
            {t.text}
          </Link>
        );
      }
      if (t.kind === 'bold') return <strong key={i} className="text-white">{t.text}</strong>;
      return <span key={i}>{t.text}</span>;
    })}
  </p>
);

const Steps = ({ steps, running }) => {
  if (!steps.length && !running) return null;
  const results = Object.fromEntries(steps.filter((s) => s.type === 'tool_result').map((s) => [s.id, s.summary]));
  const callsMade = steps.filter((s) => s.type === 'tool_call');
  return (
    <ol className="space-y-1" aria-label="What the assistant looked up">
      {callsMade.map((s) => (
        <li key={s.id} className="text-[10px] mono text-white/50 flex gap-2">
          <span className="text-cyan-500/70">›</span>
          <span className="text-white/70">{describeToolCall(s.name, s.arguments)}</span>
          {results[s.id] ? (
            <span className={results[s.id].startsWith('error') ? 'text-orange-400/80' : 'text-white/40'}>→ {results[s.id]}</span>
          ) : (
            <Loader2 className="w-3 h-3 animate-spin text-cyan-400/60" aria-label="running" />
          )}
        </li>
      ))}
      {running && callsMade.length === 0 && (
        <li className="text-[10px] mono text-white/40 flex items-center gap-2">
          <Loader2 className="w-3 h-3 animate-spin text-cyan-400/60" aria-hidden="true" /> Thinking...
        </li>
      )}
    </ol>
  );
};

const CitedEvents = ({ events }) => {
  if (!events?.length) return null;
  return (
    <div className="grid gap-1.5 sm:grid-cols-2">
      {events.map((ev) => (
        <Link
          key={ev.id}
          to={eventLink(ev.id)}
          className="block border border-white/10 hover:border-cyan-400/40 bg-black/40 px-2.5 py-2 transition-colors"
        >
          <div className="flex items-center justify-between gap-2">
            <span className="text-[10px] mono uppercase text-cyan-300">#{ev.id} {ev.type}</span>
            <span className={`px-1.5 py-px border text-[8px] mono uppercase font-bold ${severityBadge(ev.severity)}`}>{ev.severity}</span>
          </div>
          <div className="text-[10px] mono text-white/50 mt-1">
            {formatTs(ev.start_ts)} · {ev.camera_id}{ev.zone_id ? ` · ${ev.zone_id}` : ''}{ev.track_id != null ? ` · track ${ev.track_id}` : ''}
          </div>
        </Link>
      ))}
    </div>
  );
};

const RUN_ENDINGS = {
  max_steps: 'Stopped at the step limit; the answer may be incomplete.',
  token_budget: 'Stopped at the token budget.',
  llm_error: 'The language model could not be reached.',
  refusal: 'The model declined to answer.',
  max_tokens: 'The model ran out of output tokens.',
};

const SearchPage = () => {
  const { data: status, error: statusError } = useFetch('/api/search/status');
  const [question, setQuestion] = useState('');
  const [asked, setAsked] = useState('');
  const [steps, setSteps] = useState([]);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [running, setRunning] = useState(false);
  const [history, setHistory] = useState(() => loadStored(HISTORY_KEY, []));
  const abortRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => () => abortRef.current?.abort(), []);

  const ask = async (q) => {
    const text = (q ?? question).trim();
    if (!text || running) return;
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setQuestion(text);
    setAsked(text);
    setSteps([]);
    setResult(null);
    setError(null);
    setRunning(true);
    const nextHistory = [text, ...history.filter((h) => h !== text)].slice(0, MAX_HISTORY);
    setHistory(nextHistory);
    saveStored(HISTORY_KEY, nextHistory);
    try {
      await streamSearch(text, (step) => {
        if (step.type === 'done') setResult(step);
        else if (step.type === 'error') setError(step.error);
        else setSteps((s) => [...s, step]);
      }, { signal: controller.signal });
    } catch (err) {
      if (err.name !== 'AbortError') setError(err.status ? err.message : describeError(err));
    } finally {
      if (abortRef.current === controller) setRunning(false);
    }
  };

  const stop = () => {
    abortRef.current?.abort();
    setRunning(false);
  };

  const disabled = status && !status.enabled;
  const providerLabel = status?.provider === 'anthropic' ? 'Anthropic' : status?.provider === 'nvidia' ? 'NVIDIA' : status?.provider;

  return (
    <div className="space-y-4 max-w-4xl">
      <DashboardPanel
        title="Ask about events"
        headerAction={status ? (status.enabled ? `${status.provider} // ${status.model}` : 'NOT CONFIGURED') : null}
      >
        <form
          onSubmit={(e) => { e.preventDefault(); ask(); }}
          className="flex gap-2"
          role="search"
        >
          <label htmlFor="search-question" className="sr-only">Question</label>
          <div className="relative flex-1">
            <Search className="w-3.5 h-3.5 text-cyan-500/50 absolute left-2 top-1/2 -translate-y-1/2" aria-hidden="true" />
            <input
              id="search-question"
              ref={inputRef}
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              maxLength={500}
              placeholder='e.g. "who went near the loading dock after 6pm?"'
              disabled={disabled}
              className={`${inputClass} w-full pl-7 py-2 text-xs outfit`}
              autoComplete="off"
            />
          </div>
          {running ? (
            <button type="button" onClick={stop} className={`${chipClass(true)} flex items-center gap-1.5`}>
              <Square className="w-3 h-3" aria-hidden="true" /> Stop
            </button>
          ) : (
            <button type="submit" disabled={disabled || !question.trim()} className={`${chipClass(true)} flex items-center gap-1.5 disabled:opacity-30 disabled:cursor-not-allowed`}>
              Ask <CornerDownLeft className="w-3 h-3" aria-hidden="true" />
            </button>
          )}
        </form>

        {disabled && (
          <p className="mt-3 text-[11px] mono text-orange-300/90 flex gap-2">
            <TriangleAlert className="w-3.5 h-3.5 shrink-0" aria-hidden="true" />
            <span>Search is off: {status.reason}. Add the key to <code>.env</code> and restart the server.</span>
          </p>
        )}
        {statusError && <p className="mt-3 text-[11px] mono text-white/40">{describeError(statusError)}</p>}

        <div className="mt-3 flex flex-wrap gap-1.5">
          {(history.length ? history : EXAMPLE_QUESTIONS).slice(0, 6).map((q) => (
            <button key={q} type="button" disabled={disabled || running} onClick={() => ask(q)} className={`${chipClass(false)} normal-case disabled:opacity-30`}>
              {q}
            </button>
          ))}
        </div>
        <p className="mt-3 text-[9px] mono text-white/30 uppercase tracking-wider">
          Answers come from the event log only. Event details (type, zone, time, track) are sent to {providerLabel || 'the LLM provider'}; images and clips never are.
        </p>
      </DashboardPanel>

      {(asked || running) && (
        <DashboardPanel
          title={asked || 'Answer'}
          headerAction={result ? `${result.latency_s?.toFixed(1)} s // ${result.steps} turns // ${(result.usage.input_tokens + result.usage.output_tokens).toLocaleString()} tokens` : running ? 'WORKING' : null}
        >
          <div className="space-y-3">
            <Steps steps={steps} running={running && !result} />
            {error && <p className="text-[11px] mono text-orange-300">{error}</p>}
            {result && (
              <>
                {result.answer ? <Answer text={result.answer} /> : <EmptyState>No answer.</EmptyState>}
                {RUN_ENDINGS[result.stop] && (
                  <p className="text-[10px] mono text-orange-300/80">
                    {RUN_ENDINGS[result.stop]}{result.error ? ` (${result.error})` : ''}
                  </p>
                )}
                {result.removed_citations?.length > 0 && (
                  <p className="text-[9px] mono text-white/30">
                    Removed {result.removed_citations.length} citation(s) to events the search never looked up.
                  </p>
                )}
                <CitedEvents events={result.citations} />
              </>
            )}
          </div>
        </DashboardPanel>
      )}
    </div>
  );
};

export default SearchPage;
