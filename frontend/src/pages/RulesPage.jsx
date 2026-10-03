import { useState } from 'react';
import { Check, Loader2, Power, Sparkles, Trash2, TriangleAlert, X } from 'lucide-react';
import DashboardPanel from '../components/DashboardPanel';
import EmptyState from '../components/EmptyState';
import { deleteJson, describeError, formatTs, patchJson, postJson, useFetch } from '../lib/api';
import { SEVERITIES, severityBadge } from '../lib/severity';
import { chipClass, inputClass, labelClass } from '../lib/ui';

const EXAMPLES = [
  'Alert if someone stays in the loading dock for more than 30 seconds',
  'Notify me if anyone enters chemical storage between 10pm and 6am',
  'Flag anyone holding a phone in the forklift lane for 5 seconds',
  'Warn when someone works in a high-risk posture for 20 seconds',
];
const ACTION_LABELS = { record_clip: 'Save a clip', notify: 'Send a notification' };

const Severity = ({ value }) => (
  <span className={`px-1.5 py-px border text-[9px] mono uppercase font-bold ${severityBadge(value)}`}>{value}</span>
);

/** The compiled draft: plain-words preview, what to check, and the settings you can adjust. */
const Draft = ({ draft, onConfirm, onCancel, busy }) => {
  const [rule, setRule] = useState(draft.rule);
  const set = (patch) => setRule((r) => ({ ...r, ...patch }));
  const toggleAction = (a) => set({
    actions: rule.actions.includes(a) ? rule.actions.filter((x) => x !== a) : [...rule.actions, a],
  });
  return (
    <div className="border-2 border-cyan-400/50 bg-cyan-950/20 p-4 space-y-3" role="region" aria-label="Compiled rule">
      <div className="text-[10px] mono uppercase tracking-widest text-white/40">Check this before it runs</div>
      <div className="text-lg font-semibold outfit text-white">{rule.name}</div>
      <p className="mono text-sm text-cyan-200">{draft.preview}</p>
      {draft.warnings.length > 0 && (
        <ul className="space-y-1">
          {draft.warnings.map((w) => (
            <li key={w} className="flex gap-2 text-[11px] text-amber-200 outfit">
              <TriangleAlert className="w-3.5 h-3.5 shrink-0 mt-0.5" aria-hidden="true" /> {w}
            </li>
          ))}
        </ul>
      )}
      <div className="grid sm:grid-cols-3 gap-3">
        <label className="block">
          <span className={labelClass}>Must hold for (s)</span>
          <input type="number" min="0" max="3600" value={rule.duration_s} className={inputClass}
            onChange={(e) => set({ duration_s: Number(e.target.value) })} />
        </label>
        <label className="block">
          <span className={labelClass}>Severity</span>
          <select value={rule.severity} className={inputClass} onChange={(e) => set({ severity: e.target.value })}>
            {SEVERITIES.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
        </label>
        <label className="block">
          <span className={labelClass}>Wait before alerting again (s)</span>
          <input type="number" min="0" max="86400" value={rule.cooldown_s} className={inputClass}
            onChange={(e) => set({ cooldown_s: Number(e.target.value) })} />
        </label>
      </div>
      <div className="flex flex-wrap gap-2">
        {Object.entries(ACTION_LABELS).map(([a, label]) => (
          <button key={a} type="button" aria-pressed={rule.actions.includes(a)} onClick={() => toggleAction(a)}
            className={chipClass(rule.actions.includes(a))}>{label}</button>
        ))}
      </div>
      <p className="text-[10px] text-white/40 outfit">
        To change what the rule checks, edit the sentence and compile again. Nothing runs until you confirm.
      </p>
      <div className="flex gap-2">
        <button type="button" disabled={busy} onClick={() => onConfirm(rule)}
          className="px-4 py-2 text-[11px] mono uppercase font-bold tracking-widest border border-cyan-400 text-black bg-cyan-400 hover:bg-cyan-300 disabled:opacity-40 flex items-center gap-2">
          <Check className="w-4 h-4" aria-hidden="true" /> Confirm and turn on
        </button>
        <button type="button" disabled={busy} onClick={onCancel} className={`${chipClass(false)} flex items-center gap-1.5`}>
          <X className="w-3 h-3" aria-hidden="true" /> Cancel
        </button>
      </div>
    </div>
  );
};

const Composer = ({ status, onSaved }) => {
  const [text, setText] = useState('');
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const compileOk = status?.compile?.enabled;

  const compile = async (e) => {
    e?.preventDefault();
    if (!text.trim()) return;
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      setResult(await postJson('/api/rules/compile', { text: text.trim() }));
    } catch (err) {
      setError(err.detail || describeError(err));
    } finally {
      setBusy(false);
    }
  };
  const confirm = async (rule) => {
    setBusy(true);
    setError(null);
    try {
      await postJson('/api/rules', { text: text.trim(), rule, camera_ids: result.camera_ids });
      setResult(null);
      setText('');
      onSaved();
    } catch (err) {
      setError(err.detail || describeError(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <DashboardPanel title="New rule" headerAction={status?.compile?.model ? `${status.compile.provider} · ${status.compile.model}` : null}>
      <div className="space-y-3">
        {!compileOk && status && (
          <p className="text-[11px] text-amber-300 outfit">
            Writing new rules needs an LLM key: {status.compile.reason}. Existing rules and presets still work.
          </p>
        )}
        <form onSubmit={compile} className="flex gap-2">
          <label className="sr-only" htmlFor="rule-text">Describe the rule</label>
          <input id="rule-text" value={text} maxLength={300} onChange={(e) => setText(e.target.value)}
            placeholder="Describe the rule in plain English..." className={`${inputClass} flex-1`} disabled={!compileOk} />
          <button type="submit" disabled={busy || !compileOk || !text.trim()}
            className="px-4 py-2 text-[11px] mono uppercase font-bold tracking-widest border border-cyan-400 text-black bg-cyan-400 hover:bg-cyan-300 disabled:opacity-40 flex items-center gap-2">
            {busy ? <Loader2 className="w-4 h-4 animate-spin" aria-hidden="true" /> : <Sparkles className="w-4 h-4" aria-hidden="true" />}
            Compile
          </button>
        </form>
        <div className="flex flex-wrap gap-1.5">
          {EXAMPLES.map((ex) => (
            <button key={ex} type="button" onClick={() => setText(ex)} disabled={!compileOk}
              className="text-[10px] outfit text-white/55 hover:text-cyan-200 border border-white/10 px-2 py-1 text-left">
              {ex}
            </button>
          ))}
        </div>
        <p className="text-[10px] text-white/35 outfit">
          Only your sentence and the zone names go to the LLM provider, once. Rules are then checked on every frame by
          code on this computer.
        </p>
        {error && <p className="text-[11px] text-red-300">{error}</p>}
        {result?.status === 'refusal' && (
          <div className="p-3 border border-amber-400/40 bg-amber-950/20 text-sm text-amber-100 outfit">
            <div className="font-semibold">This can&apos;t be checked yet</div>
            <p>{result.refusal}</p>
          </div>
        )}
        {result?.status === 'error' && <p className="text-[11px] text-red-300">Couldn&apos;t compile: {result.error}</p>}
        {result?.status === 'rule' && (
          <Draft key={result.preview} draft={result} busy={busy} onConfirm={confirm} onCancel={() => setResult(null)} />
        )}
      </div>
    </DashboardPanel>
  );
};

const RuleRow = ({ rule, onChanged, readOnly }) => {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const act = async (fn) => {
    setBusy(true);
    setError(null);
    try {
      await fn();
      onChanged();
    } catch (err) {
      setError(err.detail || describeError(err));
    } finally {
      setBusy(false);
    }
  };
  return (
    <li className={`border p-3 space-y-1.5 ${rule.enabled ? 'border-white/15 bg-white/5' : 'border-white/5 bg-black/30 opacity-60'}`}>
      <div className="flex items-center justify-between gap-2 flex-wrap">
        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold outfit text-white">{rule.name}</span>
          <Severity value={rule.severity} />
          {rule.preset && <span className="text-[9px] mono uppercase text-white/40">preset</span>}
          {rule.builtin && <span className="text-[9px] mono uppercase text-white/40">built in</span>}
        </div>
        {!readOnly && (
          <div className="flex gap-1.5">
            <button type="button" disabled={busy} aria-pressed={rule.enabled}
              onClick={() => act(() => patchJson(`/api/rules/${rule.id}`, { enabled: !rule.enabled }))}
              className={`${chipClass(rule.enabled)} flex items-center gap-1`}>
              <Power className="w-3 h-3" aria-hidden="true" /> {rule.enabled ? 'On' : 'Off'}
            </button>
            <button type="button" disabled={busy} aria-label={`Delete ${rule.name}`}
              onClick={() => { if (window.confirm(`Delete "${rule.name}"?`)) act(() => deleteJson(`/api/rules/${rule.id}`)); }}
              className={`${chipClass(false)} flex items-center gap-1`}>
              <Trash2 className="w-3 h-3" aria-hidden="true" />
            </button>
          </div>
        )}
      </div>
      <p className="mono text-[11px] text-cyan-200/90">{rule.preview}</p>
      {rule.source_text && <p className="text-[11px] text-white/45 outfit">&ldquo;{rule.source_text}&rdquo;</p>}
      <p className="text-[10px] mono text-white/40">
        {rule.camera_ids?.length ? rule.camera_ids.join(', ') : 'all cameras'} · fired {rule.fired ?? 0}×
        {rule.last_fired ? ` · last ${formatTs(rule.last_fired)}` : ''}
      </p>
      {error && <p className="text-[11px] text-red-300">{error}</p>}
    </li>
  );
};

const Presets = ({ onApplied }) => {
  const { data } = useFetch('/api/presets');
  const [msg, setMsg] = useState(null);
  const [busy, setBusy] = useState(false);
  const apply = async (name, replace = false) => {
    setBusy(true);
    setMsg(null);
    try {
      const r = await postJson(`/api/presets/${name}/apply`, { replace });
      setMsg(`Loaded ${r.rules.length} rules from the preset.`);
      onApplied();
    } catch (err) {
      if (err.status === 409 && window.confirm(`${err.detail}. Replace the preset rules?`)) {
        setBusy(false);
        apply(name, true);
        return;
      }
      setMsg(err.detail || describeError(err));
    } finally {
      setBusy(false);
    }
  };
  return (
    <DashboardPanel title="Presets">
      <div className="space-y-2">
        <p className="text-[11px] text-white/55 outfit">A ready-made set of rules. You can edit or turn off any of them after.</p>
        <div className="flex flex-wrap gap-2">
          {(data?.presets || []).map((p) => (
            <button key={p.name} type="button" disabled={busy || !p.available} onClick={() => apply(p.name)}
              title={p.note || ''} className={chipClass(p.available)}>
              {p.label}{p.available ? '' : ' (later)'}
            </button>
          ))}
        </div>
        {msg && <p className="text-[11px] text-white/70 outfit">{msg}</p>}
      </div>
    </DashboardPanel>
  );
};

const RulesPage = () => {
  const [reload, setReload] = useState(0);
  const bump = () => setReload((n) => n + 1);
  const { data: status } = useFetch('/api/rules/status', reload);
  const { data, error } = useFetch('/api/rules', reload);
  const rules = data?.rules || [];
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-12 gap-4">
        <div className="col-span-12 lg:col-span-8 space-y-4">
          <Composer status={status} onSaved={bump} />
          <DashboardPanel title="Your rules" headerAction={`${rules.filter((r) => r.enabled).length} ON`}>
            {error ? (
              <p className="text-[11px] text-red-300">{describeError(error)}</p>
            ) : rules.length ? (
              <ul className="space-y-2">{rules.map((r) => <RuleRow key={r.id} rule={r} onChanged={bump} />)}</ul>
            ) : (
              <EmptyState>No rules yet. Describe one above, or load the Warehouse safety preset.</EmptyState>
            )}
          </DashboardPanel>
        </div>
        <div className="col-span-12 lg:col-span-4 space-y-4">
          <Presets onApplied={bump} />
          <DashboardPanel title="Built-in alerts">
            <div className="space-y-2">
              <p className="text-[11px] text-white/55 outfit">
                {status?.builtins
                  ? 'Falls and restricted zones run as the built-in rules below.'
                  : 'Falls and restricted zones use the original detectors. Set RULES_BUILTINS=true to run them as rules instead.'}
              </p>
              {data?.builtins?.length > 0 && (
                <ul className="space-y-2">{data.builtins.map((r) => <RuleRow key={r.id} rule={r} readOnly />)}</ul>
              )}
            </div>
          </DashboardPanel>
        </div>
      </div>
    </div>
  );
};

export default RulesPage;
