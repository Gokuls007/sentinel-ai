import { useState } from 'react';
import DashboardPanel from './DashboardPanel';
import { describeError, putJson, useFetch } from '../lib/api';
import { chipClass, inputClass } from '../lib/ui';

const MODE_TABS = [
  { id: 'warehouse', label: 'Warehouse' },
  { id: 'home', label: 'Home care' },
  { id: 'exam', label: 'Exam Hall' },
  { id: 'posture', label: 'Desk posture' },
];

/** Objects per mode: each mode detects and draws only its own list (plain words). COCO classes
 * come from YOLO11m, the rest from YOLO-World (re-encoded once when changed, then cached). */
const ObjectsPanel = () => {
  const [reload, setReload] = useState(0);
  const [mode, setMode] = useState('warehouse');
  const { data, error } = useFetch(`/api/objects?mode=${mode}`, reload);
  const [text, setText] = useState(null);
  const [busy, setBusy] = useState(false);
  const [saveError, setSaveError] = useState(null);
  const [saved, setSaved] = useState(false);
  const value = text ?? (data ? data.classes.join(', ') : '');

  const save = async (classes) => {
    setBusy(true);
    setSaveError(null);
    setSaved(false);
    try {
      await putJson('/api/objects', { classes, mode });
      setText(null);
      setSaved(true);
      setReload((n) => n + 1);
    } catch (err) {
      setSaveError(err.detail ? JSON.stringify(err.detail) : describeError(err));
    } finally {
      setBusy(false);
    }
  };
  const parsed = value.split(',').map((c) => c.trim()).filter(Boolean);
  const [hazards, setHazards] = useState(null);
  const shownHazards = hazards ?? data?.hazards ?? [];
  const saveHazards = async () => {
    setBusy(true);
    setSaveError(null);
    try {
      await putJson('/api/objects/hazards', { hazards: shownHazards });
      setHazards(null);
      setReload((n) => n + 1);
    } catch (err) {
      setSaveError(err.detail ? JSON.stringify(err.detail) : describeError(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <DashboardPanel title="Objects per mode" headerAction={data ? (data.error ? 'OFF' : `${data.classes.length} CLASSES`) : null}>
      <div className="flex gap-1 mb-2" role="tablist" aria-label="Mode">
        {MODE_TABS.map((m) => (
          <button key={m.id} type="button" role="tab" aria-selected={mode === m.id}
            onClick={() => { setMode(m.id); setText(null); setHazards(null); setSaved(false); }} className={chipClass(mode === m.id)}>
            {m.label}
          </button>
        ))}
      </div>
      {!data ? (
        <p className="text-[11px] text-white/50 outfit">{error ? describeError(error) : 'Loading...'}</p>
      ) : (
        <div className="space-y-2">
          <p className="text-[10px] text-white/50 outfit max-w-2xl">
            Besides people, this mode detects and draws only these objects. COCO classes (mouse, cell phone, book...)
            come from YOLO11m; anything else is matched by YOLO-World from plain words, encoded once and then cached.
            {mode === 'warehouse' && <> The unsafe-lift rule uses &quot;cardboard box&quot; and the standing-on-a-chair rule uses &quot;chair&quot;.</>}
            {' '}An empty list means this mode detects no objects.
          </p>
          {data.error && <p className="text-[11px] text-amber-300 outfit">Object detection is off: {data.error}</p>}
          <label className="block text-[10px] mono uppercase text-white/50">
            Classes (comma-separated)
            <textarea rows={2} value={value} onChange={(e) => { setText(e.target.value); setSaved(false); }}
              className={`${inputClass} w-full mt-1 normal-case`} />
          </label>
          <div className="flex flex-wrap gap-2">
            <button type="button" disabled={busy || text === null} onClick={() => save(parsed)}
              className={chipClass(true)}>
              {busy ? 'Encoding...' : 'Save classes'}
            </button>
            <button type="button" disabled={busy} onClick={() => save(data.defaults)} className={chipClass(false)}>
              Reset to defaults
            </button>
          </div>
          {saved && <p className="text-[11px] text-emerald-300 outfit">Saved.</p>}
          {saveError && <p className="text-[11px] text-red-300 break-words">{saveError}</p>}
          {mode === 'warehouse' && (
            <label className="flex items-start gap-2 text-[11px] outfit text-white/70">
              <input type="checkbox" checked={Boolean(data.show_all)} disabled={busy}
                onChange={async (e) => {
                  setBusy(true);
                  try {
                    await putJson('/api/objects/show-all', { on: e.target.checked });
                    setReload((n) => n + 1);
                  } catch (err) {
                    setSaveError(describeError(err));
                  } finally {
                    setBusy(false);
                  }
                }} />
              <span>
                Show all objects (debug): detect and draw every class the detectors know (85) instead of this list.
                {data.show_all && <span className="text-amber-300"> On: warehouse is using all 85.</span>}
              </span>
            </label>
          )}
          {mode === 'warehouse' && (
          <fieldset className="border border-white/10 p-2">
            <legend className="px-1 text-[10px] mono uppercase text-white/50">
              Hazards: touching one alerts (&quot;Hand on knife (hazard)&quot;)
            </legend>
            <div className="flex flex-wrap gap-x-3 gap-y-1 max-h-40 overflow-y-auto">
              {data.classes.map((c) => (
                <label key={c} className="text-[11px] outfit text-white/70 flex items-center gap-1">
                  <input type="checkbox" checked={shownHazards.includes(c)}
                    onChange={(e) => setHazards(e.target.checked ? [...shownHazards, c] : shownHazards.filter((x) => x !== c))} />
                  {c}
                </label>
              ))}
            </div>
            <button type="button" disabled={busy || hazards === null} onClick={saveHazards} className={`${chipClass(true)} mt-2`}>
              Save hazards
            </button>
          </fieldset>
          )}
          {Object.keys(data.synonyms || {}).length > 0 && (
            <ul className="text-[10px] mono text-white/50 space-y-0.5">
              {Object.entries(data.synonyms).map(([cls, words]) => (
                <li key={cls}>
                  <span className="text-white/70">{cls}</span> &larr; {words.join(', ')}
                  {data.floors?.[cls] != null && <span className="text-white/40"> (floor {data.floors[cls]})</span>}
                </li>
              ))}
            </ul>
          )}
          <p className="text-[10px] mono text-white/40">
            Other classes: confidence {data.confidence} · model {data.model} · each object shows its most frequent
            label over the last 15 frames
          </p>
        </div>
      )}
    </DashboardPanel>
  );
};

export default ObjectsPanel;
