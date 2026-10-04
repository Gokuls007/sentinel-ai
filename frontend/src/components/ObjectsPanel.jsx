import { useState } from 'react';
import DashboardPanel from './DashboardPanel';
import { describeError, putJson, useFetch } from '../lib/api';
import { chipClass, inputClass } from '../lib/ui';

/** Warehouse objects (YOLO-World): the class list is plain text, re-encoded once when changed. */
const ObjectsPanel = () => {
  const [reload, setReload] = useState(0);
  const { data, error } = useFetch('/api/objects', reload);
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
      await putJson('/api/objects', { classes });
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
    <DashboardPanel title="Warehouse objects" headerAction={data ? (data.error ? 'OFF' : `${data.classes.length} CLASSES`) : null}>
      {!data ? (
        <p className="text-[11px] text-white/50 outfit">{error ? describeError(error) : 'Loading...'}</p>
      ) : (
        <div className="space-y-2">
          <p className="text-[10px] text-white/50 outfit max-w-2xl">
            Besides people, Warehouse mode looks for these objects (YOLO-World, open vocabulary: plain words, no
            training). The unsafe-lift rule uses &quot;cardboard box&quot; and the standing-on-a-chair rule uses &quot;chair&quot;.
            New names take a few seconds to encode once; after that they load from a cache.
          </p>
          {data.error && <p className="text-[11px] text-amber-300 outfit">Object detection is off: {data.error}</p>}
          <label className="block text-[10px] mono uppercase text-white/50">
            Classes (comma-separated)
            <textarea rows={2} value={value} onChange={(e) => { setText(e.target.value); setSaved(false); }}
              className={`${inputClass} w-full mt-1 normal-case`} />
          </label>
          <div className="flex flex-wrap gap-2">
            <button type="button" disabled={busy || !parsed.length || text === null} onClick={() => save(parsed)}
              className={chipClass(true)}>
              {busy ? 'Encoding...' : 'Save classes'}
            </button>
            <button type="button" disabled={busy} onClick={() => save(data.defaults)} className={chipClass(false)}>
              Reset to defaults
            </button>
          </div>
          {saved && <p className="text-[11px] text-emerald-300 outfit">Saved.</p>}
          {saveError && <p className="text-[11px] text-red-300 break-words">{saveError}</p>}
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
