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
          <p className="text-[10px] mono text-white/40">Confidence threshold {data.confidence} · model {data.model}</p>
        </div>
      )}
    </DashboardPanel>
  );
};

export default ObjectsPanel;
