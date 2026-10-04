import { useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { CheckCircle2, Circle, Plus, ScanSearch, Trash2, XCircle } from 'lucide-react';
import DashboardPanel from '../components/DashboardPanel';
import SeatEditor from '../components/SeatEditor';
import { useAppMode } from '../context/appMode';
import { describeError, fetchJson, postJson, putJson, usePoll } from '../lib/api';
import { useCameraFrames } from '../lib/laptopCamera';
import { chipClass, inputClass } from '../lib/ui';

// Exam Hall, stage E1: sessions, the setup check, the seat map and per-seat calibration.
// Flags come in stage E2. People are seats (A1, B2...), never names or identities.

const SEAT_LABEL = /^[A-Za-z]{1,2}[0-9]{1,2}$/;
const STATUS_TEXT = {
  setup: 'Setup',
  calibrating: 'Calibrating',
  live: 'Live',
  ended: 'Ended',
  reviewed: 'Reviewed',
};

const mmss = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;

const NewSession = ({ cameras, onCreated }) => {
  const [name, setName] = useState('');
  const [room, setRoom] = useState('');
  const [camera, setCamera] = useState('laptop');
  const [calibration, setCalibration] = useState(120);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const create = async (e) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const s = await postJson('/api/exam/sessions', { name, room, camera_id: camera, calibration_s: Number(calibration) });
      setName('');
      onCreated(s);
    } catch (err) {
      setError(err.detail || describeError(err));
    } finally {
      setBusy(false);
    }
  };
  return (
    <form onSubmit={create} className="flex flex-wrap items-end gap-2">
      <label className="text-[10px] mono uppercase text-white/50 flex flex-col gap-1">
        Name
        <input required maxLength={80} value={name} onChange={(e) => setName(e.target.value)} className={inputClass}
          placeholder="Maths mock, Tuesday" />
      </label>
      <label className="text-[10px] mono uppercase text-white/50 flex flex-col gap-1">
        Room
        <input maxLength={80} value={room} onChange={(e) => setRoom(e.target.value)} className={inputClass} placeholder="R101" />
      </label>
      <label className="text-[10px] mono uppercase text-white/50 flex flex-col gap-1">
        Camera
        <select value={camera} onChange={(e) => setCamera(e.target.value)} className={inputClass}>
          {(cameras || [{ id: 'laptop' }]).map((c) => <option key={c.id} value={c.id}>{c.id === 'laptop' ? 'Laptop camera' : c.id === 'cam-0' ? 'Demo footage (cam-0)' : c.id}</option>)}
        </select>
      </label>
      <label className="text-[10px] mono uppercase text-white/50 flex flex-col gap-1">
        Calibration (s)
        <input type="number" min={30} max={600} value={calibration} onChange={(e) => setCalibration(e.target.value)}
          className={`${inputClass} w-24`} />
      </label>
      <button type="submit" disabled={busy || !name.trim()} className={`${chipClass(true)} flex items-center gap-1`}>
        <Plus className="w-3 h-3" aria-hidden="true" /> New exam session
      </button>
      {error && <p className="w-full text-[11px] text-red-300">{error}</p>}
    </form>
  );
};

const SetupChecklist = ({ exam }) => {
  const setup = exam?.setup;
  if (!setup) return <p className="text-[11px] text-white/50 outfit">Waiting for video...</p>;
  return (
    <ul className="space-y-1.5">
      <li className="text-[11px] outfit text-white/80">
        People in view: <span className="mono text-cyan-300">{setup.people}</span>
        {exam.staff > 0 && <span className="text-white/50"> ({exam.staff} in no seat: ignored as staff)</span>}
      </li>
      {setup.checks.map((c) => (
        <li key={c.id} className="flex items-start gap-2 text-[11px] outfit">
          {c.ok ? <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 flex-none mt-0.5" aria-hidden="true" />
            : <XCircle className="w-3.5 h-3.5 text-amber-300 flex-none mt-0.5" aria-hidden="true" />}
          <span>
            <span className={c.ok ? 'text-white/80' : 'text-amber-200'}>{c.label}</span>
            {c.message && <span className="block text-amber-200/80">{c.message}</span>}
          </span>
        </li>
      ))}
      {setup.checks.length === 0 && <li className="text-[11px] text-white/50 outfit">Nobody in view yet.</li>}
    </ul>
  );
};

const SeatStatus = ({ seats }) => (
  <ul className="grid grid-cols-2 sm:grid-cols-4 gap-1.5">
    {seats.map((s) => (
      <li key={s.label} className="border border-white/10 px-2 py-1 text-[10px] mono">
        <span className="font-bold text-white">{s.label}</span>{' '}
        {!s.occupied ? <span className="text-white/40">empty</span>
          : s.calibrated ? <span className="text-emerald-300">calibrated</span>
            : <span className="text-cyan-300">{Math.round(s.calibration_progress * 100)}%</span>}
      </li>
    ))}
  </ul>
);

const SessionView = ({ session, sessions, onChanged }) => {
  const { mode, setMode } = useAppMode();
  const { status: feed, frame, frameData } = useCameraFrames(session.camera_id);
  const exam = frameData?.exam && frameData.exam.session_id === session.id ? frameData.exam : null;
  const editable = session.status === 'setup';
  const [draft, setDraft] = useState(session.seats);
  const [dirty, setDirty] = useState(false);
  const [selected, setSelected] = useState(null);
  const [message, setMessage] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const [copyFrom, setCopyFrom] = useState('');

  // The session is refetched every few seconds: take the saved seats only when they really
  // changed, and never over unsaved edits.
  const savedKey = JSON.stringify(session.seats.map(({ label, rect }) => [label, rect]));
  const [lastSaved, setLastSaved] = useState(savedKey);
  if (savedKey !== lastSaved && !dirty) {
    setLastSaved(savedKey);
    setDraft(session.seats);
    setSelected(null);
  }

  const states = useMemo(() => Object.fromEntries((exam?.seats || []).map((s) => [s.label, s])), [exam]);
  const change = (seats) => { setDraft(seats); setDirty(true); };

  const act = async (fn) => {
    setBusy(true);
    setError(null);
    setMessage(null);
    try {
      await fn();
    } catch (err) {
      setError(err.detail || describeError(err));
    } finally {
      setBusy(false);
    }
  };

  const save = () => act(async () => {
    const bad = draft.find((s) => !SEAT_LABEL.test(s.label));
    if (bad) throw Object.assign(new Error(), { detail: `"${bad.label}" isn't a seat label (like A1 or B12).` });
    const s = await putJson(`/api/exam/sessions/${session.id}/seats`, { seats: draft.map(({ label, rect }) => ({ label, rect })) });
    setDirty(false);
    onChanged(s);
    setMessage('Seats saved.');
  });

  const detect = () => act(async () => {
    const r = await postJson(`/api/exam/sessions/${session.id}/detect-seats`, {});
    if (!r.seats.length) { setMessage(r.message); return; }
    change(r.seats);
    setMessage(`${r.seats.length} seat(s) found. Adjust them if needed, then Save seats.`);
  });

  const addSeat = () => {
    const used = new Set(draft.map((s) => s.label));
    let n = 1;
    while (used.has(`X${n}`)) n += 1;
    change([...draft, { label: `X${n}`, rect: [0.42, 0.4, 0.58, 0.65] }]);
    setSelected(draft.length);
  };

  const copy = () => act(async () => {
    const s = await postJson(`/api/exam/sessions/${session.id}/copy-seats`, { from_session_id: Number(copyFrom) });
    onChanged(s);
    setMessage('Seats copied.');
  });

  const start = () => act(async () => onChanged(await postJson(`/api/exam/sessions/${session.id}/start`, {})));
  const end = () => act(async () => onChanged(await postJson(`/api/exam/sessions/${session.id}/end`, {})));

  const sel = selected != null ? draft[selected] : null;
  const others = sessions.filter((s) => s.id !== session.id);
  const status = exam?.status || session.status;

  return (
    <div className="space-y-4">
      {mode !== 'exam' && (
        <div className="border border-amber-400/40 bg-amber-950/30 p-2 text-[11px] outfit text-amber-200 flex flex-wrap items-center gap-2">
          The app is in another mode, so the exam monitor isn&apos;t running.
          <button type="button" className={chipClass(true)} onClick={() => setMode('exam')}>Switch to Exam Hall</button>
        </div>
      )}
      {feed !== 'live' && (
        <p className="text-[11px] outfit text-white/60">
          Waiting for the {session.camera_id === 'laptop' ? 'laptop camera' : `"${session.camera_id}" camera`}.{' '}
          {session.camera_id === 'laptop' && <>Start it on the <Link to="/camera" className="underline text-cyan-300">Camera page</Link>.</>}
          {session.camera_id === 'cam-0' && 'Turn on demo footage in Settings.'}
        </p>
      )}
      {status === 'calibrating' && exam?.calibration_remaining_s != null && (
        <div className="border border-cyan-400/40 bg-cyan-950/30 p-2 text-[12px] outfit text-cyan-100" role="status">
          Calibrating — {mmss(exam.calibration_remaining_s)} left. Each seat learns its own normal head and hand
          position; nothing is flagged during calibration.
        </div>
      )}
      {status === 'live' && (
        <div className="border border-emerald-400/30 bg-emerald-950/20 p-2 text-[12px] outfit text-emerald-100" role="status">
          Exam running. Seats that are taken late calibrate on their own over their first {mmss(session.calibration_s)}.
          Flags for review come in the next stage.
        </div>
      )}

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
        <div className="xl:col-span-2 space-y-2">
          <SeatEditor frame={frame} seats={draft} editable={editable} selected={selected} onSelect={setSelected}
            onChange={change} states={states} />
          {editable && (
            <div className="flex flex-wrap items-center gap-2">
              <button type="button" disabled={busy} onClick={detect} className={`${chipClass(false)} flex items-center gap-1`}>
                <ScanSearch className="w-3 h-3" aria-hidden="true" /> Detect seats
              </button>
              <button type="button" disabled={busy} onClick={addSeat} className={`${chipClass(false)} flex items-center gap-1`}>
                <Plus className="w-3 h-3" aria-hidden="true" /> Add seat
              </button>
              {others.length > 0 && (
                <>
                  <select aria-label="Copy seats from" value={copyFrom} onChange={(e) => setCopyFrom(e.target.value)} className={inputClass}>
                    <option value="">Copy seats from...</option>
                    {others.map((s) => <option key={s.id} value={s.id}>{s.name}{s.room ? ` (${s.room})` : ''}</option>)}
                  </select>
                  <button type="button" disabled={busy || !copyFrom} onClick={copy} className={chipClass(false)}>Copy</button>
                </>
              )}
              <button type="button" disabled={busy || !dirty} onClick={save} className={chipClass(true)}>
                Save seats{dirty ? ' *' : ''}
              </button>
            </div>
          )}
          {sel && editable && (
            <div className="flex flex-wrap items-center gap-2 border border-white/10 p-2">
              <label className="text-[10px] mono uppercase text-white/50 flex items-center gap-2">
                Label
                <input value={sel.label} maxLength={4} className={`${inputClass} w-20`}
                  aria-invalid={!SEAT_LABEL.test(sel.label)}
                  onChange={(e) => change(draft.map((s, i) => (i === selected ? { ...s, label: e.target.value.toUpperCase() } : s)))} />
              </label>
              <button type="button" onClick={() => { change(draft.filter((_s, i) => i !== selected)); setSelected(null); }}
                className={`${chipClass(false)} flex items-center gap-1`}>
                <Trash2 className="w-3 h-3" aria-hidden="true" /> Delete seat
              </button>
              <span className="text-[10px] text-white/40 outfit">Drag to move, drag the corner to resize.</span>
            </div>
          )}
          {message && <p className="text-[11px] text-emerald-300 outfit">{message}</p>}
          {error && <p className="text-[11px] text-red-300">{error}</p>}
        </div>

        <div className="space-y-4">
          <DashboardPanel title="Setup check" headerAction={exam?.setup?.ok ? 'READY' : null}>
            <SetupChecklist exam={exam} />
          </DashboardPanel>
          <DashboardPanel title="Seats" headerAction={`${session.seats.length} SAVED`}>
            {exam?.seats?.length ? <SeatStatus seats={exam.seats} /> : (
              <p className="text-[11px] text-white/50 outfit">
                {session.seats.length ? 'Waiting for video...' : 'No seats yet: Detect seats (people sitting still for 5 s), or Add seat.'}
              </p>
            )}
          </DashboardPanel>
          <div className="flex flex-wrap gap-2">
            {session.status === 'setup' && (
              <button type="button" disabled={busy || dirty || !session.seats.length} onClick={start} className={chipClass(true)}>
                Start exam
              </button>
            )}
            {['setup', 'calibrating', 'live'].includes(session.status) && (
              <button type="button" disabled={busy} onClick={end} className={chipClass(false)}>End exam</button>
            )}
          </div>
          {session.status === 'setup' && dirty && <p className="text-[10px] text-amber-200 outfit">Save the seats before starting.</p>}
        </div>
      </div>
    </div>
  );
};

const ExamPage = () => {
  const [reload, setReload] = useState(0);
  const [sessions, setSessions] = useState(null);
  const [current, setCurrent] = useState(null);
  const [error, setError] = useState(null);
  const { data: cameras } = usePoll('/api/cameras', 10000);

  useEffect(() => {
    fetchJson('/api/exam/sessions')
      .then((d) => {
        setSessions(d.sessions);
        setCurrent((c) => c ?? d.sessions.find((s) => ['setup', 'calibrating', 'live'].includes(s.status))?.id ?? d.sessions[0]?.id ?? null);
      })
      .catch((err) => setError(describeError(err)));
  }, [reload]);

  const [session, setSession] = useState(null);
  useEffect(() => {
    if (current == null) return undefined;
    let cancelled = false;
    const load = () => fetchJson(`/api/exam/sessions/${current}`).then((s) => !cancelled && setSession(s)).catch(() => {});
    load();
    const id = setInterval(load, 5000); // status changes (calibrating -> live) and saved baselines
    return () => { cancelled = true; clearInterval(id); };
  }, [current, reload]);

  const changed = (s) => { setSession(s); setCurrent(s.id); setReload((n) => n + 1); };

  return (
    <div className="space-y-4">
      <DashboardPanel title="Exam Hall" headerAction={session && current != null ? STATUS_TEXT[session.status]?.toUpperCase() : null}>
        <div className="space-y-3">
          <p className="text-[11px] outfit text-white/60 max-w-3xl">
            Exam Hall flags moments for a human invigilator to review. It never decides what happened, never
            identifies anyone (people are seat labels like A1), and blurs faces in everything it saves.
          </p>
          <NewSession cameras={cameras} onCreated={changed} />
          {sessions && sessions.length > 0 && (
            <label className="text-[10px] mono uppercase text-white/50 flex items-center gap-2">
              Session
              <select value={current ?? ''} onChange={(e) => setCurrent(Number(e.target.value))} className={inputClass}>
                {sessions.map((s) => (
                  <option key={s.id} value={s.id}>{s.name}{s.room ? ` (${s.room})` : ''}: {STATUS_TEXT[s.status]}</option>
                ))}
              </select>
            </label>
          )}
          {error && <p className="text-[11px] text-red-300">{error}</p>}
        </div>
      </DashboardPanel>
      {session && current != null ? <SessionView key={session.id} session={session} sessions={sessions || []} onChanged={changed} /> : (
        <p className="text-[11px] outfit text-white/50 flex items-center gap-2">
          <Circle className="w-3 h-3" aria-hidden="true" /> Create an exam session to set up the seats.
        </p>
      )}
    </div>
  );
};

export default ExamPage;
