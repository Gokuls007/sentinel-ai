import DashboardPanel from '../components/DashboardPanel';
import EmptyState from '../components/EmptyState';
import { MODES, useAppMode } from '../context/appMode';
import { useFeed } from '../context/liveFeed';
import { describeError, useFetch } from '../lib/api';
import { chipClass } from '../lib/ui';

const Row = ({ label, children }) => (
  <div className="flex justify-between gap-4 py-1.5 border-b border-white/5 text-[10px] mono">
    <dt className="text-white/40 uppercase">{label}</dt>
    <dd className="text-cyan-100 text-right break-all">{children}</dd>
  </div>
);

const fmt = (v) => {
  if (v == null || v === '') return '--';
  if (typeof v === 'boolean') return v ? 'yes' : 'no';
  if (Array.isArray(v)) return v.join(', ');
  if (typeof v === 'object') return JSON.stringify(v);
  return String(v);
};

const humanize = (k) => k.replace(/_/g, ' ');

const ChannelState = ({ enabled }) =>
  enabled ? (
    <span className="text-green-400 font-bold">ENABLED</span>
  ) : (
    <span className="text-white/40">not configured</span>
  );

/** Mode and demo footage: the two settings you can change here. */
const ModeAndFootage = () => {
  const { mode, demoFootage, setMode, setDemoFootage, error } = useAppMode();
  return (
    <DashboardPanel title="Mode and video source">
      <div className="space-y-4">
        <div>
          <div className="text-[9px] uppercase font-bold text-white/40 tracking-widest mb-1.5">Mode</div>
          <div className="flex flex-wrap gap-1.5">
            {MODES.map((m) => (
              <button key={m.id} type="button" aria-pressed={mode === m.id} onClick={() => setMode(m.id)}
                className={chipClass(mode === m.id)}>
                {m.label}
              </button>
            ))}
          </div>
          <p className="text-[10px] text-white/50 mt-1.5 outfit">
            The mode decides what the camera analyses and which alerts can fire. Desk Posture Coach runs only the
            posture coach; Warehouse Safety runs falls, zones, loitering and ergonomics.
          </p>
        </div>
        <div className="flex items-start justify-between gap-4 pt-3 border-t border-white/5">
          <div>
            <div className="text-[12px] font-bold text-white outfit">Demo footage</div>
            <p className="text-[10px] text-white/50 outfit max-w-md">
              Off: your laptop webcam is the source everywhere. On: Warehouse Safety's Live page shows the
              bundled sample videos instead (useful without a camera). The sample pipeline is paused while off.
            </p>
          </div>
          <button type="button" role="switch" aria-checked={demoFootage} onClick={() => setDemoFootage(!demoFootage)}
            className={`relative flex-none w-11 h-6 border transition-colors ${demoFootage ? 'bg-cyan-400 border-cyan-300' : 'bg-black/60 border-white/20'}`}>
            <span className="sr-only">Demo footage</span>
            <span className={`absolute top-0.5 w-4.5 h-4.5 bg-white transition-all ${demoFootage ? 'left-[22px]' : 'left-0.5'}`}
              style={{ width: 18, height: 18 }} aria-hidden="true" />
          </button>
        </div>
        {error && <p className="text-[10px] text-red-300">{describeError(error)}</p>}
      </div>
    </DashboardPanel>
  );
};

const SettingsPage = () => {
  const { status } = useFeed();
  const { data: meta, error } = useFetch('/api/meta', status);

  const notice = (
    <>
      <ModeAndFootage />
      <p className="text-[11px] outfit text-white/60">
        The settings below are read-only here. Change them in the backend config and{' '}
        <code className="mono text-cyan-300">.env</code>, then restart the backend.
      </p>
    </>
  );

  if (!meta) {
    return (
      <div className="space-y-4 max-w-5xl">
        {notice}
        <DashboardPanel title="Settings">
          <EmptyState className="py-8">{error ? describeError(error) : 'Loading...'}</EmptyState>
        </DashboardPanel>
      </div>
    );
  }

  const n = meta.notifications || {};
  const t = meta.thresholds || {};
  const zones = meta.zones || [];

  return (
    <div className="space-y-4 max-w-5xl">
      {notice}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <DashboardPanel title="Camera">
          <dl>
            <Row label="Camera id">{fmt(meta.camera_id)}</Row>
            <Row label="Source">{fmt(meta.source)}</Row>
            <Row label="Known cameras">{fmt(meta.cameras)}</Row>
          </dl>
        </DashboardPanel>

        <DashboardPanel title="Notifications">
          <dl>
            <Row label="Telegram"><ChannelState enabled={n.telegram} /></Row>
            <Row label="Email"><ChannelState enabled={n.email} /></Row>
            <Row label="Webhook"><ChannelState enabled={n.webhook} /></Row>
            <Row label="Active channels">{fmt(n.channels?.length ? n.channels : 'none')}</Row>
            <Row label="Debounce">{n.debounce_s != null ? `${n.debounce_s}s` : '--'}</Row>
            <Row label="Min severity">{fmt(n.min_severity)}</Row>
          </dl>
          <div className="mt-3 p-2 bg-black/40 border border-cyan-500/10">
            <p className="text-[10px] outfit text-white/60 mb-1.5">
              To enable Telegram, add these to the <code className="mono text-cyan-300">.env</code> file in the
              project root and restart the backend (values are never shown here):
            </p>
            <pre className="text-[10px] mono text-cyan-300 whitespace-pre-wrap">{'TELEGRAM_BOT_TOKEN=<your bot token>\nTELEGRAM_CHAT_ID=<your chat id>'}</pre>
          </div>
        </DashboardPanel>

        <DashboardPanel title="Thresholds">
          <dl>
            <Row label="Detection confidence">{fmt(t.detection_confidence)}</Row>
            <Row label="Zone alert cooldown">{t.zone_alert_cooldown_s != null ? `${t.zone_alert_cooldown_s}s` : '--'}</Row>
            {Object.entries(t.fall || {}).map(([k, v]) => <Row key={`fall-${k}`} label={`Fall: ${humanize(k)}`}>{fmt(v)}</Row>)}
            {Object.entries(t.loiter || {}).map(([k, v]) => <Row key={`loiter-${k}`} label={`Loiter: ${humanize(k)}`}>{fmt(v)}</Row>)}
          </dl>
        </DashboardPanel>

        <DashboardPanel title="Zones" headerAction={`${zones.length} DEFINED`}>
          {zones.length === 0 ? (
            <EmptyState>No zones configured</EmptyState>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-[10px] mono">
                <thead>
                  <tr className="text-[9px] uppercase text-white/40 tracking-widest border-b border-white/10">
                    <th scope="col" className="py-1.5 pr-3">Id</th>
                    <th scope="col" className="py-1.5 pr-3">Name</th>
                    <th scope="col" className="py-1.5 pr-3">Type</th>
                    <th scope="col" className="py-1.5 pr-3">Time limit</th>
                    <th scope="col" className="py-1.5">Direction</th>
                  </tr>
                </thead>
                <tbody>
                  {zones.map((z) => (
                    <tr key={z.id} className="border-b border-white/5 text-cyan-100">
                      <td className="py-1.5 pr-3">{z.id}</td>
                      <td className="py-1.5 pr-3">{fmt(z.name)}</td>
                      <td className="py-1.5 pr-3 uppercase text-white/60">{fmt(z.type)}</td>
                      <td className="py-1.5 pr-3">{z.time_limit ? `${z.time_limit}s` : '--'}</td>
                      <td className="py-1.5">{fmt(z.direction)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </DashboardPanel>
      </div>
    </div>
  );
};

export default SettingsPage;
