import DashboardPanel from './DashboardPanel';

/** Placeholder for pages planned in a later phase. */
const ComingSoon = ({ title, phase, children }) => (
  <div className="max-w-2xl">
    <DashboardPanel title={title} headerAction={phase}>
      <p className="text-sm text-white/70 outfit leading-relaxed">{children}</p>
    </DashboardPanel>
  </div>
);

export default ComingSoon;
