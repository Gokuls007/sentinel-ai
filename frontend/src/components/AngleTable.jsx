import { angleRows, partLabel } from '../lib/ergonomics';

/**
 * Joint angles (degrees) from ergonomics data. Trunk, neck and upper arm are signed
 * (flex/ext); lower arm and knee are flexion (0 = straight).
 */
const AngleTable = ({ angles, estimated, caption, className = 'text-[9px]' }) => (
  <table className={`w-full mono ${className}`}>
    {caption && <caption className="text-left text-[9px] uppercase text-white/40 mb-1">{caption}</caption>}
    <thead>
      <tr className="text-white/30 uppercase text-[8px]">
        <th scope="col" className="text-left font-normal py-0.5">Joint</th>
        <th scope="col" className="text-right font-normal py-0.5">Left</th>
        <th scope="col" className="text-right font-normal py-0.5">Right</th>
      </tr>
    </thead>
    <tbody>
      {angleRows(angles).map(([label, left, right]) => (
        <tr key={label} className="border-t border-white/5">
          <th scope="row" className="text-left font-normal text-white/50 py-0.5">{label}</th>
          {right == null ? (
            <td colSpan={2} className="text-right text-cyan-100/80 tabular-nums py-0.5">{left}</td>
          ) : (
            <>
              <td className="text-right text-cyan-100/80 tabular-nums py-0.5">{left}</td>
              <td className="text-right text-cyan-100/80 tabular-nums py-0.5">{right}</td>
            </>
          )}
        </tr>
      ))}
      {Array.isArray(estimated) && estimated.length > 0 && (
        <tr className="border-t border-white/5">
          <td colSpan={3} className="text-white/35 py-0.5">Estimated (not visible): {estimated.map(partLabel).join(', ')}</td>
        </tr>
      )}
    </tbody>
  </table>
);

export default AngleTable;
