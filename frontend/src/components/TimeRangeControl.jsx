import { useId } from 'react';
import { PRESETS } from '../lib/timeRange';
import { chipClass, inputClass, labelClass } from '../lib/ui';


/** Quick presets (1h / 24h / 7d / all) plus custom from/to datetime-local inputs. */
const TimeRangeControl = ({ value, onChange }) => {
  const id = useId();
  const custom = value.preset === 'custom';
  return (
    <fieldset className="min-w-0">
      <legend className={labelClass}>Time range</legend>
      <div className="flex flex-wrap items-end gap-1.5">
        {PRESETS.map(([key, label]) => (
          <button
            key={key}
            type="button"
            aria-pressed={value.preset === key}
            onClick={() => onChange({ ...value, preset: key })}
            className={chipClass(value.preset === key)}
          >
            {label}
          </button>
        ))}
        <button
          type="button"
          aria-pressed={custom}
          onClick={() => onChange({ ...value, preset: 'custom' })}
          className={chipClass(custom)}
        >
          Custom
        </button>
        {custom && (
          <>
            <div>
              <label htmlFor={`${id}-from`} className={labelClass}>From</label>
              <input
                id={`${id}-from`}
                type="datetime-local"
                value={value.from}
                onChange={(e) => onChange({ ...value, from: e.target.value })}
                className={inputClass}
              />
            </div>
            <div>
              <label htmlFor={`${id}-to`} className={labelClass}>To</label>
              <input
                id={`${id}-to`}
                type="datetime-local"
                value={value.to}
                onChange={(e) => onChange({ ...value, to: e.target.value })}
                className={inputClass}
              />
            </div>
          </>
        )}
      </div>
    </fieldset>
  );
};

export default TimeRangeControl;
