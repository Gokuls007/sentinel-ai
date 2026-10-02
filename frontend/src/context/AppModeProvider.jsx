import { useCallback, useEffect, useMemo, useState } from 'react';
import { fetchJson, loadStored, putJson, saveStored } from '../lib/api';
import { AppModeContext, MODES } from './appMode';

const STORE_KEY = 'sentinel.app';

/**
 * The app mode and the demo-footage switch. The backend is the source of truth (it decides
 * what the camera pipelines run); the last known value is cached in localStorage so the
 * right mode shows instantly on reload, before the backend answers.
 */
const AppModeProvider = ({ children }) => {
  const [state, setState] = useState(() => {
    const cached = loadStored(STORE_KEY, null);
    return {
      mode: MODES.some((m) => m.id === cached?.mode) ? cached.mode : 'warehouse',
      demoFootage: Boolean(cached?.demo_footage),
      loaded: false,
      error: null,
    };
  });

  const apply = useCallback((data) => {
    saveStored(STORE_KEY, data);
    setState({ mode: data.mode, demoFootage: Boolean(data.demo_footage), loaded: true, error: null });
  }, []);

  useEffect(() => {
    let cancelled = false;
    const load = () =>
      fetchJson('/api/app')
        .then((data) => !cancelled && apply(data))
        .catch((err) => {
          if (cancelled) return;
          setState((s) => ({ ...s, error: err }));
          setTimeout(load, 3000); // backend still starting
        });
    load();
    return () => { cancelled = true; };
  }, [apply]);

  const update = useCallback(async (patch) => {
    // Optimistic: switch the UI now, then confirm with the backend.
    setState((s) => ({
      ...s,
      ...(patch.mode ? { mode: patch.mode } : {}),
      ...('demo_footage' in patch ? { demoFootage: patch.demo_footage } : {}),
    }));
    try {
      apply(await putJson('/api/app', patch));
    } catch (err) {
      setState((s) => ({ ...s, error: err }));
    }
  }, [apply]);

  const value = useMemo(() => ({
    ...state,
    setMode: (mode) => update({ mode }),
    setDemoFootage: (on) => update({ demo_footage: on }),
  }), [state, update]);

  return <AppModeContext.Provider value={value}>{children}</AppModeContext.Provider>;
};

export default AppModeProvider;
