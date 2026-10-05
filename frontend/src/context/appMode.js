import { createContext, useContext } from 'react';

export const MODES = [
  { id: 'posture', label: 'Desk Posture Coach', short: 'Posture' },
  { id: 'warehouse', label: 'Warehouse Safety', short: 'Warehouse' },
  { id: 'home', label: 'Home Care', short: 'Home' },
  { id: 'exam', label: 'Exam Hall', short: 'Exam' },
];

export const AppModeContext = createContext({
  mode: 'warehouse',
  demoFootage: false,
  loaded: false,
  error: null,
  setMode: () => {},
  setDemoFootage: () => {},
});

/** { mode, demoFootage, loaded, error, setMode(id), setDemoFootage(bool) } */
export function useAppMode() {
  return useContext(AppModeContext);
}
