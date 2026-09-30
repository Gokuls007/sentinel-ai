// Shared class strings for HUD form controls.

export const chipClass = (active) =>
  `px-2 py-1 text-[9px] mono uppercase font-bold border cursor-pointer transition-colors outline-none focus-visible:ring-1 focus-visible:ring-cyan-400 ${
    active ? 'border-cyan-400/60 text-cyan-300 bg-cyan-500/10' : 'border-white/10 text-white/40 hover:text-white/70'
  }`;

export const inputClass =
  'bg-black/60 border border-cyan-500/20 px-2 py-1 text-[10px] mono text-cyan-100 outline-none focus:border-cyan-400 [color-scheme:dark]';

export const labelClass = 'block text-[9px] uppercase font-bold text-white/40 tracking-widest small-caps mb-1';
