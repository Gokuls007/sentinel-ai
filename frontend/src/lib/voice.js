import { useEffect, useRef, useState } from 'react';
import { useFeed } from '../context/liveFeed';
import { loadStored, saveStored } from './api';

import { COOLDOWN_MS, pickAnnouncements } from './voiceRules';

export { COOLDOWN_MS, SPOKEN_TYPES, spokenText, pickAnnouncements } from './voiceRules';
export const VOICE_KEY = 'sentinel.voice.muted';

export function useVoiceAlerts() {
  const { alerts } = useFeed();
  const [muted, setMuted] = useState(() => loadStored(VOICE_KEY, false));
  const sinceRef = useRef(null); // only alerts that arrive after the page opened
  const last = useRef({});
  const supported = typeof window !== 'undefined' && 'speechSynthesis' in window;

  useEffect(() => {
    sinceRef.current = Date.now();
  }, []);

  useEffect(() => {
    saveStored(VOICE_KEY, muted);
    if (muted && supported) window.speechSynthesis.cancel();
  }, [muted, supported]);

  useEffect(() => {
    if (!alerts.length || sinceRef.current === null) return;
    const now = Date.now();
    const texts = supported && !muted ? pickAnnouncements(alerts, sinceRef.current, last.current, now) : [];
    // Move past these alerts even while muted, so unmuting doesn't read out old ones.
    sinceRef.current = Math.max(sinceRef.current, (alerts[0].timestamp || 0) * 1000 + 1);
    for (const text of texts) {
      const u = new SpeechSynthesisUtterance(text);
      u.rate = 1.05;
      window.speechSynthesis.speak(u);
    }
  }, [alerts, muted, supported]);

  return { muted, setMuted, supported };
}
