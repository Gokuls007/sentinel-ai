import { Component } from 'react';

/**
 * Keeps the sidebar and live connection alive when a page fails to render,
 * most often because its lazy chunk could not be fetched (backend offline or
 * restarted with a new build). Reset by remounting (Layout keys it by path).
 */
export default class PageErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error) {
    console.error('[page]', error);
  }

  render() {
    if (!this.state.error) return this.props.children;
    const chunk = /dynamically imported module|Failed to fetch|Importing a module script failed/i.test(
      String(this.state.error?.message || ''),
    );
    return (
      <div className="p-6 space-y-3">
        <p className="text-[11px] mono uppercase tracking-widest text-red-400">
          {chunk ? 'Could not load this page: backend offline or starting' : 'This page failed to render'}
        </p>
        <button
          type="button"
          onClick={() => window.location.reload()}
          className="px-2 py-1 text-[9px] mono uppercase font-bold border border-cyan-400/50 text-cyan-300 cursor-pointer outline-none focus-visible:ring-1 focus-visible:ring-cyan-400"
        >
          Reload
        </button>
      </div>
    );
  }
}
