import React from 'react';

const DashboardPanel = ({ children, title, headerAction, className = "", severity = "cyan" }) => {
  const accentColor = severity === "red" ? "border-red-500/30" : "border-cyan-500/30";
  const titleColor = severity === "red" ? "text-red-500" : "text-cyan-400";

  return (
    <div className={`hud-panel-base ${accentColor} ${className}`}>
      {/* Nested Bracket Wrapper for 4-corners */}
      <div className="brackets-primary h-full w-full">
        <div className="brackets-secondary h-full w-full p-4 flex flex-col">
          
          {/* Header */}
          {title && (
            <div className="flex items-center justify-between mb-4 pb-2 border-b border-white/5">
              <h3 className={`text-[11px] font-bold uppercase tracking-[0.2em] ${titleColor} small-caps flex items-center`}>
                <span className={`w-1 h-1 rounded-full mr-2 ${severity === 'red' ? 'bg-red-500 animate-pulse' : 'bg-cyan-400'}`} />
                {title}
              </h3>
              {headerAction && (
                <div className="text-[10px] mono text-white/40 uppercase">
                  {headerAction}
                </div>
              )}
            </div>
          )}

          {/* Content */}
          <div className="flex-1 min-h-0">
            {children}
          </div>

        </div>
      </div>
    </div>
  );
};

export default DashboardPanel;
