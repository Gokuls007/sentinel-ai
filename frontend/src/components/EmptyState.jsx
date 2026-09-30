/** Dim centered HUD message ("Backend offline", "No events", ...). */
const EmptyState = ({ children, className = '' }) => (
  <div className={`flex flex-col items-center justify-center h-full min-h-[80px] opacity-40 ${className}`}>
    <p className="text-[10px] mono uppercase tracking-widest text-center">{children}</p>
  </div>
);

export default EmptyState;
