import React from 'react';

const SegmentedTabs = ({ tabs = [], activeTab, onChange, className = '', id = 'segmented-tabs' }) => {
  return (
    <div
      role="tablist"
      aria-label="Input mode selector"
      id={id}
      className={`inline-flex p-1 bg-muted/80 border border-border rounded-md gap-1 ${className}`}
    >
      {tabs.map((tab) => {
        const isActive = activeTab === tab.id;
        const Icon = tab.icon;

        return (
          <button
            key={tab.id}
            role="tab"
            aria-selected={isActive}
            aria-controls={`panel-${tab.id}`}
            id={`tab-${tab.id}`}
            onClick={() => onChange(tab.id)}
            className={`flex items-center gap-2 px-4 py-2 text-xs md:text-sm font-semibold rounded-md transition-all duration-200 focus-visible:ring-2 focus-visible:ring-ring ${
              isActive
                ? 'bg-card text-primary border border-primary/40 shadow-sm font-bold'
                : 'text-muted-foreground hover:text-foreground hover:bg-card/50 border border-transparent'
            }`}
          >
            {Icon && <Icon className={`w-4 h-4 ${isActive ? 'text-primary' : 'text-muted-foreground'}`} />}
            <span>{tab.label}</span>
            {tab.badge && (
              <span className="px-1.5 py-0.5 text-[10px] rounded-full bg-accent/20 text-amber-400 border border-amber-500/30 font-bold">
                {tab.badge}
              </span>
            )}
          </button>
        );
      })}
    </div>
  );
};

export default SegmentedTabs;
