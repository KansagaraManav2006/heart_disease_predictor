import React from 'react';

const SegmentedTabs = ({ tabs = [], activeTab, onChange, className = '', id = 'segmented-tabs' }) => {
  return (
    <div
      role="tablist"
      aria-label="Input mode selector"
      id={id}
      className={`inline-flex p-1 bg-slate-900/90 border border-slate-800 rounded-xl gap-1 ${className}`}
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
            className={`flex items-center gap-2 px-4 py-2.5 text-xs md:text-sm font-semibold rounded-xl transition-all duration-180 focus-visible:ring-2 focus-visible:ring-teal-400 ${
              isActive
                ? 'bg-slate-800 text-teal-400 border border-teal-500/40 shadow-sm font-bold'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-850/50 border border-transparent'
            }`}
          >
            {Icon && <Icon className={`w-4 h-4 ${isActive ? 'text-teal-400' : 'text-slate-400'}`} />}
            <span>{tab.label}</span>
            {tab.badge && (
              <span className="px-2 py-0.5 text-[10px] rounded-full bg-amber-500/20 text-amber-300 border border-amber-500/30 font-bold">
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
