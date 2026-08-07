import React from 'react';
import { Search, X } from 'lucide-react';

const SearchBar = ({
  value,
  onChange,
  onClear,
  placeholder = 'Search records...',
  className = '',
  id = 'search-input',
  ariaLabel = 'Search records',
}) => {
  return (
    <div className={`relative flex items-center w-full ${className}`}>
      <Search className="absolute left-3.5 w-4 h-4 text-slate-400 pointer-events-none" />
      <input
        type="text"
        id={id}
        value={value}
        onChange={onChange}
        placeholder={placeholder}
        aria-label={ariaLabel}
        className="w-full pl-10 pr-10 py-2.5 bg-slate-900/90 text-slate-100 placeholder-slate-500 rounded-xl text-sm border border-slate-700/80 hover:border-slate-600 focus:border-teal-400 focus:ring-2 focus:ring-teal-400/20 transition-all duration-200 outline-none"
      />
      {value && (
        <button
          type="button"
          onClick={onClear}
          aria-label="Clear search query"
          className="absolute right-3 p-1 text-slate-400 hover:text-slate-200 rounded-lg hover:bg-slate-800 transition-colors"
        >
          <X className="w-4 h-4" />
        </button>
      )}
    </div>
  );
};

export default SearchBar;
