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
      <Search className="absolute left-3.5 w-4 h-4 text-muted-foreground pointer-events-none" />
      <input
        type="text"
        id={id}
        value={value}
        onChange={onChange}
        placeholder={placeholder}
        aria-label={ariaLabel}
        className="w-full pl-10 pr-10 py-2.5 bg-input/40 text-foreground placeholder:text-muted-foreground/60 rounded-md text-sm border border-border hover:border-stone-500 focus:border-primary focus:ring-2 focus:ring-primary/20 transition-all duration-200 outline-none"
      />
      {value && (
        <button
          type="button"
          onClick={onClear}
          aria-label="Clear search query"
          className="absolute right-3 p-1 text-muted-foreground hover:text-foreground rounded hover:bg-muted transition-colors"
        >
          <X className="w-4 h-4" />
        </button>
      )}
    </div>
  );
};

export default SearchBar;
