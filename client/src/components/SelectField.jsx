import React, { useId } from 'react';
import { ChevronDown } from 'lucide-react';

const SelectField = ({
  label,
  name,
  value,
  onChange,
  options = [],
  required = false,
  helperText = null,
  error = null,
  disabled = false,
  placeholder = 'Select an option',
  className = '',
  id,
  ...props
}) => {
  const generatedId = useId();
  const selectId = id || name || generatedId;
  const helperId = helperText ? `${selectId}-helper` : undefined;
  const errorId = error ? `${selectId}-error` : undefined;
  const describedBy = [errorId, helperId].filter(Boolean).join(' ') || undefined;

  const hasError = Boolean(error);

  return (
    <div className={`mb-4 w-full text-left ${className}`}>
      {label && (
        <label
          htmlFor={selectId}
          className="block text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2"
        >
          {label} {required && <span className="text-destructive font-bold" aria-hidden="true">*</span>}
        </label>
      )}

      <div className="relative">
        <select
          id={selectId}
          name={name}
          value={value ?? ''}
          onChange={onChange}
          required={required}
          disabled={disabled}
          aria-invalid={hasError}
          aria-describedby={describedBy}
          className={`w-full appearance-none bg-input/40 text-foreground rounded-md px-4 py-3 pr-10 text-sm min-h-[44px] border transition-all duration-200 focus:outline-none cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed ${
            hasError
              ? 'border-destructive focus:border-destructive focus:ring-2 focus:ring-destructive/30'
              : 'border-border hover:border-stone-500 focus:border-primary focus:ring-2 focus:ring-primary/20'
          }`}
          {...props}
        >
          {placeholder && (
            <option value="" disabled className="bg-card text-muted-foreground">
              {placeholder}
            </option>
          )}
          {options.map((opt) => (
            <option
              key={typeof opt === 'object' ? opt.value : opt}
              value={typeof opt === 'object' ? opt.value : opt}
              className="bg-card text-foreground"
            >
              {typeof opt === 'object' ? opt.label : opt}
            </option>
          ))}
        </select>
        <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-3.5 text-muted-foreground">
          <ChevronDown className="w-4 h-4" />
        </div>
      </div>

      {error && (
        <p id={errorId} className="mt-1.5 text-xs text-destructive font-medium flex items-center gap-1">
          <span aria-hidden="true">⚠</span> {error}
        </p>
      )}

      {!error && helperText && (
        <p id={helperId} className="mt-1.5 text-xs text-muted-foreground">
          {helperText}
        </p>
      )}
    </div>
  );
};

export default SelectField;
