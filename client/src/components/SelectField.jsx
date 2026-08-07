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
          className="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2"
        >
          {label} {required && <span className="text-coral-400 font-bold" aria-hidden="true">*</span>}
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
          className={`w-full appearance-none bg-slate-900/90 text-slate-100 placeholder-slate-500 rounded-xl px-4 py-3 pr-10 text-sm border transition-all duration-200 focus:outline-none cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed ${
            hasError
              ? 'border-coral-500/80 focus:border-coral-400 focus:ring-2 focus:ring-coral-500/30'
              : 'border-slate-700/80 hover:border-slate-600 focus:border-teal-400 focus:ring-2 focus:ring-teal-400/20'
          }`}
          {...props}
        >
          {placeholder && (
            <option value="" disabled className="bg-slate-900 text-slate-500">
              {placeholder}
            </option>
          )}
          {options.map((opt) => (
            <option
              key={typeof opt === 'object' ? opt.value : opt}
              value={typeof opt === 'object' ? opt.value : opt}
              className="bg-slate-900 text-slate-100"
            >
              {typeof opt === 'object' ? opt.label : opt}
            </option>
          ))}
        </select>
        <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-3.5 text-slate-400">
          <ChevronDown className="w-4 h-4" />
        </div>
      </div>

      {error && (
        <p id={errorId} className="mt-1.5 text-xs text-coral-400 font-medium flex items-center gap-1">
          <span aria-hidden="true">⚠</span> {error}
        </p>
      )}

      {!error && helperText && (
        <p id={helperId} className="mt-1.5 text-xs text-slate-400">
          {helperText}
        </p>
      )}
    </div>
  );
};

export default SelectField;
