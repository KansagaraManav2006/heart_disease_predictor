import React, { useId } from 'react';

const InputField = ({
  label,
  type = 'text',
  name,
  value,
  onChange,
  placeholder,
  min,
  max,
  step,
  required = false,
  unit = null,
  helperText = null,
  error = null,
  disabled = false,
  className = '',
  id,
  ...props
}) => {
  const generatedId = useId();
  const inputId = id || name || generatedId;
  const helperId = helperText ? `${inputId}-helper` : undefined;
  const errorId = error ? `${inputId}-error` : undefined;
  const describedBy = [errorId, helperId].filter(Boolean).join(' ') || undefined;

  const hasError = Boolean(error);

  return (
    <div className={`mb-4 w-full text-left ${className}`}>
      {label && (
        <label
          htmlFor={inputId}
          className="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2"
        >
          {label} {required && <span className="text-coral-400 font-bold" aria-hidden="true">*</span>}
        </label>
      )}

      <div className="relative flex items-center">
        <input
          type={type}
          id={inputId}
          name={name}
          value={value ?? ''}
          onChange={onChange}
          placeholder={placeholder}
          min={min}
          max={max}
          step={step}
          required={required}
          disabled={disabled}
          aria-invalid={hasError}
          aria-describedby={describedBy}
          className={`w-full bg-slate-900/90 text-slate-100 placeholder-slate-500 rounded-xl px-4 py-3 text-sm min-h-[44px] border transition-all duration-200 focus:outline-none disabled:opacity-50 disabled:cursor-not-allowed ${
            hasError
              ? 'border-coral-500/80 focus:border-coral-400 focus:ring-2 focus:ring-coral-500/30'
              : 'border-slate-700/80 hover:border-slate-600 focus:border-teal-400 focus:ring-2 focus:ring-teal-400/20'
          } ${unit ? 'pr-14' : ''}`}
          {...props}
        />
        {unit && (
          <span className="absolute right-3 text-xs font-medium text-slate-400 pointer-events-none select-none px-2 py-1 rounded-lg bg-slate-800 border border-slate-700">
            {unit}
          </span>
        )}
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

export default InputField;
