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
          className="block text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2"
        >
          {label} {required && <span className="text-destructive font-bold" aria-hidden="true">*</span>}
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
          className={`w-full bg-input/40 text-foreground placeholder:text-muted-foreground/60 rounded-md px-4 py-3 text-sm min-h-[44px] border transition-all duration-200 focus:outline-none disabled:opacity-50 disabled:cursor-not-allowed ${
            hasError
              ? 'border-destructive focus:border-destructive focus:ring-2 focus:ring-destructive/30'
              : 'border-border hover:border-stone-500 focus:border-primary focus:ring-2 focus:ring-primary/20'
          } ${unit ? 'pr-14' : ''}`}
          {...props}
        />
        {unit && (
          <span className="absolute right-3 text-xs font-medium text-muted-foreground pointer-events-none select-none px-2 py-1 rounded bg-muted border border-border">
            {unit}
          </span>
        )}
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

export default InputField;
