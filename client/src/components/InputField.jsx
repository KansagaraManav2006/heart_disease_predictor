import React from 'react';

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
    required = false
}) => {
    return (
        <div className="mb-4 w-full">
            {label && (
                <label htmlFor={name} className="block text-sm font-medium text-slate-700 mb-1.5 ml-1">
                    {label} {required && <span className="text-danger">*</span>}
                </label>
            )}
            <input
                type={type}
                id={name}
                name={name}
                value={value}
                onChange={onChange}
                placeholder={placeholder}
                min={min}
                max={max}
                step={step}
                required={required}
                className="glass-input transition-colors duration-200"
            />
        </div>
    );
};

export default InputField;
