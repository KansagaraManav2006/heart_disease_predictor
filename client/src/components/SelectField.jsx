import React from 'react';

const SelectField = ({
    label,
    name,
    value,
    onChange,
    options = [],
    required = false
}) => {
    return (
        <div className="mb-4 w-full">
            {label && (
                <label htmlFor={name} className="block text-sm font-medium text-slate-300 mb-1.5 ml-1">
                    {label} {required && <span className="text-danger">*</span>}
                </label>
            )}
            <div className="relative">
                <select
                    id={name}
                    name={name}
                    value={value}
                    onChange={onChange}
                    required={required}
                    className="glass-input appearance-none w-full cursor-pointer pr-10"
                >
                    <option value="" disabled className="text-slate-500">Select an option</option>
                    {options.map((opt) => (
                        <option key={opt.value} value={opt.value} className="bg-slate-800 text-white">
                            {opt.label}
                        </option>
                    ))}
                </select>
                <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-4 text-slate-400">
                    <svg className="fill-current h-4 w-4" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20">
                        <path d="M9.293 12.95l.707.707L15.657 8l-1.414-1.414L10 10.828 5.757 6.586 4.343 8z" />
                    </svg>
                </div>
            </div>
        </div>
    );
};

export default SelectField;
