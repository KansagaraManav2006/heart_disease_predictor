import React from 'react';

const Button = ({
    children,
    onClick,
    type = 'button',
    variant = 'primary',
    className = '',
    disabled = false,
    fullWidth = true
}) => {
    const baseClasses = variant === 'primary' ? 'btn-primary' : 'btn-secondary';
    const widthClass = fullWidth ? 'w-full' : '';

    return (
        <button
            type={type}
            onClick={onClick}
            disabled={disabled}
            className={`${baseClasses} ${widthClass} ${className} flex justify-center items-center`}
        >
            {children}
        </button>
    );
};

export default Button;
