import React from 'react';

const GlassCard = ({ children, className = '', strong = false }) => {
    const baseClass = strong ? 'glass-strong' : 'glass';
    return (
        <div className={`${baseClass} p-6 md:p-8 ${className}`}>
            {children}
        </div>
    );
};

export default GlassCard;
