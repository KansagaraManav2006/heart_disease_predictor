import React from 'react';

const GlassCard = ({ children, className = '', strong = false, style }) => {
    const baseClass = strong ? 'glass-strong' : 'glass';
    return (
        <div className={`${baseClass} p-6 md:p-8 ${className}`} style={style}>
            {children}
        </div>
    );
};

export default GlassCard;
