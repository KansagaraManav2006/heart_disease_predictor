import React from 'react';
import Surface from './Surface';

const GlassCard = ({ children, className = '', ...props }) => {
  return (
    <Surface variant="glass" className={className} {...props}>
      {children}
    </Surface>
  );
};

export default GlassCard;
