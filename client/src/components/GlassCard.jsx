import React from 'react';
import Surface from './Surface';

const GlassCard = ({ children, className = '', strong = false, style, ...props }) => {
  return (
    <Surface
      variant={strong ? 'raised' : 'flat'}
      className={className}
      style={style}
      {...props}
    >
      {children}
    </Surface>
  );
};

export default GlassCard;
