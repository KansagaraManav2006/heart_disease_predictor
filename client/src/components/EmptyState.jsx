import React from 'react';
import Surface from './Surface';
import Button from './Button';
import { Inbox } from 'lucide-react';

const EmptyState = ({
  icon: Icon = Inbox,
  title = 'No records found',
  description = 'There are no items matching your criteria at this time.',
  actionLabel = null,
  onAction = null,
  actionIcon = null,
  className = '',
}) => {
  return (
    <Surface variant="flat" className={`text-center py-12 px-6 flex flex-col items-center justify-center ${className}`}>
      <div className="w-14 h-14 rounded-2xl bg-slate-900 border border-slate-800 text-teal-400 flex items-center justify-center mb-4 shadow-inner">
        {Icon && <Icon className="w-7 h-7" />}
      </div>
      <h3 className="text-lg font-bold text-slate-100 mb-1">{title}</h3>
      <p className="text-sm text-slate-400 max-w-sm leading-relaxed mb-6">{description}</p>
      {actionLabel && onAction && (
        <Button onClick={onAction} variant="primary" size="md" icon={actionIcon}>
          {actionLabel}
        </Button>
      )}
    </Surface>
  );
};

export default EmptyState;
