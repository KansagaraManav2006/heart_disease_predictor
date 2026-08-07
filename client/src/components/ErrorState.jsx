import React from 'react';
import Surface from './Surface';
import Button from './Button';
import { AlertTriangle, RefreshCw } from 'lucide-react';

const ErrorState = ({
  title = 'Service Unavailable',
  message = 'An unexpected error occurred while fetching clinical data. Please try again.',
  onRetry = null,
  retryLabel = 'Retry Request',
  className = '',
}) => {
  return (
    <Surface variant="flat" accent="coral" className={`p-6 bg-coral-950/20 text-left ${className}`}>
      <div className="flex items-start gap-4">
        <div className="p-2.5 rounded-xl bg-coral-500/20 text-coral-400 border border-coral-500/30 flex-shrink-0">
          <AlertTriangle className="w-5 h-5" />
        </div>
        <div className="flex-1">
          <h4 className="text-sm font-bold text-coral-300 mb-1">{title}</h4>
          <p className="text-xs text-slate-300 leading-relaxed mb-4">{message}</p>
          {onRetry && (
            <Button
              onClick={onRetry}
              variant="danger"
              size="sm"
              icon={RefreshCw}
              className="mt-1"
            >
              {retryLabel}
            </Button>
          )}
        </div>
      </div>
    </Surface>
  );
};

export default ErrorState;
