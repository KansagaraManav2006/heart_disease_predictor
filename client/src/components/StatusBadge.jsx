import React from 'react';
import { CheckCircle2, AlertTriangle, XCircle, Info, Activity } from 'lucide-react';

const StatusBadge = ({
  status = 'healthy',
  label,
  showIcon = true,
  size = 'md',
  className = '',
}) => {
  const isHealthy = status === 'healthy' || status === 'complete' || status === 'low_risk';
  const isAttention = status === 'attention' || status === 'warning' || status === 'medium_risk';
  const isDanger = status === 'high_risk' || status === 'error' || status === 'danger';
  const isProcessing = status === 'processing' || status === 'info';

  const styleClasses = isHealthy
    ? 'bg-teal-500/15 text-teal-300 border-teal-500/30'
    : isAttention
    ? 'bg-amber-500/15 text-amber-300 border-amber-500/30'
    : isDanger
    ? 'bg-coral-500/15 text-coral-300 border-coral-500/30'
    : isProcessing
    ? 'bg-cyan-500/15 text-cyan-300 border-cyan-500/30'
    : 'bg-violet-500/15 text-violet-300 border-violet-500/30';

  const iconMap = {
    healthy: CheckCircle2,
    complete: CheckCircle2,
    low_risk: CheckCircle2,
    attention: AlertTriangle,
    warning: AlertTriangle,
    medium_risk: AlertTriangle,
    high_risk: XCircle,
    error: XCircle,
    danger: XCircle,
    processing: Activity,
    info: Info,
    secondary: Info,
  };

  const IconComponent = iconMap[status] || Info;
  const sizeClass = size === 'sm' ? 'px-2 py-0.5 text-[11px] gap-1' : 'px-2.5 py-1 text-xs gap-1.5';

  return (
    <span
      className={`inline-flex items-center font-semibold rounded-full border ${styleClasses} ${sizeClass} ${className}`}
    >
      {showIcon && <IconComponent className={size === 'sm' ? 'w-3 h-3 flex-shrink-0' : 'w-3.5 h-3.5 flex-shrink-0'} />}
      <span>{label}</span>
    </span>
  );
};

export default StatusBadge;
