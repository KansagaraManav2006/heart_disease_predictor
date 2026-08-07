import React from 'react';
import { ShieldCheck, AlertTriangle, ShieldAlert } from 'lucide-react';

const RiskBadge = ({ riskBand = 'LOW', score = null, size = 'md', className = '' }) => {
  const normalizedBand = (riskBand || 'LOW').toUpperCase();

  const isHigh = normalizedBand === 'HIGH' || normalizedBand === 'ELEVATED';
  const isModerate = normalizedBand === 'MODERATE' || normalizedBand === 'MEDIUM';

  const config = isHigh
    ? {
        bg: 'bg-coral-500/15 border-coral-500/30 text-coral-300',
        label: 'HIGH RISK',
        Icon: ShieldAlert,
      }
    : isModerate
    ? {
        bg: 'bg-amber-500/15 border-amber-500/30 text-amber-300',
        label: 'MODERATE RISK',
        Icon: AlertTriangle,
      }
    : {
        bg: 'bg-teal-500/15 border-teal-500/30 text-teal-300',
        label: 'LOW RISK',
        Icon: ShieldCheck,
      };

  const { Icon } = config;

  const sizeClasses = size === 'lg'
    ? 'px-4 py-2 text-base rounded-xl gap-2 font-bold'
    : size === 'sm'
    ? 'px-2 py-0.5 text-xs rounded-lg gap-1 font-semibold'
    : 'px-3 py-1.5 text-xs rounded-xl gap-1.5 font-semibold';

  return (
    <span className={`inline-flex items-center border ${config.bg} ${sizeClasses} ${className}`}>
      <Icon className={size === 'lg' ? 'w-5 h-5' : 'w-4 h-4'} />
      <span>{config.label}</span>
      {score !== null && <span className="opacity-80 font-mono">({score}%)</span>}
    </span>
  );
};

export default RiskBadge;
