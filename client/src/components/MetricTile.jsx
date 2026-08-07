import React from 'react';
import Surface from './Surface';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';

const MetricTile = ({
  title,
  value,
  unit = '',
  trend = null, // { direction: 'up' | 'down' | 'neutral', label: string, isPositive: boolean }
  icon: Icon = null,
  accent = 'none',
  subtitle = null,
  className = '',
}) => {
  const getTrendColor = () => {
    if (!trend) return '';
    if (trend.direction === 'neutral') return 'text-slate-400';
    return trend.isPositive ? 'text-teal-400' : 'text-coral-400';
  };

  const TrendIcon = trend
    ? trend.direction === 'up'
      ? TrendingUp
      : trend.direction === 'down'
      ? TrendingDown
      : Minus
    : null;

  return (
    <Surface variant="flat" accent={accent} className={`p-5 flex flex-col justify-between ${className}`}>
      <div className="flex items-center justify-between gap-2 mb-3">
        <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">{title}</span>
        {Icon && (
          <div className="p-2 rounded-xl bg-slate-800 text-teal-400 border border-slate-700/60">
            <Icon className="w-4 h-4" />
          </div>
        )}
      </div>

      <div className="flex items-baseline gap-2 my-1">
        <span className="text-2xl md:text-3xl font-black text-slate-100 tracking-tight font-mono">{value}</span>
        {unit && <span className="text-xs font-medium text-slate-400">{unit}</span>}
      </div>

      {(trend || subtitle) && (
        <div className="flex items-center justify-between text-xs mt-2 pt-2 border-t border-slate-800/80">
          {trend ? (
            <div className={`flex items-center gap-1.5 font-medium ${getTrendColor()}`}>
              {TrendIcon && <TrendIcon className="w-3.5 h-3.5" />}
              <span>{trend.label}</span>
              <span className="sr-only">({trend.isPositive ? 'positive indicator' : 'attention indicator'})</span>
            </div>
          ) : (
            <span className="text-slate-400">{subtitle}</span>
          )}
        </div>
      )}
    </Surface>
  );
};

export default MetricTile;
