import React from 'react';
import Surface from './Surface';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';

const MetricTile = ({
  title,
  value,
  unit = '',
  trend = null,
  icon: Icon = null,
  accent = 'none',
  subtitle = null,
  className = '',
}) => {
  const getTrendColor = () => {
    if (!trend) return '';
    if (trend.direction === 'neutral') return 'text-muted-foreground';
    return trend.isPositive ? 'text-emerald-400' : 'text-destructive';
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
        <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">{title}</span>
        {Icon && (
          <div className="p-2 rounded-md bg-muted text-primary border border-border">
            <Icon className="w-4 h-4" />
          </div>
        )}
      </div>

      <div className="flex items-baseline gap-2 my-1">
        <span className="text-2xl md:text-3xl font-black text-foreground tracking-tight font-mono">{value}</span>
        {unit && <span className="text-xs font-medium text-muted-foreground">{unit}</span>}
      </div>

      {(trend || subtitle) && (
        <div className="flex items-center justify-between text-xs mt-2 pt-2 border-t border-border">
          {trend ? (
            <div className={`flex items-center gap-1.5 font-medium ${getTrendColor()}`}>
              {TrendIcon && <TrendIcon className="w-3.5 h-3.5" />}
              <span>{trend.label}</span>
              <span className="sr-only">({trend.isPositive ? 'positive indicator' : 'attention indicator'})</span>
            </div>
          ) : (
            <span className="text-muted-foreground">{subtitle}</span>
          )}
        </div>
      )}
    </Surface>
  );
};

export default MetricTile;
