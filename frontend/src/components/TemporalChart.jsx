import React, { useState, useMemo } from 'react';
import { 
  Activity, 
  TrendingUp, 
  TrendingDown, 
  Info, 
  ArrowUpRight,
  ArrowDownRight
} from 'lucide-react';

export const PARAMETER_CATALOG = {
  xrsb_flux: {
    shortName: 'Longwave X-Ray',
    fullName: 'GOES Longwave X-Ray Flux (0.1 - 0.8 nm)',
    unit: 'W/m²',
    description: 'Primary baseline used internationally for M-class and X-class flare classification.',
    isPrimary: true,
  },
  xrsa_flux: {
    shortName: 'Shortwave X-Ray',
    fullName: 'GOES Shortwave X-Ray Flux (0.05 - 0.4 nm)',
    unit: 'W/m²',
    description: 'Harder spectral X-ray emissions accompanying violent magnetic reconnection.',
    isPrimary: true,
  },
  proton_gt10mep: {
    shortName: 'Proton Flux',
    fullName: 'Energetic Solar Protons (>10 MeV)',
    unit: 'pfu',
    description: 'Energetic particle flux driving Solar Radiation Storms and satellite anomalies.',
    isPrimary: true,
  },
  ratio_ab: {
    shortName: 'X-Ray Ratio',
    fullName: 'X-Ray Flux Ratio (XRSA / XRSB)',
    unit: 'ratio',
    description: 'Thermal spectral hardness ratio indicating active coronal heating.',
    isPrimary: true,
  },
  xrsb_max_6h: {
    shortName: '6h Max Flux',
    fullName: '6-Hour Peak Longwave X-Ray Flux',
    unit: 'W/m²',
    description: 'Highest intensity reached over the preceding 6 hours.',
    isPrimary: false,
  },
  xrsb_rate_1h: {
    shortName: '1h Rate of Change',
    fullName: '1-Hour Flux Temporal Derivative',
    unit: 'W/m²·h',
    description: 'Temporal rate of escalation marking rapid flare impulsive phases.',
    isPrimary: false,
  },
  xrsb_std_6h: {
    shortName: '6h Turbulence',
    fullName: '6-Hour Flux Standard Deviation',
    unit: 'W/m²',
    description: 'Variance in flux capturing active region magnetic volatility.',
    isPrimary: false,
  },
  xrsa_log: {
    shortName: 'XRSA Log',
    fullName: 'Log10 Shortwave X-Ray Flux',
    unit: 'log10(W/m²)',
    description: 'Base-10 logarithmic representation of shortwave solar flux.',
    isPrimary: false,
  },
  xrsb_log: {
    shortName: 'XRSB Log',
    fullName: 'Log10 Longwave X-Ray Flux',
    unit: 'log10(W/m²)',
    description: 'Base-10 logarithmic representation of longwave solar flux.',
    isPrimary: false,
  },
};

export default function TemporalChart({ telemetry, featureNames = [] }) {
  const [selectedFeature, setSelectedFeature] = useState('xrsb_flux');
  const [hoveredIndex, setHoveredIndex] = useState(null);

  const timestamps = telemetry?.timestamps || [];
  const rawValues = telemetry?.features?.[selectedFeature] || [];

  // Parameter metadata
  const paramInfo = PARAMETER_CATALOG[selectedFeature] || {
    shortName: selectedFeature,
    fullName: selectedFeature,
    unit: '',
    description: `Solar feature: ${selectedFeature}`,
  };

  // Quick select common chips
  const primaryFeatures = ['xrsb_flux', 'xrsa_flux', 'proton_gt10mep', 'ratio_ab'];

  // Filtered dropdown features
  const availableFeatures = featureNames.length > 0 ? featureNames : Object.keys(PARAMETER_CATALOG);

  // Calculate stats for current 24h window
  const stats = useMemo(() => {
    if (!rawValues.length) return { min: 0, max: 0, current: 0, mean: 0, maxIndex: 0 };
    const valid = rawValues.map((v) => (isNaN(v) ? 0 : v));
    let min = valid[0];
    let max = valid[0];
    let maxIndex = 0;
    let sum = 0;

    valid.forEach((v, idx) => {
      if (v < min) min = v;
      if (v > max) {
        max = v;
        maxIndex = idx;
      }
      sum += v;
    });

    const current = valid[valid.length - 1];
    const mean = sum / valid.length;
    return { min, max, current, mean, maxIndex };
  }, [rawValues]);

  // Compute delta vs average
  const deltaVsAvg = useMemo(() => {
    if (stats.mean === 0) return 0;
    return ((stats.current - stats.mean) / Math.abs(stats.mean)) * 100;
  }, [stats]);

  // Chart coordinates calculation
  const chart = useMemo(() => {
    const width = 850;
    const height = 340;
    const padding = { top: 30, right: 35, bottom: 45, left: 75 };

    const innerWidth = width - padding.left - padding.right;
    const innerHeight = height - padding.top - padding.bottom;

    if (!rawValues.length) {
      return { width, height, padding, innerWidth, innerHeight, points: '', dataPoints: [] };
    }

    const minVal = stats.min === stats.max ? stats.min - 1 : stats.min;
    const maxVal = stats.min === stats.max ? stats.max + 1 : stats.max;
    const valRange = maxVal - minVal;

    const xLabels = ['-23h', '-18h', '-12h', '-6h', 'Now'];
    const xLabelIndices = [0, 5, 11, 17, rawValues.length - 1];

    const dataPoints = rawValues.map((val, idx) => {
      const x = padding.left + (idx / Math.max(1, rawValues.length - 1)) * innerWidth;
      const normalizedY = (val - minVal) / valRange;
      const y = padding.top + (1 - normalizedY) * innerHeight;
      return {
        x,
        y,
        val,
        timestamp: timestamps[idx] || `Step ${idx}`,
        offsetLabel: idx === rawValues.length - 1 ? 'Now (T0)' : `T-${rawValues.length - 1 - idx}h`,
      };
    });

    const points = dataPoints.map((p) => `${p.x},${p.y}`).join(' ');

    return {
      width,
      height,
      padding,
      innerWidth,
      innerHeight,
      minVal,
      maxVal,
      dataPoints,
      points,
      xLabels,
      xLabelIndices,
    };
  }, [rawValues, timestamps, stats]);

  return (
    <div className="glass-panel rounded-xl p-4 lg:p-5 flex flex-col justify-between space-y-3.5 shadow-lg h-full">
      {/* =========================================================================
          HEADER + PARAMETER PICKER (CHIPS + DROPDOWN)
      ========================================================================= */}
      <div className="space-y-2.5 border-b border-slate-800 pb-3">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2.5">
          <div className="flex items-center space-x-2">
            <div className="w-7 h-7 rounded-lg bg-slate-800/80 border border-slate-700/80 flex items-center justify-center text-teal-400">
              <Activity className="w-4 h-4" />
            </div>
            <div>
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-100">
                24-Hour Telemetry Progression
              </h3>
              <p className="text-[11px] text-slate-400 font-mono">
                Continuous 24-step progression up to observation time (T0)
              </p>
            </div>
          </div>

          {/* Quick-Select Chips + Dropdown */}
          <div className="flex items-center space-x-2">
            <div className="flex items-center gap-1">
              {primaryFeatures.map((featKey) => {
                const info = PARAMETER_CATALOG[featKey];
                const isCurrent = selectedFeature === featKey;
                return (
                  <button
                    key={featKey}
                    type="button"
                    onClick={() => setSelectedFeature(featKey)}
                    className={`px-2.5 py-1 rounded-md text-[11px] font-medium transition-all cursor-pointer border ${
                      isCurrent
                        ? 'bg-teal-950/70 text-teal-300 border-teal-500 shadow-xs'
                        : 'bg-slate-900/60 text-slate-400 border-slate-800 hover:text-slate-200'
                    }`}
                  >
                    {info?.shortName || featKey}
                  </button>
                );
              })}
            </div>

            <select
              value={selectedFeature}
              onChange={(e) => setSelectedFeature(e.target.value)}
              className="bg-slate-900 border border-slate-700 text-slate-200 text-[11px] rounded-md px-2 py-1 focus:outline-none focus:border-teal-500 font-sans cursor-pointer shadow-inner"
            >
              {availableFeatures.map((feat) => {
                const info = PARAMETER_CATALOG[feat];
                return (
                  <option key={feat} value={feat}>
                    {info ? info.shortName : feat}
                  </option>
                );
              })}
            </select>
          </div>
        </div>

        {/* Friendly Parameter Description */}
        <div className="px-3 py-1.5 rounded-lg bg-slate-950/60 border border-slate-800/80 text-[11px] text-slate-300 flex items-center justify-between gap-2">
          <div className="flex items-center space-x-2">
            <Info className="w-3.5 h-3.5 text-teal-400 shrink-0" />
            <span className="font-semibold text-slate-200">{paramInfo.fullName}</span>
          </div>
          <span className="text-slate-400 font-mono text-[10px] hidden md:inline truncate max-w-sm">
            {paramInfo.description}
          </span>
        </div>
      </div>

      {/* =========================================================================
          COMPACT 4-STAT KPI ROW (Current, Min, Max, Average with Delta)
      ========================================================================= */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
        {/* 1. Current */}
        <div className="p-2.5 rounded-lg bg-slate-950/70 border border-slate-800/80 shadow-xs space-y-0.5">
          <div className="flex items-center justify-between text-[10px] font-mono text-slate-400 uppercase">
            <span>Current (T0)</span>
            {deltaVsAvg >= 0 ? (
              <span className="flex items-center text-teal-400 text-[9px]">
                <ArrowUpRight className="w-2.5 h-2.5" />
                <span>+{deltaVsAvg.toFixed(1)}%</span>
              </span>
            ) : (
              <span className="flex items-center text-rose-400 text-[9px]">
                <ArrowDownRight className="w-2.5 h-2.5" />
                <span>{deltaVsAvg.toFixed(1)}%</span>
              </span>
            )}
          </div>
          <div className="text-base font-bold font-mono-numbers text-slate-100">
            {stats.current.toExponential(3)}
          </div>
          <div className="text-[9px] text-slate-500 font-mono">{paramInfo.unit || 'units'}</div>
        </div>

        {/* 2. Min */}
        <div className="p-2.5 rounded-lg bg-slate-950/70 border border-slate-800/80 shadow-xs space-y-0.5">
          <div className="text-[10px] font-mono text-slate-400 uppercase">24h Min</div>
          <div className="text-base font-bold font-mono-numbers text-slate-300">
            {stats.min.toExponential(3)}
          </div>
          <div className="text-[9px] text-slate-500 font-mono">{paramInfo.unit || 'units'}</div>
        </div>

        {/* 3. Max (Peak) */}
        <div className="p-2.5 rounded-lg bg-slate-950/70 border border-slate-800/80 shadow-xs space-y-0.5">
          <div className="text-[10px] font-mono text-slate-400 uppercase">24h Peak</div>
          <div className="text-base font-bold font-mono-numbers text-amber-400">
            {stats.max.toExponential(3)}
          </div>
          <div className="text-[9px] text-slate-500 font-mono">at T-{rawValues.length - 1 - stats.maxIndex}h</div>
        </div>

        {/* 4. Average */}
        <div className="p-2.5 rounded-lg bg-slate-950/70 border border-slate-800/80 shadow-xs space-y-0.5">
          <div className="text-[10px] font-mono text-slate-400 uppercase">24h Average</div>
          <div className="text-base font-bold font-mono-numbers text-slate-200">
            {stats.mean.toExponential(3)}
          </div>
          <div className="text-[9px] text-slate-500 font-mono">Mean baseline</div>
        </div>
      </div>

      {/* =========================================================================
          ENLARGED HIGH-FIDELITY CHART (RESPONSIVE SVG FITTING LANDSCAPE)
      ========================================================================= */}
      <div className="relative w-full flex-1 min-h-[260px] max-h-[440px] rounded-xl bg-slate-950/90 border border-slate-800/80 overflow-hidden shadow-inner p-2 flex flex-col justify-center">
        {chart.dataPoints.length > 0 ? (
          <div className="w-full h-full flex items-center">
            <svg
              viewBox={`0 0 ${chart.width} ${chart.height}`}
              className="w-full h-full max-h-[360px] select-none"
              onMouseLeave={() => setHoveredIndex(null)}
            >
              {/* Shaded Min-Max Fluctuation Band */}
              <defs>
                <linearGradient id="chartGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#10b981" stopOpacity="0.22" />
                  <stop offset="100%" stopColor="#10b981" stopOpacity="0.0" />
                </linearGradient>
              </defs>

              {/* Horizontal Grid lines with 12px font */}
              {[0, 0.25, 0.5, 0.75, 1.0].map((frac, i) => {
                const y = chart.padding.top + frac * chart.innerHeight;
                const val = chart.maxVal - frac * (chart.maxVal - chart.minVal);
                return (
                  <g key={i}>
                    <line
                      x1={chart.padding.left}
                      y1={y}
                      x2={chart.width - chart.padding.right}
                      y2={y}
                      stroke="#1e293b"
                      strokeDasharray="4 4"
                      strokeWidth="1"
                    />
                    <text
                      x={chart.padding.left - 10}
                      y={y + 4}
                      textAnchor="end"
                      fill="#94a3b8"
                      fontSize="11"
                      fontWeight="500"
                      fontFamily="JetBrains Mono, monospace"
                    >
                      {val.toExponential(1)}
                    </text>
                  </g>
                );
              })}

              {/* X-Axis "-23h ... Now" Labels */}
              {chart.xLabelIndices.map((idx, i) => {
                const pt = chart.dataPoints[idx];
                if (!pt) return null;
                const label = chart.xLabels[i];
                return (
                  <g key={i}>
                    <line
                      x1={pt.x}
                      y1={chart.padding.top}
                      x2={pt.x}
                      y2={chart.height - chart.padding.bottom}
                      stroke="#1e293b"
                      strokeWidth="1"
                    />
                    <text
                      x={pt.x}
                      y={chart.height - chart.padding.bottom + 22}
                      textAnchor="middle"
                      fill="#cbd5e1"
                      fontSize="11"
                      fontWeight="600"
                      fontFamily="Inter, sans-serif"
                    >
                      {label}
                    </text>
                  </g>
                );
              })}

              {/* Shaded Area Fill */}
              {chart.dataPoints.length > 1 && (
                <polygon
                  points={`${chart.dataPoints[0].x},${chart.height - chart.padding.bottom} ${chart.points} ${chart.dataPoints[chart.dataPoints.length - 1].x},${chart.height - chart.padding.bottom}`}
                  fill="url(#chartGradient)"
                />
              )}

              {/* Main Line Trace */}
              <polyline
                fill="none"
                stroke="#10b981"
                strokeWidth="2.5"
                strokeLinecap="round"
                strokeLinejoin="round"
                points={chart.points}
              />

              {/* Peak Marker (Highest Value in 24h Window) */}
              {chart.dataPoints[stats.maxIndex] && (
                <g>
                  <circle
                    cx={chart.dataPoints[stats.maxIndex].x}
                    cy={chart.dataPoints[stats.maxIndex].y}
                    r="5"
                    fill="#f59e0b"
                    stroke="#070b12"
                    strokeWidth="2"
                  />
                  <rect
                    x={Math.max(chart.padding.left, Math.min(chart.width - chart.padding.right - 70, chart.dataPoints[stats.maxIndex].x - 35))}
                    y={chart.dataPoints[stats.maxIndex].y - 24}
                    width="70"
                    height="18"
                    rx="3"
                    fill="#0f172a"
                    stroke="#f59e0b"
                    strokeWidth="1"
                  />
                  <text
                    x={Math.max(chart.padding.left + 35, Math.min(chart.width - chart.padding.right - 35, chart.dataPoints[stats.maxIndex].x))}
                    y={chart.dataPoints[stats.maxIndex].y - 12}
                    textAnchor="middle"
                    fill="#f59e0b"
                    fontSize="9"
                    fontWeight="700"
                    fontFamily="Inter, sans-serif"
                  >
                    PEAK: {stats.max.toExponential(1)}
                  </text>
                </g>
              )}

              {/* Current Value Marker (Now) */}
              {chart.dataPoints[chart.dataPoints.length - 1] && (
                <g>
                  <circle
                    cx={chart.dataPoints[chart.dataPoints.length - 1].x}
                    cy={chart.dataPoints[chart.dataPoints.length - 1].y}
                    r="7"
                    fill="#10b981"
                    fillOpacity="0.3"
                  />
                  <circle
                    cx={chart.dataPoints[chart.dataPoints.length - 1].x}
                    cy={chart.dataPoints[chart.dataPoints.length - 1].y}
                    r="4"
                    fill="#34d399"
                    stroke="#070b12"
                    strokeWidth="1.5"
                  />
                </g>
              )}

              {/* Hover Hitboxes */}
              {chart.dataPoints.map((pt, idx) => (
                <circle
                  key={idx}
                  cx={pt.x}
                  cy={pt.y}
                  r="7"
                  fill="transparent"
                  className="cursor-pointer"
                  onMouseEnter={() => setHoveredIndex(idx)}
                />
              ))}

              {/* Hover Crosshairs Line */}
              {hoveredIndex !== null && chart.dataPoints[hoveredIndex] && (
                <g>
                  <line
                    x1={chart.dataPoints[hoveredIndex].x}
                    y1={chart.padding.top}
                    x2={chart.dataPoints[hoveredIndex].x}
                    y2={chart.height - chart.padding.bottom}
                    stroke="#cbd5e1"
                    strokeWidth="1.5"
                    strokeDasharray="3 3"
                  />
                  <circle
                    cx={chart.dataPoints[hoveredIndex].x}
                    cy={chart.dataPoints[hoveredIndex].y}
                    r="5.5"
                    fill="#38bdf8"
                    stroke="#070b12"
                    strokeWidth="2"
                  />
                </g>
              )}
            </svg>
          </div>
        ) : (
          <div className="h-full flex items-center justify-center text-xs font-mono text-slate-500">
            Awaiting telemetry data sequence...
          </div>
        )}

        {/* Floating Tooltip Box */}
        {hoveredIndex !== null && chart.dataPoints[hoveredIndex] && (
          <div className="absolute top-4 right-4 glass-panel border border-slate-700/80 p-3 rounded-lg text-xs font-mono shadow-xl backdrop-blur-md z-20 space-y-1 animate-fadeIn">
            <div className="text-slate-400 font-semibold">
              Time: <span className="text-slate-100">{chart.dataPoints[hoveredIndex].offsetLabel}</span>
            </div>
            <div className="text-slate-400 text-[11px]">
              UTC: <span className="text-slate-200">{chart.dataPoints[hoveredIndex].timestamp}</span>
            </div>
            <div className="text-teal-400 font-bold text-xs pt-1 border-t border-slate-800">
              Value: {chart.dataPoints[hoveredIndex].val.toExponential(4)} {paramInfo.unit}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
