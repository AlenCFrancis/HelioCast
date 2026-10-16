import React, { useMemo, useRef, useState, useLayoutEffect } from 'react';
import { ArrowUpRight, ArrowDownRight, Minus } from 'lucide-react';
import { PARAMETER_CATALOG } from './TemporalChart';

/* ---------------------------------------------------------------------------
   One stable color per telemetry parameter so the eye can track a feature
   between its bar chart and the anomaly summary.
--------------------------------------------------------------------------- */
const FEATURE_COLORS = {
  xrsa_flux: '#38bdf8',
  xrsb_flux: '#2dd4bf',
  proton_gt10mep: '#a78bfa',
  ratio_ab: '#f472b6',
  xrsa_log: '#60a5fa',
  xrsb_log: '#34d399',
  xrsb_rate_1h: '#fbbf24',
  xrsb_rate_3h: '#fb923c',
  xrsb_max_6h: '#f87171',
  xrsb_std_6h: '#c084fc',
  xrsa_max_6h: '#22d3ee',
};
const FALLBACK_COLORS = ['#94a3b8', '#f472b6', '#34d399', '#fb923c', '#60a5fa'];
const PEAK_COLOR = '#f59e0b';
const MONO = 'JetBrains Mono, monospace';

const colorOf = (key, i) => FEATURE_COLORS[key] || FALLBACK_COLORS[i % FALLBACK_COLORS.length];
const infoOf = (key) =>
  PARAMETER_CATALOG[key] || { shortName: key, fullName: key, unit: '', description: `Solar feature: ${key}` };

export function fmtVal(v) {
  if (typeof v !== 'number' || !Number.isFinite(v)) return '—';
  const a = Math.abs(v);
  if (a === 0) return '0';
  if (a < 1e-2 || a >= 1e4) return v.toExponential(2);
  if (a < 1) return v.toFixed(3);
  return v.toFixed(2);
}

function fmtTick(v) {
  if (!Number.isFinite(v)) return '';
  const a = Math.abs(v);
  if (a === 0) return '0';
  if (a < 1e-2 || a >= 1e4) return v.toExponential(1);
  if (a < 10) return v.toFixed(2);
  return v.toFixed(0);
}

const fmtSigned = (v) => (v >= 0 ? '+' : '') + fmtVal(v);
const offsetLabel = (n, i) => (i === n - 1 ? 'T0 (now)' : `T-${n - 1 - i}h`);

/** Tracks the content box of an element so SVGs can fill whatever space the grid gives them. */
function useBox() {
  const ref = useRef(null);
  const [box, setBox] = useState({ w: 0, h: 0 });
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return undefined;
    const update = (r) => setBox({ w: Math.floor(r.width), h: Math.floor(r.height) });
    update(el.getBoundingClientRect());
    const ro = new ResizeObserver(([entry]) => update(entry.contentRect));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, box];
}

function computeStats(values) {
  const vals = (values || []).map((v) => (typeof v === 'number' && Number.isFinite(v) ? v : 0));
  const n = vals.length;
  if (!n) return null;
  let min = vals[0];
  let max = vals[0];
  let minIdx = 0;
  let maxIdx = 0;
  let sum = 0;
  vals.forEach((v, i) => {
    if (v < min) { min = v; minIdx = i; }
    if (v > max) { max = v; maxIdx = i; }
    sum += v;
  });
  const mean = sum / n;
  const std = Math.sqrt(vals.reduce((acc, v) => acc + (v - mean) ** 2, 0) / n);
  const current = vals[n - 1];
  const z = std > 0 ? (current - mean) / std : 0;
  return { vals, n, min, max, minIdx, maxIdx, mean, std, current, z };
}

function Tooltip({ tip }) {
  if (!tip) return null;
  const flip = tip.x > window.innerWidth - 240;
  const lift = tip.y > window.innerHeight - 140;
  return (
    <div
      className="fixed z-50 pointer-events-none rounded-lg border border-slate-700/80 bg-slate-950/95 px-3 py-2 shadow-2xl backdrop-blur-md"
      style={{
        left: flip ? tip.x - 14 : tip.x + 14,
        top: lift ? tip.y - 14 : tip.y + 14,
        transform: `${flip ? 'translateX(-100%)' : ''} ${lift ? 'translateY(-100%)' : ''}`,
      }}
    >
      <div className="flex items-center gap-2 text-[11px] font-semibold text-slate-100">
        <span className="h-2 w-2 rounded-sm" style={{ background: tip.color }} />
        {tip.title}
      </div>
      <div className="mt-1.5 space-y-0.5">
        {tip.rows.map(([k, v]) => (
          <div key={k} className="flex items-center justify-between gap-6 font-mono-numbers text-[11px]">
            <span className="text-slate-400">{k}</span>
            <span className="text-slate-100">{v}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function TrendChip({ z }) {
  if (Math.abs(z) < 0.25) {
    return (
      <span className="flex items-center justify-end gap-0.5 text-[10px] font-mono-numbers text-slate-500">
        <Minus className="h-2.5 w-2.5" /> {z.toFixed(1)}σ
      </span>
    );
  }
  const up = z > 0;
  const Icon = up ? ArrowUpRight : ArrowDownRight;
  return (
    <span className={`flex items-center justify-end gap-0.5 text-[10px] font-mono-numbers ${up ? 'text-amber-300' : 'text-teal-300'}`}>
      <Icon className="h-2.5 w-2.5" /> {up ? '+' : ''}{z.toFixed(1)}σ
    </span>
  );
}

/* ---------------------------------------------------------------------------
   24 hourly bars for a single parameter. Height adapts to the grid cell.
--------------------------------------------------------------------------- */
function FeatureBarCard({ featKey, stats, timestamps, color, index, setTip }) {
  const info = infoOf(featKey);
  const [ref, { w, h }] = useBox();
  const [hover, setHover] = useState(null);
  const gid = `fbar-${featKey}`;

  const geo = useMemo(() => {
    if (!stats || w <= 0 || h <= 0) return null;
    const m = { t: 10, r: 4, b: 16, l: 46 };
    const pw = Math.max(0, w - m.l - m.r);
    const ph = Math.max(0, h - m.t - m.b);
    const { min, max } = stats;
    const range = max - min || Math.abs(max) * 0.1 || 1;

    // Signed series (rates of change) diverge from zero; one-sided series use a
    // padded floor so small hour-to-hour movements remain visible.
    let lo;
    let hi;
    let base;
    if (min < 0 && max > 0) {
      lo = min - range * 0.08;
      hi = max + range * 0.08;
      base = 0;
    } else if (min >= 0) {
      lo = Math.max(0, min - range * 0.35);
      hi = max + range * 0.12;
      base = lo;
    } else {
      lo = min - range * 0.35;
      hi = max + range * 0.12;
      base = lo;
    }
    const y = (v) => m.t + ph * (1 - (v - lo) / (hi - lo));
    const band = pw / stats.n;
    const bw = Math.max(1.5, Math.min(band * 0.7, 28));
    const ticks = base === 0 && lo < 0 ? [max, 0, min] : [max, (max + min) / 2, min];
    const n = stats.n;
    const xIdx = [...new Set([0, Math.round((n - 1) / 2), Math.round((3 * (n - 1)) / 4), n - 1])];
    return { m, pw, ph, base, y, band, bw, ticks, xIdx };
  }, [stats, w, h]);

  const peakOffset = stats ? stats.n - 1 - stats.maxIdx : 0;

  return (
    <section className="group flex min-h-0 flex-col rounded-xl border border-slate-800/80 bg-slate-900/50 p-3 transition-colors hover:border-slate-700">
      <header className="mb-1 flex items-start justify-between gap-2">
        <div className="min-w-0">
          <h3 className="flex items-center gap-1.5 truncate text-[12.5px] font-semibold tracking-tight text-slate-100" title={info.fullName}>
            <span className="h-2 w-2 shrink-0 rounded-sm" style={{ background: color }} />
            <span className="truncate">{info.shortName}</span>
          </h3>
          <p className="mt-0.5 truncate font-mono-numbers text-[10px] text-slate-500" title={info.description}>
            {info.unit || 'value'} · peak {stats ? fmtVal(stats.max) : '—'} @ {peakOffset === 0 ? 'T0' : `T-${peakOffset}h`}
          </p>
        </div>
        <div className="shrink-0 text-right">
          <div className="font-mono-numbers text-[13px] font-semibold leading-tight text-slate-50">
            {stats ? fmtVal(stats.current) : '—'}
          </div>
          {stats && <TrendChip z={stats.z} />}
        </div>
      </header>

      <div ref={ref} className="relative min-h-[80px] flex-1">
        {geo && stats && (
          <svg
            className="absolute inset-0 select-none"
            width={w}
            height={h}
            role="img"
            aria-label={`${info.fullName}, last 24 hours`}
            onMouseLeave={() => { setHover(null); setTip(null); }}
          >
            <defs>
              <linearGradient id={gid} x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={color} stopOpacity="0.95" />
                <stop offset="100%" stopColor={color} stopOpacity="0.3" />
              </linearGradient>
            </defs>

            {geo.ticks.map((t, i) => (
              <g key={i}>
                <line
                  x1={geo.m.l} x2={geo.m.l + geo.pw} y1={geo.y(t)} y2={geo.y(t)}
                  stroke={t === 0 && geo.base === 0 ? '#475569' : '#1e293b'}
                  strokeDasharray={t === 0 && geo.base === 0 ? '' : '3 3'}
                />
                <text x={geo.m.l - 6} y={geo.y(t) + 3} textAnchor="end" fontSize="9" fontFamily={MONO} className="fill-slate-500">
                  {fmtTick(t)}
                </text>
              </g>
            ))}

            {geo.xIdx.map((i) => (
              <text
                key={i}
                x={geo.m.l + geo.band * i + geo.band / 2}
                y={geo.m.t + geo.ph + 12}
                textAnchor={i === 0 ? 'start' : i === stats.n - 1 ? 'end' : 'middle'}
                fontSize="9"
                fontFamily={MONO}
                className={i === stats.n - 1 ? 'fill-slate-300' : 'fill-slate-500'}
              >
                {i === stats.n - 1 ? 'Now' : `-${stats.n - 1 - i}h`}
              </text>
            ))}

            {hover !== null && (
              <line
                x1={geo.m.l + geo.band * hover + geo.band / 2}
                x2={geo.m.l + geo.band * hover + geo.band / 2}
                y1={geo.m.t}
                y2={geo.m.t + geo.ph}
                stroke="#cbd5e1"
                strokeOpacity="0.35"
                strokeDasharray="2 3"
                pointerEvents="none"
              />
            )}

            {stats.vals.map((v, i) => {
              const cx = geo.m.l + geo.band * i + geo.band / 2;
              const yv = geo.y(v);
              const yb = geo.y(geo.base);
              const top = Math.min(yv, yb);
              const bh = Math.max(1, Math.abs(yb - yv));
              const isNow = i === stats.n - 1;
              const neg = v < geo.base;
              const dim = hover !== null && hover !== i;
              return (
                <rect
                  key={i}
                  x={cx - geo.bw / 2}
                  y={top}
                  width={geo.bw}
                  height={bh}
                  rx={Math.min(2, geo.bw / 2)}
                  fill={isNow ? color : `url(#${gid})`}
                  fillOpacity={dim ? 0.28 : 1}
                  stroke={isNow ? '#f8fafc' : 'none'}
                  strokeOpacity="0.75"
                  strokeWidth={isNow ? 1 : 0}
                  className="bar-grow"
                  pointerEvents="none"
                  style={{
                    animationDelay: `${index * 35 + i * 12}ms`,
                    transformOrigin: neg ? '50% 0%' : '50% 100%',
                    transition: 'y 350ms ease, height 350ms ease, fill-opacity 150ms',
                  }}
                />
              );
            })}

            {stats.max !== stats.min && (
              <circle
                cx={geo.m.l + geo.band * stats.maxIdx + geo.band / 2}
                cy={geo.y(stats.max) - 5}
                r="2.6"
                fill={PEAK_COLOR}
                stroke="#0b1220"
                strokeWidth="1"
                pointerEvents="none"
              />
            )}

            {stats.vals.map((v, i) => (
              <rect
                key={`hit-${i}`}
                x={geo.m.l + geo.band * i}
                y={geo.m.t}
                width={geo.band}
                height={geo.ph}
                fill="transparent"
                className="cursor-crosshair"
                onMouseMove={(e) => {
                  setHover(i);
                  setTip({
                    x: e.clientX,
                    y: e.clientY,
                    color,
                    title: info.shortName,
                    rows: [
                      ['Time', offsetLabel(stats.n, i)],
                      ['UTC', timestamps[i] || '—'],
                      ['Value', `${fmtVal(v)} ${info.unit}`.trim()],
                      ['vs 24h mean', fmtSigned(v - stats.mean)],
                    ],
                  });
                }}
              />
            ))}
          </svg>
        )}
      </div>
    </section>
  );
}

/* ---------------------------------------------------------------------------
   Diverging bars: how far each parameter's T0 value sits from its 24h mean.
--------------------------------------------------------------------------- */
function AnomalyCard({ items, setTip }) {
  const LIMIT = 3;
  return (
    <section className="flex min-h-0 flex-col rounded-xl border border-slate-800/80 bg-slate-900/50 p-3">
      <header className="mb-1.5">
        <h3 className="text-[12.5px] font-semibold tracking-tight text-slate-100">T0 anomaly</h3>
        <p className="mt-0.5 text-[10px] text-slate-500">Now vs 24h mean, in σ (clamped ±{LIMIT})</p>
      </header>
      <div className="flex min-h-0 flex-1 flex-col justify-between gap-px">
        {items.map(({ key, info, color, stats }) => {
          const z = Math.max(-LIMIT, Math.min(LIMIT, stats.z));
          const pct = (Math.abs(z) / LIMIT) * 50;
          return (
            <div
              key={key}
              className="grid grid-cols-[minmax(0,92px)_1fr_38px] items-center gap-2"
              onMouseMove={(e) =>
                setTip({
                  x: e.clientX,
                  y: e.clientY,
                  color,
                  title: info.shortName,
                  rows: [
                    ['Now', fmtVal(stats.current)],
                    ['24h mean', fmtVal(stats.mean)],
                    ['Deviation', `${stats.z >= 0 ? '+' : ''}${stats.z.toFixed(2)}σ`],
                  ],
                })
              }
              onMouseLeave={() => setTip(null)}
            >
              <span className="truncate text-[10px] text-slate-400">{info.shortName}</span>
              <div className="relative h-1.5 rounded-full bg-slate-950">
                <span className="absolute inset-y-[-2px] left-1/2 w-px bg-slate-600" />
                <span
                  className="absolute inset-y-0 rounded-full transition-all duration-500"
                  style={{
                    background: color,
                    width: `${pct}%`,
                    left: z >= 0 ? '50%' : `${50 - pct}%`,
                  }}
                />
              </div>
              <span
                className={`text-right font-mono-numbers text-[10px] ${
                  Math.abs(stats.z) < 0.25 ? 'text-slate-500' : stats.z > 0 ? 'text-amber-300' : 'text-teal-300'
                }`}
              >
                {stats.z >= 0 ? '+' : ''}{stats.z.toFixed(1)}
              </span>
            </div>
          );
        })}
      </div>
    </section>
  );
}

export default function FeatureDashboard({ telemetry, featureNames = [] }) {
  const [tip, setTip] = useState(null);
  const timestamps = telemetry?.timestamps || [];

  const features = useMemo(() => {
    const available = telemetry?.features || {};
    const order = featureNames.length ? featureNames : Object.keys(available);
    return order
      .filter((key) => Array.isArray(available[key]) && available[key].length)
      .map((key, i) => ({ key, info: infoOf(key), color: colorOf(key, i), stats: computeStats(available[key]) }));
  }, [telemetry, featureNames]);

  if (!features.length) {
    return (
      <div className="flex flex-1 items-center justify-center rounded-xl border border-slate-800 bg-slate-900/40 font-mono text-xs text-slate-500">
        Awaiting telemetry window…
      </div>
    );
  }

  const rows = Math.ceil((features.length + 1) / 4);

  return (
    <div className="flex flex-col gap-2.5 lg:min-h-0 lg:flex-1" onMouseLeave={() => setTip(null)}>
      <div className="flex shrink-0 flex-col gap-1.5 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <h2 className="text-sm font-semibold tracking-tight text-slate-50">
            Telemetry features
            <span className="ml-2 text-xs font-normal text-slate-500">24 hourly readings per model input</span>
          </h2>
          <p className="text-[11px] text-slate-500">
            Window <span className="font-mono-numbers text-slate-400">{timestamps[0] || '—'}</span>
            {' → '}
            <span className="font-mono-numbers text-slate-400">{timestamps[timestamps.length - 1] || '—'}</span> UTC ·{' '}
            <span className="font-mono-numbers text-slate-400">{features.length}</span> parameters
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-3 text-[10px] text-slate-400">
          <span className="flex items-center gap-1.5">
            <span className="h-2.5 w-2 rounded-[2px] bg-slate-300 ring-1 ring-white/70" /> T0 (now)
          </span>
          <span className="flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full" style={{ background: PEAK_COLOR }} /> 24h peak
          </span>
          <span className="flex items-center gap-1.5">
            <ArrowUpRight className="h-3 w-3 text-amber-300" /> above 24h mean
          </span>
        </div>
      </div>

      <div
        className="grid auto-rows-[200px] grid-cols-1 gap-2.5 sm:grid-cols-2 lg:min-h-0 lg:flex-1 lg:auto-rows-auto lg:grid-cols-4 lg:[grid-template-rows:repeat(var(--rows),minmax(0,1fr))]"
        style={{ '--rows': rows }}
      >
        {features.map((f, i) => (
          <FeatureBarCard
            key={f.key}
            featKey={f.key}
            stats={f.stats}
            timestamps={timestamps}
            color={f.color}
            index={i}
            setTip={setTip}
          />
        ))}
        <AnomalyCard items={features} setTip={setTip} />
      </div>

      <Tooltip tip={tip} />
    </div>
  );
}
