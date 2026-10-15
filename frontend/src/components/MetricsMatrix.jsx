import React, { useState, useMemo, useRef, useLayoutEffect } from 'react';

/* ---------------------------------------------------------------------------
   Model identity — one color per architecture, reused in every chart so the
   eye can track a model across panels.
--------------------------------------------------------------------------- */
const MODEL_STYLE = {
  lstm: { color: '#2dd4bf', short: 'LSTM' },
  xgboost: { color: '#fbbf24', short: 'XGB' },
  random_forest: { color: '#a78bfa', short: 'RF' },
  logistic_regression: { color: '#60a5fa', short: 'LR' },
};
const FALLBACK_COLORS = ['#f472b6', '#34d399', '#fb923c', '#94a3b8'];

const PANELS = [
  { key: 'tss', label: 'TSS', name: 'True Skill Statistic', note: 'Recall − false-positive rate' },
  { key: 'pr_auc', label: 'PR-AUC', name: 'Precision–Recall AUC', note: 'Penalises false alarms on rare events', baseline: 'prevalence' },
  { key: 'roc_auc', label: 'ROC-AUC', name: 'ROC AUC', note: 'Ranking quality across all thresholds', baseline: 0.5 },
  { key: 'f1_score', label: 'F1', name: 'F1 Score', note: 'Harmonic mean of precision and recall' },
  { key: 'recall', label: 'Recall', name: 'Detection rate', note: 'Share of real M+ flares caught' },
  { key: 'precision', label: 'Precision', name: 'Alert reliability', note: 'Share of alerts that were real' },
];

const TABLE_COLS = [
  { key: 'tss', label: 'TSS', higher: true },
  { key: 'pr_auc', label: 'PR-AUC', higher: true },
  { key: 'roc_auc', label: 'ROC-AUC', higher: true },
  { key: 'f1_score', label: 'F1', higher: true },
  { key: 'recall', label: 'Recall', higher: true },
  { key: 'precision', label: 'Precision', higher: true },
  { key: 'far', label: 'FAR', higher: false },
  { key: 'threshold', label: 'Threshold', higher: null },
];

const fmt = (v, d = 3) => (typeof v === 'number' ? v.toFixed(d) : '—');

function useSize() {
  const ref = useRef(null);
  const [width, setWidth] = useState(0);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return undefined;
    setWidth(el.clientWidth);
    const ro = new ResizeObserver(([entry]) => setWidth(entry.contentRect.width));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, width];
}

function barPath(x, y, w, h, r) {
  if (h <= 0) return '';
  const rr = Math.min(r, w / 2, h);
  return `M${x},${y + h} V${y + rr} Q${x},${y} ${x + rr},${y} H${x + w - rr} Q${x + w},${y} ${x + w},${y + rr} V${y + h} Z`;
}

function Tooltip({ tip }) {
  if (!tip) return null;
  const flip = tip.x > window.innerWidth - 230;
  return (
    <div
      className="fixed z-50 pointer-events-none rounded-lg border border-slate-700/80 bg-slate-950/95 px-3 py-2 shadow-2xl backdrop-blur-md"
      style={{ left: flip ? tip.x - 14 : tip.x + 14, top: tip.y + 14, transform: flip ? 'translateX(-100%)' : 'none' }}
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

function Card({ title, subtitle, right, children, className = '' }) {
  return (
    <section className={`rounded-xl border border-slate-800/80 bg-slate-900/50 p-4 ${className}`}>
      {(title || right) && (
        <header className="mb-3 flex items-start justify-between gap-3">
          <div>
            <h3 className="text-[13px] font-semibold tracking-tight text-slate-100">{title}</h3>
            {subtitle && <p className="mt-0.5 text-[11px] leading-snug text-slate-500">{subtitle}</p>}
          </div>
          {right}
        </header>
      )}
      {children}
    </section>
  );
}

function BarPanel({ def, rows, styleOf, active, setHover, setTip, baseline, index }) {
  const [ref, w] = useSize();
  const H = 190;
  const m = { t: 20, r: 8, b: 24, l: 30 };
  const pw = Math.max(0, w - m.l - m.r);
  const ph = H - m.t - m.b;
  const y = (v) => m.t + ph * (1 - v);
  const band = pw / rows.length;
  const bw = Math.min(42, band * 0.58);

  const values = rows.map((r) => r[def.key] ?? 0);
  const best = Math.max(...values);
  const leader = rows[values.indexOf(best)];
  const ranked = [...rows].sort((a, b) => (b[def.key] ?? 0) - (a[def.key] ?? 0)).map((r) => r.model_key);

  const baseVal = def.baseline === 'prevalence' ? baseline.prevalence : def.baseline;

  return (
    <Card
      title={def.label}
      subtitle={def.note}
      right={
        <span className="flex items-center gap-1.5 rounded-md border border-slate-800 bg-slate-950/60 px-2 py-1 text-[10px] font-mono-numbers text-slate-400">
          <span className="h-1.5 w-1.5 rounded-full" style={{ background: styleOf(leader).color }} />
          {styleOf(leader).short} <span className="text-slate-200">{fmt(best)}</span>
        </span>
      }
    >
      <div ref={ref} className="w-full">
        {w > 0 && (
          <svg width={w} height={H} role="img" aria-label={`${def.name} by model`}>
            <defs>
              {rows.map((r) => (
                <linearGradient key={r.model_key} id={`bar-${def.key}-${r.model_key}`} x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={styleOf(r).color} stopOpacity="1" />
                  <stop offset="100%" stopColor={styleOf(r).color} stopOpacity="0.45" />
                </linearGradient>
              ))}
            </defs>

            {[0, 0.25, 0.5, 0.75, 1].map((t) => (
              <g key={t}>
                <line
                  x1={m.l} x2={m.l + pw} y1={y(t)} y2={y(t)}
                  stroke={t === 0 ? '#334155' : '#1e293b'} strokeWidth="1"
                />
                {(t === 0 || t === 0.5 || t === 1) && (
                  <text x={m.l - 6} y={y(t) + 3} textAnchor="end" className="fill-slate-500" fontSize="9.5" fontFamily="JetBrains Mono, monospace">
                    {t === 0 ? '0' : t === 1 ? '1.0' : t.toFixed(1)}
                  </text>
                )}
              </g>
            ))}

            {rows.map((r, i) => {
              const v = r[def.key] ?? 0;
              const cx = m.l + band * i + band / 2;
              const x = cx - bw / 2;
              const bh = ph * v;
              const dim = active && active !== r.model_key;
              const isBest = v === best;
              return (
                <g
                  key={r.model_key}
                  style={{ opacity: dim ? 0.22 : 1, transition: 'opacity 150ms' }}
                  onMouseEnter={() => setHover(r.model_key)}
                  onMouseMove={(e) =>
                    setTip({
                      x: e.clientX, y: e.clientY, color: styleOf(r).color, title: r.model_name,
                      rows: [
                        [def.label, fmt(v, 4)],
                        ['Rank', `${ranked.indexOf(r.model_key) + 1} of ${rows.length}`],
                        ['Δ vs best', isBest ? 'leader' : (v - best).toFixed(4)],
                      ],
                    })
                  }
                  onMouseLeave={() => { setHover(null); setTip(null); }}
                >
                  <rect x={cx - band / 2} y={m.t} width={band} height={ph} fill="transparent" />
                  <path
                    d={barPath(x, y(v), bw, bh, 4)}
                    fill={`url(#bar-${def.key}-${r.model_key})`}
                    className="bar-grow"
                    style={{ animationDelay: `${index * 40 + i * 60}ms` }}
                  />
                  {isBest && <path d={barPath(x, y(v), bw, bh, 4)} fill="none" stroke="#f8fafc" strokeOpacity="0.55" strokeWidth="1" />}
                  <text
                    x={cx} y={y(v) - 6} textAnchor="middle" fontSize="10.5" fontFamily="JetBrains Mono, monospace"
                    className={isBest ? 'fill-slate-50' : 'fill-slate-400'} fontWeight={isBest ? 700 : 500}
                  >
                    {fmt(v)}
                  </text>
                  <text x={cx} y={H - 8} textAnchor="middle" fontSize="10" fontWeight="600" className="fill-slate-400">
                    {styleOf(r).short}
                  </text>
                </g>
              );
            })}

            {baseVal !== undefined && baseVal !== null && (
              <g pointerEvents="none">
                <line
                  x1={m.l} x2={m.l + pw} y1={y(baseVal)} y2={y(baseVal)}
                  stroke="#f8fafc" strokeOpacity="0.5" strokeDasharray="3 3"
                />
                <text x={m.l + pw} y={y(baseVal) - 4} textAnchor="end" fontSize="9" className="fill-slate-400" fontFamily="JetBrains Mono, monospace">
                  no skill {baseVal.toFixed(2)}
                </text>
              </g>
            )}
          </svg>
        )}
      </div>
    </Card>
  );
}

function OperatingPointPlot({ rows, styleOf, active, setHover, setTip }) {
  const [ref, w] = useSize();
  const H = 300;
  const m = { t: 14, r: 40, b: 38, l: 46 };
  const pw = Math.max(0, w - m.l - m.r);
  const ph = H - m.t - m.b;

  const yMax = Math.max(0.5, Math.ceil(Math.max(...rows.map((r) => r.precision)) * 10 + 2) / 10);
  const x = (v) => m.l + pw * v;
  const y = (v) => m.t + ph * (1 - v / yMax);

  const isoCurves = [0.2, 0.4, 0.6].map((f) => {
    const pts = [];
    for (let r = f / 2 + 0.002; r <= 1.0001; r += 0.01) {
      const p = (f * r) / (2 * r - f);
      if (p <= yMax) pts.push([x(r), y(p)]);
    }
    return { f, d: pts.map(([px, py], i) => `${i ? 'L' : 'M'}${px.toFixed(1)},${py.toFixed(1)}`).join(' '), end: pts[pts.length - 1] };
  });

  const yTicks = [];
  for (let t = 0; t <= yMax + 1e-9; t += 0.1) yTicks.push(Number(t.toFixed(2)));

  return (
    <div ref={ref} className="w-full">
      {w > 0 && (
        <svg width={w} height={H} role="img" aria-label="Precision versus recall operating points">
          {yTicks.map((t) => (
            <g key={t}>
              <line x1={m.l} x2={m.l + pw} y1={y(t)} y2={y(t)} stroke={t === 0 ? '#334155' : '#1e293b'} />
              <text x={m.l - 8} y={y(t) + 3} textAnchor="end" fontSize="9.5" fontFamily="JetBrains Mono, monospace" className="fill-slate-500">
                {t.toFixed(1)}
              </text>
            </g>
          ))}
          {[0, 0.25, 0.5, 0.75, 1].map((t) => (
            <g key={t}>
              <line x1={x(t)} x2={x(t)} y1={m.t} y2={m.t + ph} stroke="#1e293b" />
              <text x={x(t)} y={m.t + ph + 15} textAnchor="middle" fontSize="9.5" fontFamily="JetBrains Mono, monospace" className="fill-slate-500">
                {t.toFixed(2)}
              </text>
            </g>
          ))}
          <text x={m.l + pw / 2} y={H - 6} textAnchor="middle" fontSize="10" className="fill-slate-400" fontWeight="600">
            Recall (flares caught)
          </text>
          <text transform={`translate(12 ${m.t + ph / 2}) rotate(-90)`} textAnchor="middle" fontSize="10" className="fill-slate-400" fontWeight="600">
            Precision (alerts that were real)
          </text>

          {isoCurves.map((c) => (
            <g key={c.f} pointerEvents="none">
              <path d={c.d} fill="none" stroke="#64748b" strokeOpacity="0.55" strokeDasharray="2 4" />
              {c.end && (
                <text x={c.end[0] + 5} y={c.end[1] + 3} fontSize="9" fontFamily="JetBrains Mono, monospace" className="fill-slate-500">
                  F1 {c.f.toFixed(1)}
                </text>
              )}
            </g>
          ))}

          {rows.map((r) => {
            const s = styleOf(r);
            const cx = x(r.recall);
            const cy = y(r.precision);
            const dim = active && active !== r.model_key;
            const isActive = active === r.model_key;
            const anchorEnd = r.recall > 0.8;
            return (
              <g
                key={r.model_key}
                style={{ opacity: dim ? 0.22 : 1, transition: 'opacity 150ms' }}
                onMouseEnter={() => setHover(r.model_key)}
                onMouseMove={(e) =>
                  setTip({
                    x: e.clientX, y: e.clientY, color: s.color, title: r.model_name,
                    rows: [
                      ['Recall', fmt(r.recall, 4)],
                      ['Precision', fmt(r.precision, 4)],
                      ['F1', fmt(r.f1_score, 4)],
                      ['Threshold', fmt(r.threshold, 2)],
                    ],
                  })
                }
                onMouseLeave={() => { setHover(null); setTip(null); }}
              >
                <circle cx={cx} cy={cy} r="16" fill="transparent" />
                <circle cx={cx} cy={cy} r={isActive ? 13 : 10} fill={s.color} fillOpacity="0.16" style={{ transition: 'r 150ms' }} />
                <circle cx={cx} cy={cy} r="5.5" fill={s.color} stroke="#0b1220" strokeWidth="1.5" />
                <text
                  x={cx + (anchorEnd ? -12 : 12)} y={cy - 9} textAnchor={anchorEnd ? 'end' : 'start'}
                  fontSize="11" fontWeight="600" fill={s.color}
                >
                  {s.short}
                </text>
              </g>
            );
          })}
        </svg>
      )}
    </div>
  );
}

const MISS_STRIPE = {
  backgroundImage:
    'repeating-linear-gradient(135deg, rgba(148,163,184,.22) 0 4px, rgba(148,163,184,.07) 4px 8px)',
};

function StackedBar({ good, bad, color, goodLabel, badLabel }) {
  const total = good + bad || 1;
  const gp = (good / total) * 100;
  return (
    <div className="flex h-6 w-full overflow-hidden rounded-md bg-slate-950" title={`${goodLabel}: ${good} · ${badLabel}: ${bad}`}>
      <div
        className="flex items-center justify-start pl-2 text-[10px] font-semibold text-slate-950 font-mono-numbers transition-all duration-500"
        style={{ width: `${gp}%`, background: color }}
      >
        {gp > 12 ? good : ''}
      </div>
      <div
        className="flex flex-1 items-center justify-end pr-2 text-[10px] text-slate-300 font-mono-numbers"
        style={MISS_STRIPE}
      >
        {100 - gp > 12 ? bad : ''}
      </div>
    </div>
  );
}

function OutcomeBreakdown({ rows, styleOf, active, setHover, baseline }) {
  return (
    <div className="space-y-1">
      <div className="grid grid-cols-[84px_1fr_40px_1fr_40px] items-end gap-x-3 pb-2 text-[10px] uppercase tracking-wider text-slate-500">
        <span />
        <span className="col-span-2 normal-case tracking-normal">
          <span className="font-semibold uppercase tracking-wider text-slate-400">Flares caught</span>
          <span className="ml-1.5">of {baseline.positives} real</span>
        </span>
        <span className="col-span-2 normal-case tracking-normal">
          <span className="font-semibold uppercase tracking-wider text-slate-400">Alerts correct</span>
          <span className="ml-1.5">of alerts issued</span>
        </span>
      </div>
      {rows.map((r) => {
        const s = styleOf(r);
        const cm = r.confusion_matrix;
        const dim = active && active !== r.model_key;
        if (!cm) return null;
        return (
          <div
            key={r.model_key}
            className="grid grid-cols-[84px_1fr_40px_1fr_40px] items-center gap-x-3 rounded-lg px-1 py-1.5 transition-opacity duration-150"
            style={{ opacity: dim ? 0.25 : 1 }}
            onMouseEnter={() => setHover(r.model_key)}
            onMouseLeave={() => setHover(null)}
          >
            <span className="flex items-center gap-2 text-[11px] font-semibold text-slate-200">
              <span className="h-2 w-2 rounded-sm" style={{ background: s.color }} />
              {s.short}
            </span>
            <StackedBar good={cm.tp} bad={cm.fn} color={s.color} goodLabel="Caught" badLabel="Missed" />
            <span className="text-right text-[11px] text-slate-300 font-mono-numbers">
              {((cm.tp / (cm.tp + cm.fn || 1)) * 100).toFixed(0)}%
            </span>
            <StackedBar good={cm.tp} bad={cm.fp} color={s.color} goodLabel="Real" badLabel="False alarm" />
            <span className="text-right text-[11px] text-slate-300 font-mono-numbers">
              {((cm.tp / (cm.tp + cm.fp || 1)) * 100).toFixed(0)}%
            </span>
          </div>
        );
      })}
      <div className="flex items-center gap-4 pt-3 text-[10px] text-slate-500">
        <span className="flex items-center gap-1.5"><span className="h-2 w-3 rounded-sm bg-slate-300" /> correct</span>
        <span className="flex items-center gap-1.5"><span className="h-2 w-3 rounded-sm" style={MISS_STRIPE} /> missed flare / false alarm</span>
      </div>
    </div>
  );
}

function ReferenceTable({ rows, styleOf, active, setHover }) {
  const best = useMemo(() => {
    const out = {};
    TABLE_COLS.forEach((c) => {
      if (c.higher === null) return;
      const vals = rows.map((r) => r[c.key] ?? 0);
      out[c.key] = c.higher ? Math.max(...vals) : Math.min(...vals);
    });
    return out;
  }, [rows]);

  const cmCols = ['tn', 'fp', 'fn', 'tp'];
  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse text-[11px] font-mono-numbers">
        <thead>
          <tr className="border-b border-slate-800 text-[10px] uppercase tracking-wider text-slate-500">
            <th className="py-2 pr-4 text-left font-medium">Model</th>
            {TABLE_COLS.map((c) => (
              <th key={c.key} className="px-3 py-2 text-right font-medium">{c.label}</th>
            ))}
            {cmCols.map((c) => (
              <th key={c} className="px-3 py-2 text-right font-medium">{c.toUpperCase()}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            const s = styleOf(r);
            const dim = active && active !== r.model_key;
            return (
              <tr
                key={r.model_key}
                className="border-b border-slate-800/60 transition-opacity duration-150 hover:bg-slate-800/30"
                style={{ opacity: dim ? 0.3 : 1 }}
                onMouseEnter={() => setHover(r.model_key)}
                onMouseLeave={() => setHover(null)}
              >
                <td className="py-2.5 pr-4 font-sans text-[12px] font-semibold text-slate-100">
                  <span className="mr-2 inline-block h-2 w-2 rounded-sm align-middle" style={{ background: s.color }} />
                  {r.model_name}
                </td>
                {TABLE_COLS.map((c) => {
                  const v = r[c.key];
                  const isBest = c.higher !== null && v === best[c.key];
                  return (
                    <td
                      key={c.key}
                      className={`px-3 py-2.5 text-right ${isBest ? 'font-bold text-slate-50' : 'text-slate-400'}`}
                    >
                      {fmt(v, c.key === 'threshold' ? 2 : 4)}
                      {isBest && <span className="ml-1 text-[8px] align-middle" style={{ color: s.color }}>●</span>}
                    </td>
                  );
                })}
                {cmCols.map((c) => (
                  <td key={c} className="px-3 py-2.5 text-right text-slate-400">
                    {r.confusion_matrix ? r.confusion_matrix[c] : '—'}
                  </td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export default function MetricsMatrix({ metrics = [] }) {
  const [hover, setHover] = useState(null);
  const [pinned, setPinned] = useState(null);
  const [tip, setTip] = useState(null);
  const active = hover ?? pinned;

  const rows = useMemo(() => [...metrics].sort((a, b) => (b.tss ?? 0) - (a.tss ?? 0)), [metrics]);

  const styleOf = useMemo(() => {
    const idx = {};
    rows.forEach((r, i) => { idx[r.model_key] = i; });
    return (r) =>
      MODEL_STYLE[r.model_key] || {
        color: FALLBACK_COLORS[(idx[r.model_key] ?? 0) % FALLBACK_COLORS.length],
        short: r.model_name?.slice(0, 4).toUpperCase() || r.model_key,
      };
  }, [rows]);

  const baseline = useMemo(() => {
    const cm = rows.find((r) => r.confusion_matrix)?.confusion_matrix;
    if (!cm) return { total: 0, positives: 0, prevalence: 0 };
    const total = cm.tn + cm.fp + cm.fn + cm.tp;
    const positives = cm.fn + cm.tp;
    return { total, positives, prevalence: total ? positives / total : 0 };
  }, [rows]);

  if (!rows.length) {
    return (
      <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-10 text-center text-sm text-slate-500">
        No benchmark data available. Verify the backend is running.
      </div>
    );
  }

  return (
    <div className="space-y-4" onMouseLeave={() => setTip(null)}>
      <div className="flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <h2 className="text-lg font-semibold tracking-tight text-slate-50">Model benchmarks</h2>
          <p className="mt-0.5 text-xs text-slate-500">
            Held-out chronological test split ·{' '}
            <span className="font-mono-numbers text-slate-400">{baseline.total}</span> windows ·{' '}
            <span className="font-mono-numbers text-slate-400">{baseline.positives}</span> M+ flares (
            <span className="font-mono-numbers text-slate-400">{(baseline.prevalence * 100).toFixed(1)}%</span> base rate) ·
            thresholds frozen on validation
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-1.5" role="group" aria-label="Highlight model">
          {rows.map((r) => {
            const s = styleOf(r);
            const isPinned = pinned === r.model_key;
            return (
              <button
                key={r.model_key}
                type="button"
                aria-pressed={isPinned}
                onMouseEnter={() => setHover(r.model_key)}
                onMouseLeave={() => setHover(null)}
                onClick={() => setPinned(isPinned ? null : r.model_key)}
                className={`flex cursor-pointer items-center gap-2 rounded-lg border px-2.5 py-1.5 text-[11px] font-medium transition-colors ${
                  isPinned
                    ? 'border-slate-500 bg-slate-800 text-slate-50'
                    : 'border-slate-800 bg-slate-900/60 text-slate-300 hover:border-slate-600'
                }`}
              >
                <span className="h-2 w-2 rounded-sm" style={{ background: s.color }} />
                {r.model_name}
              </button>
            );
          })}
        </div>
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
        {PANELS.map((def, i) => (
          <BarPanel
            key={def.key}
            def={def}
            index={i}
            rows={rows}
            styleOf={styleOf}
            active={active}
            setHover={setHover}
            setTip={setTip}
            baseline={baseline}
          />
        ))}
      </div>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-5">
        <Card
          className="xl:col-span-3"
          title="Operating points"
          subtitle="Where each model sits on the precision–recall trade-off. Dotted curves are lines of constant F1."
        >
          <OperatingPointPlot rows={rows} styleOf={styleOf} active={active} setHover={setHover} setTip={setTip} />
        </Card>
        <Card
          className="xl:col-span-2"
          title="What the alerts looked like"
          subtitle="Outcome counts at each model's frozen decision threshold."
        >
          <OutcomeBreakdown rows={rows} styleOf={styleOf} active={active} setHover={setHover} baseline={baseline} />
        </Card>
      </div>

      <Card
        title="Full results"
        subtitle="● marks the best value per column (lowest for FAR). TN / FP / FN / TP are test-set confusion counts."
      >
        <ReferenceTable rows={rows} styleOf={styleOf} active={active} setHover={setHover} />
      </Card>

      <Tooltip tip={tip} />
    </div>
  );
}
