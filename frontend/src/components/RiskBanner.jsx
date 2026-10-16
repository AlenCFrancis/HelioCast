import React, { useState } from 'react';
import {
  ShieldCheck,
  ShieldAlert,
  AlertCircle,
  CheckCircle,
  AlertTriangle,
  ChevronLeft,
  ChevronRight,
  Play,
  Sparkles,
  Loader2,
} from 'lucide-react';
import ModelComparisonModal from './ModelComparisonModal';
import apiService from '../services/api';

const MODEL_OPTIONS = [
  { id: 'lstm', short: 'LSTM', name: 'LSTM' },
  { id: 'random_forest', short: 'RF', name: 'Random Forest' },
  { id: 'xgboost', short: 'XGB', name: 'XGBoost' },
  { id: 'logistic_regression', short: 'LR', name: 'Logistic Regression' },
];

const TONES = {
  critical: {
    color: '#fb7185',
    verdict: 'Storm likely',
    headline: 'An M-class or stronger solar flare is expected within the next 24 hours.',
    Icon: ShieldAlert,
    border: 'border-rose-700/60',
    glow: 'rgba(244, 63, 94, 0.24)',
    fill: 'linear-gradient(90deg, #f59e0b, #f43f5e)',
  },
  moderate: {
    color: '#fbbf24',
    verdict: 'Elevated risk',
    headline: 'Solar activity is building toward the alert threshold. Keep watching.',
    Icon: AlertCircle,
    border: 'border-amber-700/60',
    glow: 'rgba(245, 158, 11, 0.2)',
    fill: 'linear-gradient(90deg, #10b981, #f59e0b)',
  },
  normal: {
    color: '#34d399',
    verdict: 'No storm expected',
    headline: 'An M-class or stronger flare is unlikely in the next 24 hours.',
    Icon: ShieldCheck,
    border: 'border-emerald-800/60',
    glow: 'rgba(16, 185, 129, 0.16)',
    fill: 'linear-gradient(90deg, #0d9488, #34d399)',
  },
  pending: {
    color: '#94a3b8',
    verdict: 'Awaiting forecast',
    headline: 'Run a forecast to assess the next 24 hours.',
    Icon: AlertCircle,
    border: 'border-slate-700/60',
    glow: 'rgba(100, 116, 139, 0.12)',
    fill: 'linear-gradient(90deg, #475569, #94a3b8)',
  },
};

function toneFor(prediction) {
  if (!prediction) return TONES.pending;
  if (prediction.is_flare || prediction.risk_level === 'CRITICAL') return TONES.critical;
  if (prediction.risk_level === 'MODERATE') return TONES.moderate;
  return TONES.normal;
}

function VerificationChip({ prediction }) {
  if (!prediction || prediction.ground_truth === null || prediction.ground_truth === undefined) return null;
  const actual = prediction.ground_truth === 1;
  const predicted = !!prediction.is_flare;
  let cls = 'bg-emerald-950/50 text-emerald-300 border-emerald-800/60';
  let Icon = CheckCircle;
  let text = `Verified: ${actual ? 'flare occurred' : 'quiet period'}`;
  if (actual && !predicted) {
    cls = 'bg-rose-950/50 text-rose-300 border-rose-800/60';
    Icon = AlertTriangle;
    text = 'Missed event';
  } else if (!actual && predicted) {
    cls = 'bg-amber-950/50 text-amber-300 border-amber-800/60';
    Icon = AlertCircle;
    text = 'False alarm';
  }
  return (
    <span className={`inline-flex items-center gap-1 rounded border px-1.5 py-0.5 text-[10px] font-medium ${cls}`}>
      <Icon className="h-3 w-3" /> {text}
    </span>
  );
}

export default function RiskBanner({
  models = [],
  timestamps = [],
  selectedModel,
  setSelectedModel,
  selectedTimestamp,
  setSelectedTimestamp,
  isLatest,
  setIsLatest,
  prediction,
  loading,
  onExecuteInference,
  autoRun = true,
  setAutoRun,
}) {
  const [compareOpen, setCompareOpen] = useState(false);
  const [compareData, setCompareData] = useState([]);
  const [comparing, setComparing] = useState(false);

  const tone = toneFor(prediction);
  const { Icon } = tone;
  const probPercent = Math.min(100, Math.max(0, prediction?.probability_percent ?? 0));
  const thresholdPercent = (prediction?.threshold ?? 0.5) * 100;
  const isCritical = tone === TONES.critical;

  const availableIds = models.map((m) => m.id);
  const modelOptions = availableIds.length
    ? MODEL_OPTIONS.filter((o) => availableIds.includes(o.id))
    : MODEL_OPTIONS;

  const currentIndex = timestamps.indexOf(selectedTimestamp);
  const safeIndex = currentIndex >= 0 ? currentIndex : Math.max(0, timestamps.length - 1);

  const stepTime = (delta) => {
    const next = Math.min(Math.max(0, safeIndex + delta), timestamps.length - 1);
    if (timestamps[next]) {
      setSelectedTimestamp(timestamps[next]);
      setIsLatest(next === timestamps.length - 1);
    }
  };

  const openComparison = async () => {
    setCompareOpen(true);
    setComparing(true);
    try {
      const target = isLatest ? timestamps[timestamps.length - 1] || null : selectedTimestamp;
      setCompareData(await apiService.compareAllModels(target));
    } catch (err) {
      console.error('Failed to compare models:', err);
    } finally {
      setComparing(false);
    }
  };

  return (
    <section
      className={`relative shrink-0 overflow-hidden rounded-xl border ${tone.border} px-4 py-3 shadow-lg transition-colors duration-500 lg:px-5`}
      style={{
        background: `radial-gradient(120% 160% at 0% 0%, ${tone.glow}, transparent 55%), rgba(15, 23, 42, 0.78)`,
      }}
      aria-live="polite"
    >
      <div className="flex flex-col gap-4 xl:flex-row xl:items-center">
        {/* Verdict */}
        <div className={`flex min-w-0 flex-1 items-center gap-3.5 transition-opacity ${loading ? 'opacity-60' : ''}`}>
          <div
            className="relative grid h-12 w-12 shrink-0 place-items-center rounded-xl border"
            style={{ borderColor: `${tone.color}55`, background: `${tone.color}1a` }}
          >
            {isCritical && <span className="absolute inset-0 animate-ping rounded-xl" style={{ background: `${tone.color}22` }} />}
            <Icon className="relative h-6 w-6" style={{ color: tone.color }} />
          </div>
          <div className="min-w-0">
            <div className="truncate font-mono text-[10px] uppercase tracking-[0.16em] text-slate-400">
              24h M-class flare outlook
              {prediction && (
                <>
                  {' · '}{prediction.model_display_name}
                  {' · T0 '}{prediction.target_timestamp} UTC
                </>
              )}
            </div>
            <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
              <h1 className="text-xl font-bold tracking-tight lg:text-2xl" style={{ color: tone.color }}>
                {tone.verdict}
              </h1>
              <VerificationChip prediction={prediction} />
            </div>
            <p className="truncate text-xs text-slate-300">{tone.headline}</p>
          </div>
        </div>

        {/* Probability + threshold gauge */}
        <div className="flex shrink-0 items-center gap-4">
          <div>
            <div className="font-mono text-[10px] uppercase text-slate-400">Probability</div>
            <div className="font-mono-numbers text-4xl font-bold leading-none tracking-tight" style={{ color: tone.color }}>
              {prediction ? prediction.probability_formatted : '—'}
            </div>
          </div>
          <div className="w-44 space-y-1">
            <div className="relative h-2.5 overflow-hidden rounded-full border border-slate-700/80 bg-slate-950/80">
              <div
                className="absolute inset-y-0 bg-rose-950/50"
                style={{ left: `${thresholdPercent}%`, right: 0 }}
              />
              <div
                className="relative h-full rounded-full transition-all duration-700"
                style={{ width: `${probPercent}%`, background: tone.fill }}
              />
            </div>
            <div className="relative h-3 font-mono-numbers text-[9.5px] text-slate-500">
              <span className="absolute left-0">0</span>
              <span
                className="absolute -translate-x-1/2 whitespace-nowrap text-amber-400"
                style={{ left: `${Math.min(80, Math.max(20, thresholdPercent))}%` }}
              >
                ▲ alert {thresholdPercent.toFixed(1)}%
              </span>
              <span className="absolute right-0">100</span>
            </div>
          </div>
        </div>

        <div className="hidden h-14 w-px bg-slate-700/60 xl:block" />

        {/* Controls */}
        <div className="flex shrink-0 flex-col gap-2">
          <div className="flex items-center gap-2">
            <div className="inline-flex rounded-lg border border-slate-800 bg-slate-950/60 p-0.5" role="radiogroup" aria-label="Model">
              {modelOptions.map((o) => {
                const active = selectedModel === o.id;
                return (
                  <button
                    key={o.id}
                    id={`model-${o.id}`}
                    type="button"
                    role="radio"
                    aria-checked={active}
                    title={o.name}
                    onClick={() => setSelectedModel(o.id)}
                    className={`cursor-pointer rounded-md px-2.5 py-1 text-[11px] font-semibold transition-all ${
                      active ? 'bg-teal-600/90 text-white shadow-sm' : 'text-slate-400 hover:bg-slate-800/70 hover:text-slate-200'
                    }`}
                  >
                    {o.short}
                  </button>
                );
              })}
            </div>
            <button
              id="compare-models"
              type="button"
              onClick={openComparison}
              className="flex cursor-pointer items-center gap-1.5 rounded-lg border border-slate-700 bg-slate-800/70 px-2.5 py-1.5 text-[11px] text-slate-200 transition-colors hover:bg-slate-700"
              title="Evaluate all models at this timestamp"
            >
              <Sparkles className="h-3 w-3 text-amber-400" /> Compare
            </button>
            <label className="ml-auto flex cursor-pointer select-none items-center gap-1.5 text-[11px] text-slate-400">
              <input
                id="auto-run"
                type="checkbox"
                checked={autoRun}
                onChange={(e) => setAutoRun(e.target.checked)}
                className="h-3.5 w-3.5 cursor-pointer rounded border-slate-700 bg-slate-900 accent-teal-500"
              />
              Auto-run
            </label>
          </div>

          <div className="flex items-center gap-2">
            <div className="inline-flex rounded-lg border border-slate-800 bg-slate-950/60 p-0.5 font-mono text-[11px]">
              <button
                id="time-live"
                type="button"
                onClick={() => {
                  setIsLatest(true);
                  if (timestamps.length) setSelectedTimestamp(timestamps[timestamps.length - 1]);
                }}
                className={`cursor-pointer rounded-md px-2 py-1 transition-all ${isLatest ? 'bg-slate-800 font-semibold text-teal-300' : 'text-slate-400 hover:text-slate-200'}`}
              >
                Live
              </button>
              <button
                id="time-archive"
                type="button"
                onClick={() => setIsLatest(false)}
                className={`cursor-pointer rounded-md px-2 py-1 transition-all ${!isLatest ? 'bg-slate-800 font-semibold text-teal-300' : 'text-slate-400 hover:text-slate-200'}`}
              >
                Archive
              </button>
            </div>

            <div className="flex items-center gap-1 rounded-lg border border-slate-800 bg-slate-950/60 p-0.5">
              <button
                id="time-prev"
                type="button"
                onClick={() => stepTime(-1)}
                disabled={safeIndex <= 0}
                className="cursor-pointer rounded p-1 text-slate-300 hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-40"
                title="Step back 1 hour"
              >
                <ChevronLeft className="h-3.5 w-3.5" />
              </button>
              <span className="min-w-[140px] text-center font-mono-numbers text-[11px] font-semibold text-slate-100">
                {selectedTimestamp || 'Latest'}
              </span>
              <button
                id="time-next"
                type="button"
                onClick={() => stepTime(1)}
                disabled={safeIndex >= timestamps.length - 1}
                className="cursor-pointer rounded p-1 text-slate-300 hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-40"
                title="Step forward 1 hour"
              >
                <ChevronRight className="h-3.5 w-3.5" />
              </button>
            </div>

            <button
              id="run-forecast"
              type="button"
              onClick={onExecuteInference}
              disabled={loading}
              className="flex cursor-pointer items-center gap-1.5 rounded-lg bg-teal-600 px-3 py-1.5 text-[11px] font-semibold text-white shadow-md shadow-teal-950/40 transition-colors hover:bg-teal-500 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {loading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Play className="h-3.5 w-3.5" />}
              {loading ? 'Running…' : 'Run'}
            </button>
          </div>

          {!isLatest && timestamps.length > 0 && (
            <input
              id="time-slider"
              type="range"
              min="0"
              max={timestamps.length - 1}
              value={safeIndex}
              onChange={(e) => {
                const idx = parseInt(e.target.value, 10);
                if (timestamps[idx]) {
                  setSelectedTimestamp(timestamps[idx]);
                  setIsLatest(idx === timestamps.length - 1);
                }
              }}
              className="h-1 w-full cursor-pointer appearance-none rounded bg-slate-800 accent-teal-400"
              aria-label="Observation time"
            />
          )}
        </div>
      </div>

      <ModelComparisonModal
        isOpen={compareOpen}
        onClose={() => setCompareOpen(false)}
        comparisonData={compareData}
        loading={comparing}
        targetTimestamp={selectedTimestamp}
        onSelectModel={(id) => setSelectedModel(id)}
      />
    </section>
  );
}
