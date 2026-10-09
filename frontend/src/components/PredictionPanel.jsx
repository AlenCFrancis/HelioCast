import React, { useState } from 'react';
import { 
  Play, 
  Cpu, 
  Clock, 
  AlertTriangle, 
  CheckCircle, 
  ChevronDown, 
  ChevronUp, 
  ChevronLeft, 
  ChevronRight, 
  Layers, 
  ShieldCheck, 
  ShieldAlert, 
  AlertCircle, 
  Sparkles 
} from 'lucide-react';
import ModelComparisonModal from './ModelComparisonModal';
import apiService from '../services/api';

const MODEL_SPECS = [
  {
    id: 'lstm',
    name: 'LSTM',
    badge: 'Neural Net',
    description: 'Reads 24h temporal sequences to spot magnetic energy buildup.',
    tooltip: '2-layer recurrent LSTM capturing continuous flux evolution.',
  },
  {
    id: 'random_forest',
    name: 'Random Forest',
    badge: 'Ensemble',
    description: 'Averages 300 decision trees on the latest solar snapshot.',
    tooltip: 'De-correlated decision trees with balanced sampling.',
  },
  {
    id: 'xgboost',
    name: 'XGBoost',
    badge: 'Boosted',
    description: 'Gradient boosted trees weighted for rare solar flare events.',
    tooltip: 'Iteratively boosted trees with positive class weighting.',
  },
  {
    id: 'logistic_regression',
    name: 'Logistic Reg.',
    badge: 'Baseline',
    description: 'Calibrated linear baseline with balanced class weights.',
    tooltip: 'Transparent linear model calculating calibrated log-odds.',
  },
];

export default function PredictionPanel({
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
  const [techDetailsOpen, setTechDetailsOpen] = useState(false);
  const [comparisonModalOpen, setComparisonModalOpen] = useState(false);
  const [comparisonData, setComparisonData] = useState([]);
  const [comparingLoading, setComparingLoading] = useState(false);

  // Current timestamp index for slider
  const currentIndex = timestamps.indexOf(selectedTimestamp);
  const safeIndex = currentIndex >= 0 ? currentIndex : Math.max(0, timestamps.length - 1);

  // Step backward / forward in time
  const handleStepTime = (delta) => {
    const nextIdx = Math.min(Math.max(0, safeIndex + delta), timestamps.length - 1);
    if (timestamps[nextIdx]) {
      setSelectedTimestamp(timestamps[nextIdx]);
      setIsLatest(false);
    }
  };

  const handleSliderChange = (e) => {
    const idx = parseInt(e.target.value, 10);
    if (timestamps[idx]) {
      setSelectedTimestamp(timestamps[idx]);
      setIsLatest(idx === timestamps.length - 1);
    }
  };

  // Run model comparison
  const handleOpenComparison = async () => {
    setComparisonModalOpen(true);
    setComparingLoading(true);
    try {
      const targetTs = isLatest ? (timestamps[timestamps.length - 1] || null) : selectedTimestamp;
      const res = await apiService.compareAllModels(targetTs);
      setComparisonData(res);
    } catch (err) {
      console.error('Failed to compare models:', err);
    } finally {
      setComparingLoading(false);
    }
  };

  // Determine headline, risk level, and colors
  const isCritical = prediction?.is_flare;
  const riskLevel = prediction?.risk_level || 'NORMAL';
  const probPercent = prediction?.probability_percent ?? 0;
  const threshold = prediction?.threshold ?? 0.5;
  const thresholdPercent = threshold * 100;

  let riskColor = 'text-emerald-400';
  let riskBg = 'bg-emerald-950/20 border-emerald-800/70';
  let headline = 'Low risk: an M-class or larger flare is unlikely in the next 24 hours.';
  let StatusIcon = ShieldCheck;

  if (isCritical || riskLevel === 'CRITICAL') {
    riskColor = 'text-rose-400';
    riskBg = 'bg-rose-950/25 border-rose-800/80 shadow-rose-950/20';
    headline = 'High risk: an M-class or larger solar flare is likely within the next 24 hours.';
    StatusIcon = ShieldAlert;
  } else if (riskLevel === 'MODERATE') {
    riskColor = 'text-amber-400';
    riskBg = 'bg-amber-950/20 border-amber-800/70';
    headline = 'Elevated risk: solar activity is rising towards the decision threshold.';
    StatusIcon = AlertCircle;
  }

  // Ground truth evaluation (Match vs Mismatch)
  let verificationChip = null;
  if (prediction && prediction.ground_truth !== null && prediction.ground_truth !== undefined) {
    const actualFlaresOccurred = prediction.ground_truth === 1;
    const predictedFlare = prediction.is_flare;
    const isMatch = actualFlaresOccurred === predictedFlare;

    if (isMatch) {
      verificationChip = (
        <span className="inline-flex items-center space-x-1 px-2 py-0.5 rounded text-[11px] font-medium bg-emerald-950/40 text-emerald-300 border border-emerald-800/60">
          <CheckCircle className="w-3 h-3 text-emerald-400 shrink-0" />
          <span>Match: {actualFlaresOccurred ? 'Flare erupted' : 'Nominal period'}</span>
        </span>
      );
    } else if (actualFlaresOccurred && !predictedFlare) {
      verificationChip = (
        <span className="inline-flex items-center space-x-1 px-2 py-0.5 rounded text-[11px] font-medium bg-rose-950/40 text-rose-300 border border-rose-800/60">
          <AlertTriangle className="w-3 h-3 text-rose-400 shrink-0" />
          <span>Missed Event</span>
        </span>
      );
    } else {
      verificationChip = (
        <span className="inline-flex items-center space-x-1 px-2 py-0.5 rounded text-[11px] font-medium bg-amber-950/40 text-amber-300 border border-amber-800/60">
          <AlertCircle className="w-3 h-3 text-amber-400 shrink-0" />
          <span>False Alarm</span>
        </span>
      );
    }
  }

  return (
    <div className="flex flex-col space-y-4">
      {/* =========================================================================
          HERO FORECAST RESULT CARD (LANDSCAPE COMPACT & IMPACTFUL)
      ========================================================================= */}
      <div className={`glass-panel rounded-xl p-4 lg:p-5 border ${riskBg} shadow-lg transition-all space-y-3.5`}>
        {/* Top Outcome Badge + Compare Button */}
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center space-x-2">
            <StatusIcon className={`w-5 h-5 ${riskColor} shrink-0`} />
            <span className="text-[11px] font-bold tracking-wider uppercase font-mono text-slate-300">
              24-Hour Hazard Assessment
            </span>
          </div>
          
          <button
            type="button"
            onClick={handleOpenComparison}
            className="text-[11px] font-mono px-2.5 py-1 rounded-md bg-slate-800/80 hover:bg-slate-700 text-slate-200 border border-slate-700 transition-all flex items-center space-x-1.5 cursor-pointer shadow-sm"
            title="Evaluate all 4 models simultaneously"
          >
            <Sparkles className="w-3 h-3 text-amber-400" />
            <span>Compare models</span>
          </button>
        </div>

        {/* Status Headline */}
        <div className="text-sm lg:text-base font-semibold text-slate-100 leading-snug">
          {headline}
        </div>

        {/* Probability Readout + Gauge */}
        <div className="flex items-center justify-between gap-4 pt-1 border-t border-slate-800/60">
          <div>
            <div className="text-[10px] uppercase font-mono text-slate-400">
              Probability
            </div>
            <div className={`text-4xl lg:text-5xl font-bold font-mono-numbers tracking-tight ${riskColor}`}>
              {prediction ? prediction.probability_formatted : '---'}
            </div>
          </div>

          {/* Linear Comparative Gauge */}
          <div className="flex-1 max-w-[210px] space-y-1.5">
            <div className="flex justify-between items-center text-[10px] font-mono text-slate-400">
              <span>0%</span>
              <span className="text-amber-400 font-semibold">T_val: {thresholdPercent.toFixed(1)}%</span>
              <span>100%</span>
            </div>

            <div className="relative w-full h-3 bg-slate-950/80 rounded border border-slate-700/80 overflow-hidden shadow-inner">
              {/* Threshold zone background */}
              <div 
                className="absolute top-0 bottom-0 bg-rose-950/40 border-l-2 border-amber-400 z-0" 
                style={{ left: `${thresholdPercent}%`, right: 0 }}
              />
              {/* Probability fill */}
              <div 
                className={`h-full transition-all duration-500 rounded-l-sm ${
                  isCritical ? 'bg-rose-500' : 'bg-emerald-500'
                }`}
                style={{ width: `${Math.min(100, Math.max(0, probPercent))}%` }}
              />
            </div>

            <div className="flex justify-between items-center text-[10px] font-mono text-slate-400">
              <span>{probPercent >= thresholdPercent ? 'Exceeds threshold' : 'Below threshold'}</span>
              {verificationChip}
            </div>
          </div>
        </div>

        {/* Collapsible Technical Calibration Details */}
        <div className="pt-1.5 border-t border-slate-800/60">
          <button
            type="button"
            onClick={() => setTechDetailsOpen(!techDetailsOpen)}
            className="w-full flex items-center justify-between text-[11px] font-mono text-slate-400 hover:text-slate-200 transition-colors cursor-pointer"
          >
            <span>Technical details & calibration</span>
            {techDetailsOpen ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
          </button>

          {techDetailsOpen && prediction && (
            <div className="mt-2 p-3 rounded-lg bg-slate-950/80 border border-slate-800 text-[11px] font-mono space-y-1.5 text-slate-300 animate-fadeIn">
              <div className="flex justify-between">
                <span className="text-slate-500">Architecture:</span>
                <span className="text-slate-200">{prediction.model_display_name}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">Target T0:</span>
                <span className="text-slate-200">{prediction.target_timestamp} UTC</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">Raw Probability:</span>
                <span className="text-slate-200">{prediction.probability}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">Validation Threshold:</span>
                <span className="text-amber-400">{prediction.threshold}</span>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* =========================================================================
          CONTROLS CARD (MODEL SELECTOR + TIME SELECTOR + PRIMARY RUN BUTTON)
      ========================================================================= */}
      <div className="glass-panel rounded-xl p-4 lg:p-5 space-y-3.5 shadow-lg">
        {/* Header with Auto-run */}
        <div className="flex items-center justify-between border-b border-slate-800 pb-2.5">
          <div className="flex items-center space-x-2">
            <Layers className="w-4 h-4 text-teal-400" />
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-200">
              Forecast Controls
            </h3>
          </div>

          {/* Auto-run Toggle */}
          <label className="flex items-center space-x-1.5 text-[11px] text-slate-400 cursor-pointer select-none">
            <input
              type="checkbox"
              checked={autoRun}
              onChange={(e) => setAutoRun(e.target.checked)}
              className="w-3.5 h-3.5 rounded border-slate-700 bg-slate-900 text-teal-500 focus:ring-0 focus:ring-offset-0 cursor-pointer"
            />
            <span>Auto-run</span>
          </label>
        </div>

        {/* 1. Model Selector Cards (2x2 Compact Grid) */}
        <div className="space-y-1.5">
          <label className="text-[11px] font-semibold text-slate-400 uppercase tracking-wide block">
            Model Architecture
          </label>
          <div className="grid grid-cols-2 gap-2">
            {MODEL_SPECS.map((spec) => {
              const isSelected = selectedModel === spec.id;
              return (
                <div
                  key={spec.id}
                  onClick={() => setSelectedModel(spec.id)}
                  title={spec.tooltip}
                  className={`p-2.5 rounded-lg border text-left cursor-pointer transition-all ${
                    isSelected
                      ? 'bg-teal-950/35 border-teal-500 shadow-sm shadow-teal-950/30'
                      : 'bg-slate-900/50 border-slate-800 hover:border-slate-700 hover:bg-slate-900/80'
                  }`}
                >
                  <div className="flex items-center justify-between mb-0.5">
                    <span className="text-xs font-bold text-slate-100 truncate">{spec.name}</span>
                    <span className={`text-[9px] font-mono px-1 py-0.2 rounded ${
                      isSelected ? 'bg-teal-900/60 text-teal-300' : 'bg-slate-800 text-slate-400'
                    }`}>
                      {spec.badge}
                    </span>
                  </div>
                  <p className="text-[10px] text-slate-400 leading-tight line-clamp-1">
                    {spec.description}
                  </p>
                </div>
              );
            })}
          </div>
        </div>

        {/* 2. Time Selector (Toggle + Arrow Stepping + Slider) */}
        <div className="space-y-2 pt-2 border-t border-slate-800/80">
          <div className="flex items-center justify-between">
            <label className="text-[11px] font-semibold text-slate-400 uppercase tracking-wide">
              Observation Time (T0)
            </label>

            {/* Live / Historical Toggle */}
            <div className="inline-flex rounded-md bg-slate-900 p-0.5 border border-slate-800 text-[11px] font-mono">
              <button
                type="button"
                onClick={() => {
                  setIsLatest(true);
                  if (timestamps.length > 0) {
                    setSelectedTimestamp(timestamps[timestamps.length - 1]);
                  }
                }}
                className={`px-2.5 py-0.5 rounded transition-all cursor-pointer ${
                  isLatest
                    ? 'bg-slate-800 text-teal-300 font-semibold shadow-xs'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                Live
              </button>
              <button
                type="button"
                onClick={() => setIsLatest(false)}
                className={`px-2.5 py-0.5 rounded transition-all cursor-pointer ${
                  !isLatest
                    ? 'bg-slate-800 text-teal-300 font-semibold shadow-xs'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                Archive
              </button>
            </div>
          </div>

          {/* Time readout with arrows */}
          <div className="flex items-center justify-between gap-2 p-2 rounded-lg bg-slate-950/60 border border-slate-800">
            <button
              type="button"
              onClick={() => handleStepTime(-1)}
              disabled={safeIndex <= 0}
              className="p-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 transition-colors disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer"
              title="Step -1 hour"
            >
              <ChevronLeft className="w-3.5 h-3.5" />
            </button>

            <div className="text-center font-mono flex-1">
              <div className="text-xs font-bold text-slate-100 font-mono-numbers truncate">
                {selectedTimestamp || 'Latest Available'} UTC
              </div>
              <div className="text-[10px] text-slate-500">
                {isLatest ? 'Latest real-time record' : `Sample ${safeIndex + 1} of ${timestamps.length}`}
              </div>
            </div>

            <button
              type="button"
              onClick={() => handleStepTime(1)}
              disabled={safeIndex >= timestamps.length - 1}
              className="p-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 transition-colors disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer"
              title="Step +1 hour"
            >
              <ChevronRight className="w-3.5 h-3.5" />
            </button>
          </div>

          {/* Slider if in Historical mode */}
          {!isLatest && timestamps.length > 0 && (
            <div className="pt-0.5">
              <input
                type="range"
                min="0"
                max={timestamps.length - 1}
                value={safeIndex}
                onChange={handleSliderChange}
                className="w-full h-1.5 bg-slate-800 rounded appearance-none cursor-pointer accent-teal-400"
              />
            </div>
          )}
        </div>

        {/* 3. Primary Action Button */}
        <div className="pt-1">
          <button
            type="button"
            onClick={onExecuteInference}
            disabled={loading}
            className="w-full py-2.5 px-4 rounded-lg bg-teal-600 hover:bg-teal-500 active:bg-teal-700 text-white font-semibold text-xs shadow-md shadow-teal-950/40 transition-all flex items-center justify-center space-x-2 cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Play className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
            <span>{loading ? 'Evaluating Model...' : 'Run forecast'}</span>
          </button>
        </div>
      </div>

      {/* Model Comparison Dialog Modal */}
      <ModelComparisonModal
        isOpen={comparisonModalOpen}
        onClose={() => setComparisonModalOpen(false)}
        comparisonData={comparisonData}
        loading={comparingLoading}
        targetTimestamp={selectedTimestamp}
        onSelectModel={(newModelId) => {
          setSelectedModel(newModelId);
        }}
      />
    </div>
  );
}
