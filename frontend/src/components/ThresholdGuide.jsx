import React from 'react';
import { 
  Lock, 
  Sliders, 
  ShieldCheck, 
  Scale, 
  HelpCircle, 
  BookOpen, 
  CheckCircle2,
  AlertTriangle,
  Layers
} from 'lucide-react';

export default function ThresholdGuide({ metrics = [] }) {
  return (
    <div className="space-y-8">
      {/* =========================================================================
          SECTION 1: CORE METHODOLOGY (3 EXPLANATORY CARDS)
      ========================================================================= */}
      <div className="glass-panel rounded-2xl p-6 md:p-8 space-y-6">
        <div className="flex items-center space-x-3 border-b border-slate-800 pb-5">
          <div className="w-9 h-9 rounded-xl bg-slate-800/80 border border-slate-700/80 flex items-center justify-center text-teal-400">
            <BookOpen className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-base font-bold text-slate-100">
              Operational Forecasting Principles
            </h3>
            <p className="text-xs text-slate-400 font-mono">
              Scientific methodology for decision threshold freezing and zero lookahead leakage
            </p>
          </div>
        </div>

        {/* 3 Large Spaced Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Card 1: The Imbalance Problem */}
          <div className="p-6 rounded-2xl bg-slate-950/60 border border-slate-800/80 flex flex-col justify-between space-y-4 shadow-sm">
            <div className="space-y-2">
              <div className="w-8 h-8 rounded-lg bg-rose-950/40 border border-rose-800/60 flex items-center justify-center text-rose-400 mb-2">
                <Scale className="w-4 h-4" />
              </div>
              <h4 className="text-sm font-bold text-slate-100">
                1. Severe Class Imbalance
              </h4>
              <p className="text-xs text-slate-400 leading-relaxed font-sans">
                M-class or higher solar flares are rare events occurring in less than 5% of observation hours. Using a standard 50% probability threshold results in almost zero warnings, dangerously missing extreme solar radiation hazards.
              </p>
            </div>
            <div className="text-[11px] font-mono text-rose-400/90 bg-rose-950/20 p-2.5 rounded-lg border border-rose-900/40">
              Risk: Default 0.50 threshold creates unacceptable false negative rates.
            </div>
          </div>

          {/* Card 2: Validation Freezing */}
          <div className="p-6 rounded-2xl bg-slate-950/60 border border-slate-800/80 flex flex-col justify-between space-y-4 shadow-sm">
            <div className="space-y-2">
              <div className="w-8 h-8 rounded-lg bg-amber-950/40 border border-amber-800/60 flex items-center justify-center text-amber-400 mb-2">
                <Sliders className="w-4 h-4" />
              </div>
              <h4 className="text-sm font-bold text-slate-100">
                2. Validation Optimization
              </h4>
              <p className="text-xs text-slate-400 leading-relaxed font-sans">
                Each model's decision threshold was tuned strictly on the historical 15% Validation Split to optimize the True Skill Statistic (TSS = Recall - FPR) and F1 Score. Once chosen, thresholds were permanently locked before evaluating the test set.
              </p>
            </div>
            <div className="text-[11px] font-mono text-amber-400/90 bg-amber-950/20 p-2.5 rounded-lg border border-amber-900/40">
              Rule: P(Flare) &gt;= Frozen Threshold triggers an operational hazard alert.
            </div>
          </div>

          {/* Card 3: Anti-Leakage Safeguard */}
          <div className="p-6 rounded-2xl bg-slate-950/60 border border-slate-800/80 flex flex-col justify-between space-y-4 shadow-sm">
            <div className="space-y-2">
              <div className="w-8 h-8 rounded-lg bg-emerald-950/40 border border-emerald-800/60 flex items-center justify-center text-emerald-400 mb-2">
                <ShieldCheck className="w-4 h-4" />
              </div>
              <h4 className="text-sm font-bold text-slate-100">
                3. Zero-Leakage Guarantee
              </h4>
              <p className="text-xs text-slate-400 leading-relaxed font-sans">
                Data was split strictly chronologically (70% Train, 15% Val, 15% Test) without random shuffling. Feature scalers and median imputations were fitted exclusively on the training split, preventing any lookahead contamination.
              </p>
            </div>
            <div className="text-[11px] font-mono text-emerald-400/90 bg-emerald-950/20 p-2.5 rounded-lg border border-emerald-900/40">
              Guarantee: Test evaluations represent genuine generalization on unseen future data.
            </div>
          </div>
        </div>
      </div>

      {/* =========================================================================
          SECTION 2: FROZEN THRESHOLDS REFERENCE TABLE
      ========================================================================= */}
      <div className="glass-panel rounded-2xl p-6 md:p-8 space-y-6">
        <div className="flex items-center space-x-3 border-b border-slate-800 pb-4">
          <div className="w-9 h-9 rounded-xl bg-slate-800/80 border border-slate-700/80 flex items-center justify-center text-amber-400">
            <Lock className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-base font-bold text-slate-100">
              Active Operational Thresholds
            </h3>
            <p className="text-xs text-slate-400 font-mono">
              Calibrated cutoffs applied in real-time inference
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {metrics.map((m) => (
            <div key={m.model_key} className="p-4 rounded-xl bg-slate-950/70 border border-slate-800 flex flex-col justify-between space-y-2 shadow-sm">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-slate-200">{m.model_name}</span>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-400">
                  Fixed
                </span>
              </div>
              <div className="text-3xl font-bold font-mono-numbers text-amber-400">
                {(m.threshold * 100).toFixed(1)}%
              </div>
              <div className="text-[11px] text-slate-400 font-mono">
                Raw cutoff: {m.threshold.toFixed(4)}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* =========================================================================
          SECTION 3: SPACE WEATHER TERMINOLOGY GLOSSARY
      ========================================================================= */}
      <div className="glass-panel rounded-2xl p-6 md:p-8 space-y-6">
        <div className="flex items-center space-x-3 border-b border-slate-800 pb-4">
          <div className="w-9 h-9 rounded-xl bg-slate-800/80 border border-slate-700/80 flex items-center justify-center text-teal-400">
            <HelpCircle className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-base font-bold text-slate-100">
              Plain-English Space Weather Glossary
            </h3>
            <p className="text-xs text-slate-400 font-mono">
              Reference guide for scientific terminology used across this console
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs font-sans text-slate-300">
          <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800/80 space-y-1">
            <strong className="text-slate-100 text-sm">M-Class Solar Flare:</strong>
            <p className="text-slate-400 leading-relaxed">
              Medium-to-large solar flares (peak soft X-ray flux between 10⁻⁵ and 10⁻⁴ W/m²). They can cause brief radio blackouts at Earth's polar latitudes and minor radiation storms.
            </p>
          </div>

          <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800/80 space-y-1">
            <strong className="text-slate-100 text-sm">Observation Time (T0):</strong>
            <p className="text-slate-400 leading-relaxed">
              The focal point in time. The forecasting model examines telemetry from the preceding 24 hours (T-23h to T0) to forecast whether an M-class flare will erupt in the subsequent 24 hours.
            </p>
          </div>

          <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800/80 space-y-1">
            <strong className="text-slate-100 text-sm">SDO / HMI SHARP Parameters:</strong>
            <p className="text-slate-400 leading-relaxed">
              Space-weather HMI Active Region Patches from NASA's Solar Dynamics Observatory. These measure physical magnetic field parameters on the solar surface like total magnetic flux and current density.
            </p>
          </div>

          <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800/80 space-y-1">
            <strong className="text-slate-100 text-sm">Proton Flux Units (pfu):</strong>
            <p className="text-slate-400 leading-relaxed">
              Standard measurement of high-energy solar proton density (particles / cm² · s · sr). Protons exceeding 10 MeV pose radiation risks to high-altitude airline passengers and satellites.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
