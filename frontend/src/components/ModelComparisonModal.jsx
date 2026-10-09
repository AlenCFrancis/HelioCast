import React from 'react';
import { X, CheckCircle, AlertTriangle, ShieldCheck, ShieldAlert, Cpu, Sparkles } from 'lucide-react';

export default function ModelComparisonModal({
  isOpen,
  onClose,
  comparisonData,
  loading,
  targetTimestamp,
  onSelectModel,
}) {
  if (!isOpen) return null;

  // Extract successful models
  const results = (comparisonData || [])
    .filter((item) => item.status === 'fulfilled' && item.data)
    .map((item) => item.data);

  // Compute consensus
  const total = results.length;
  const flareVotes = results.filter((r) => r.is_flare).length;
  const isConsensusFlare = flareVotes >= Math.ceil(total / 2);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm animate-fadeIn">
      <div 
        className="glass-panel w-full max-w-4xl rounded-2xl border border-slate-700/80 shadow-2xl p-6 sm:p-8 space-y-6 overflow-hidden max-h-[90vh] flex flex-col"
        role="dialog"
        aria-modal="true"
        aria-labelledby="comparison-title"
      >
        {/* Header */}
        <div className="flex items-center justify-between border-b border-slate-800 pb-4">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 rounded-xl bg-slate-800/80 border border-slate-700 flex items-center justify-center text-teal-400">
              <Cpu className="w-5 h-5" />
            </div>
            <div>
              <h2 id="comparison-title" className="text-lg font-bold text-slate-100">
                Multi-Model Consensus Comparison
              </h2>
              <p className="text-xs text-slate-400 font-mono">
                Target: {targetTimestamp || 'Latest Observation'} UTC
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="p-2 rounded-lg bg-slate-800/80 hover:bg-slate-700 text-slate-400 hover:text-slate-100 transition-colors"
            aria-label="Close dialog"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Loading state */}
        {loading ? (
          <div className="py-16 text-center space-y-3">
            <div className="w-8 h-8 border-2 border-teal-400 border-t-transparent rounded-full animate-spin mx-auto" />
            <p className="text-sm text-slate-400 font-mono">Running zero-leakage inference across all 4 architectures...</p>
          </div>
        ) : (
          <div className="space-y-6 overflow-y-auto pr-1">
            {/* Consensus Banner */}
            <div className={`p-4 rounded-xl border flex items-center justify-between gap-4 ${
              isConsensusFlare 
                ? 'bg-rose-950/40 border-rose-800/80 text-rose-200' 
                : 'bg-emerald-950/40 border-emerald-800/80 text-emerald-200'
            }`}>
              <div className="flex items-center space-x-3">
                {isConsensusFlare ? (
                  <ShieldAlert className="w-6 h-6 text-rose-400 shrink-0" />
                ) : (
                  <ShieldCheck className="w-6 h-6 text-emerald-400 shrink-0" />
                )}
                <div>
                  <div className="text-sm font-bold">
                    {isConsensusFlare 
                      ? `Consensus Warning: ${flareVotes} of ${total} models predict an M-class or larger flare`
                      : `Consensus Nominal: ${total - flareVotes} of ${total} models predict low hazard`}
                  </div>
                  <p className="text-xs text-slate-300 mt-0.5">
                    {isConsensusFlare 
                      ? 'Elevated magnetic activity observed. Satellite protective measures recommended.'
                      : 'All or majority of architectures agree that space-weather radiation risk remains nominal.'}
                  </p>
                </div>
              </div>
            </div>

            {/* 4 Models Grid */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              {results.map((pred) => {
                const isCrit = pred.is_flare;
                return (
                  <div
                    key={pred.model_name}
                    className={`p-4 rounded-xl border transition-all flex flex-col justify-between ${
                      isCrit
                        ? 'bg-rose-950/20 border-rose-800/60'
                        : 'bg-slate-900/60 border-slate-800'
                    }`}
                  >
                    <div>
                      <div className="flex items-center justify-between text-xs text-slate-400 mb-2">
                        <span className="font-semibold text-slate-200">{pred.model_display_name}</span>
                      </div>
                      
                      <div className={`text-3xl font-bold font-mono-numbers my-1 ${
                        isCrit ? 'text-rose-400' : 'text-emerald-400'
                      }`}>
                        {pred.probability_formatted}
                      </div>

                      <div className="text-xs text-slate-400 font-mono mt-1">
                        Threshold: <span className="text-amber-400 font-semibold">{((pred.threshold || 0.5) * 100).toFixed(1)}%</span>
                      </div>

                      <div className={`mt-3 inline-flex items-center space-x-1.5 px-2.5 py-1 rounded-md text-[11px] font-medium ${
                        isCrit 
                          ? 'bg-rose-900/40 text-rose-300 border border-rose-800'
                          : 'bg-emerald-900/40 text-emerald-300 border border-emerald-800'
                      }`}>
                        {isCrit ? <AlertTriangle className="w-3 h-3" /> : <CheckCircle className="w-3 h-3" />}
                        <span>{isCrit ? 'Flare Likely' : 'Unlikely'}</span>
                      </div>
                    </div>

                    <button
                      type="button"
                      onClick={() => {
                        onSelectModel(pred.model_name);
                        onClose();
                      }}
                      className="mt-4 w-full py-1.5 px-2.5 text-xs rounded-md bg-slate-800/80 hover:bg-slate-700 text-slate-200 border border-slate-700/80 transition-colors cursor-pointer"
                    >
                      Use this model
                    </button>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Footer */}
        <div className="pt-4 border-t border-slate-800 flex justify-end">
          <button
            type="button"
            onClick={onClose}
            className="px-5 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium transition-colors cursor-pointer"
          >
            Close Comparison
          </button>
        </div>
      </div>
    </div>
  );
}
