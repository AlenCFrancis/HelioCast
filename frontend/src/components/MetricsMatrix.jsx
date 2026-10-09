import React, { useState, useMemo } from 'react';
import { 
  Table, 
  BarChart3, 
  ChevronDown, 
  ChevronUp, 
  ArrowUpDown, 
  HelpCircle, 
  Sparkles,
  Award
} from 'lucide-react';

const METRIC_DEFINITIONS = {
  pr_auc: {
    label: 'PR-AUC',
    fullName: 'Precision-Recall Curve Area',
    tooltip: 'Primary metric for extreme class rarity (<5% positive events). Unlike ROC-AUC, PR-AUC directly penalizes false alarms.',
    higherIsBetter: true,
  },
  tss: {
    label: 'TSS',
    fullName: 'True Skill Statistic (Peirce Skill Score)',
    tooltip: 'TSS = Recall - False Positive Rate. Standard space-weather metric independent of flare rarity.',
    higherIsBetter: true,
  },
  roc_auc: {
    label: 'ROC-AUC',
    fullName: 'Receiver Operating Characteristic Area',
    tooltip: 'Measures model discrimination capability across all potential probability thresholds.',
    higherIsBetter: true,
  },
  f1_score: {
    label: 'F1 Score',
    fullName: 'Harmonic Mean of Precision & Recall',
    tooltip: 'Balances catching dangerous flares against avoiding costly false satellite alarms.',
    higherIsBetter: true,
  },
  recall: {
    label: 'Recall',
    fullName: 'Detection Rate (Sensitivity)',
    tooltip: 'Percentage of all actual M-class or larger flares successfully caught before eruption.',
    higherIsBetter: true,
  },
  precision: {
    label: 'Precision',
    fullName: 'Alert Reliability (PPV)',
    tooltip: 'Percentage of issued flare warnings that were followed by a real M+ flare.',
    higherIsBetter: true,
  },
  far: {
    label: 'FAR',
    fullName: 'False Alarm Rate',
    tooltip: 'Fraction of issued flare alarms that turned out to be false alarms. Lower is better.',
    higherIsBetter: false,
  },
  threshold: {
    label: 'Threshold',
    fullName: 'Frozen Validation Decision Threshold',
    tooltip: 'Decision boundary locked on the 15% validation split before evaluating the test set.',
    higherIsBetter: null,
  },
};

export default function MetricsMatrix({ metrics = [] }) {
  const [sortField, setSortField] = useState('tss');
  const [sortDirection, setSortDirection] = useState('desc'); // 'asc' | 'desc'
  const [expandedModel, setExpandedModel] = useState(null);
  const [selectedChartMetric, setSelectedChartMetric] = useState('tss');

  // Find best value for each metric
  const bestValues = useMemo(() => {
    const bests = {};
    Object.keys(METRIC_DEFINITIONS).forEach((key) => {
      const def = METRIC_DEFINITIONS[key];
      if (def.higherIsBetter === null) return;
      const values = metrics.map((m) => m[key] || 0);
      if (values.length) {
        bests[key] = def.higherIsBetter ? Math.max(...values) : Math.min(...values);
      }
    });
    return bests;
  }, [metrics]);

  // Handle column sort
  const handleSort = (field) => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection(METRIC_DEFINITIONS[field]?.higherIsBetter === false ? 'asc' : 'desc');
    }
  };

  // Sorted metrics
  const sortedMetrics = useMemo(() => {
    return [...metrics].sort((a, b) => {
      const aVal = a[sortField] ?? 0;
      const bVal = b[sortField] ?? 0;
      if (sortDirection === 'asc') {
        return aVal > bVal ? 1 : -1;
      }
      return aVal < bVal ? 1 : -1;
    });
  }, [metrics, sortField, sortDirection]);

  // Chart data for selected metric
  const chartData = useMemo(() => {
    const def = METRIC_DEFINITIONS[selectedChartMetric] || METRIC_DEFINITIONS.tss;
    const maxVal = Math.max(...metrics.map((m) => m[selectedChartMetric] || 0), 1.0);
    return metrics.map((m) => {
      const val = m[selectedChartMetric] || 0;
      const isBest = val === bestValues[selectedChartMetric];
      return {
        key: m.model_key,
        name: m.model_name,
        value: val,
        percentage: (val / maxVal) * 100,
        isBest,
      };
    });
  }, [metrics, selectedChartMetric, bestValues]);

  return (
    <div className="space-y-8">
      {/* =========================================================================
          SECTION 1: INTERACTIVE BAR CHART COMPARISON
      ========================================================================= */}
      <div className="glass-panel rounded-2xl p-6 md:p-8 space-y-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-800 pb-5">
          <div className="flex items-center space-x-3">
            <div className="w-9 h-9 rounded-xl bg-slate-800/80 border border-slate-700/80 flex items-center justify-center text-amber-400">
              <BarChart3 className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-base font-bold text-slate-100">
                Visual Model Comparison
              </h3>
              <p className="text-xs text-slate-400 font-mono">
                Select an operational metric to compare architecture performance side-by-side
              </p>
            </div>
          </div>

          {/* Metric Selector Chips */}
          <div className="flex flex-wrap items-center gap-1.5">
            {['tss', 'pr_auc', 'f1_score', 'recall', 'precision', 'roc_auc'].map((metricKey) => {
              const def = METRIC_DEFINITIONS[metricKey];
              const isSelected = selectedChartMetric === metricKey;
              return (
                <button
                  key={metricKey}
                  type="button"
                  onClick={() => setSelectedChartMetric(metricKey)}
                  className={`px-3 py-1.5 rounded-lg text-xs font-mono font-medium transition-all cursor-pointer border ${
                    isSelected
                      ? 'bg-amber-950/60 text-amber-300 border-amber-500 shadow-md'
                      : 'bg-slate-900/60 text-slate-400 border-slate-800 hover:text-slate-200 hover:border-slate-700'
                  }`}
                >
                  {def.label}
                </button>
              );
            })}
          </div>
        </div>

        {/* Selected Metric Explanation */}
        <div className="p-3.5 rounded-xl bg-slate-950/60 border border-slate-800/80 text-xs text-slate-300 flex items-center space-x-2.5">
          <HelpCircle className="w-4 h-4 text-amber-400 shrink-0" />
          <div>
            <strong className="text-slate-100">{METRIC_DEFINITIONS[selectedChartMetric]?.fullName}:</strong>{' '}
            <span className="text-slate-400">{METRIC_DEFINITIONS[selectedChartMetric]?.tooltip}</span>
          </div>
        </div>

        {/* Visual Comparison Horizontal Bars */}
        <div className="space-y-4 pt-2">
          {chartData.map((item) => (
            <div key={item.key} className="space-y-1.5">
              <div className="flex items-center justify-between text-xs font-mono">
                <div className="flex items-center space-x-2">
                  <span className="font-semibold text-slate-200">{item.name}</span>
                  {item.isBest && (
                    <span className="px-2 py-0.5 rounded bg-amber-950/60 text-amber-300 border border-amber-600/70 text-[10px] font-bold flex items-center space-x-1">
                      <Award className="w-3 h-3" />
                      <span>Lead Benchmark</span>
                    </span>
                  )}
                </div>
                <span className="text-sm font-bold font-mono-numbers text-slate-100">
                  {item.value.toFixed(4)}
                </span>
              </div>

              {/* Progress bar */}
              <div className="w-full h-4 bg-slate-950 rounded-lg border border-slate-800 overflow-hidden relative shadow-inner">
                <div
                  className={`h-full rounded-md transition-all duration-500 ${
                    item.isBest ? 'bg-amber-500 shadow-lg shadow-amber-500/30' : 'bg-slate-600'
                  }`}
                  style={{ width: `${Math.min(100, Math.max(5, item.percentage))}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* =========================================================================
          SECTION 2: SORTABLE VALIDATION BENCHMARKS TABLE
      ========================================================================= */}
      <div className="glass-panel rounded-2xl p-6 md:p-8 space-y-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800 pb-5">
          <div className="flex items-center space-x-3">
            <div className="w-9 h-9 rounded-xl bg-slate-800/80 border border-slate-700/80 flex items-center justify-center text-teal-400">
              <Table className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-base font-bold text-slate-100">
                Offline Test-Set Evaluation Matrix
              </h3>
              <p className="text-xs text-slate-400 font-mono">
                Evaluated on 15% held-out chronological test split (N = 634 observation windows, 69 positive flares)
              </p>
            </div>
          </div>
          <div className="text-xs font-mono text-slate-400">
            Click column headers to sort
          </div>
        </div>

        {/* The Table */}
        <div className="overflow-x-auto rounded-xl border border-slate-800 bg-slate-950/60 shadow-inner">
          <table className="w-full text-left text-xs font-mono border-collapse">
            <thead>
              <tr className="border-b border-slate-800 bg-slate-900/90 text-slate-300 uppercase tracking-wider">
                <th className="py-3.5 px-4 font-semibold text-slate-200">Model</th>
                {Object.keys(METRIC_DEFINITIONS).map((colKey) => {
                  const def = METRIC_DEFINITIONS[colKey];
                  const isSorted = sortField === colKey;
                  return (
                    <th
                      key={colKey}
                      onClick={() => handleSort(colKey)}
                      className="py-3.5 px-3 font-semibold text-right cursor-pointer select-none hover:text-white transition-colors group"
                      title={def.tooltip}
                    >
                      <div className="flex items-center justify-end space-x-1">
                        <span>{def.label}</span>
                        <ArrowUpDown className={`w-3 h-3 ${isSorted ? 'text-teal-400' : 'text-slate-600 group-hover:text-slate-400'}`} />
                      </div>
                    </th>
                  );
                })}
                <th className="py-3.5 px-4 text-center">Matrix</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/70 text-slate-300">
              {sortedMetrics.map((row) => {
                const isExpanded = expandedModel === row.model_key;
                return (
                  <React.Fragment key={row.model_key}>
                    <tr className="hover:bg-slate-900/60 transition-colors font-mono-numbers">
                      <td className="py-4 px-4 font-semibold text-slate-100">
                        {row.model_name}
                      </td>

                      {Object.keys(METRIC_DEFINITIONS).map((colKey) => {
                        const val = row[colKey];
                        const isBest = val !== undefined && val === bestValues[colKey];
                        return (
                          <td
                            key={colKey}
                            className={`py-4 px-3 text-right ${
                              isBest 
                                ? 'text-amber-400 font-bold bg-amber-950/20' 
                                : 'text-slate-300'
                            }`}
                          >
                            {typeof val === 'number' ? val.toFixed(4) : val}
                          </td>
                        );
                      })}

                      <td className="py-4 px-4 text-center">
                        {row.confusion_matrix ? (
                          <button
                            type="button"
                            onClick={() => setExpandedModel(isExpanded ? null : row.model_key)}
                            className="px-2.5 py-1 text-xs rounded-md bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition-colors inline-flex items-center space-x-1 cursor-pointer"
                          >
                            <span>Matrix</span>
                            {isExpanded ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
                          </button>
                        ) : (
                          <span className="text-slate-600">N/A</span>
                        )}
                      </td>
                    </tr>

                    {/* Expanded Confusion Matrix Row */}
                    {isExpanded && row.confusion_matrix && (
                      <tr className="bg-slate-900/95">
                        <td colSpan={10} className="py-5 px-6 border-b border-slate-800">
                          <div className="flex flex-col md:flex-row items-center gap-6 justify-between">
                            <div className="text-xs font-mono text-slate-300 max-w-sm space-y-1">
                              <strong className="text-slate-100 text-sm">{row.model_name} Test Confusion Matrix:</strong>
                              <p className="text-slate-400 text-[11px]">
                                Evaluated on 634 test observation windows with frozen decision threshold {row.threshold}.
                              </p>
                            </div>
                            <div className="grid grid-cols-2 gap-3 text-center text-xs font-mono w-full md:w-auto">
                              <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 shadow-sm">
                                <span className="text-slate-400 block text-[10px]">TRUE NEGATIVES (TN)</span>
                                <span className="text-lg font-bold font-mono-numbers text-slate-200">{row.confusion_matrix.tn}</span>
                                <span className="text-[10px] text-slate-500 block">Quiet periods correct</span>
                              </div>
                              <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 shadow-sm">
                                <span className="text-slate-400 block text-[10px]">FALSE POSITIVES (FP)</span>
                                <span className="text-lg font-bold font-mono-numbers text-rose-400">{row.confusion_matrix.fp}</span>
                                <span className="text-[10px] text-slate-500 block">False alarms</span>
                              </div>
                              <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 shadow-sm">
                                <span className="text-slate-400 block text-[10px]">FALSE NEGATIVES (FN)</span>
                                <span className="text-lg font-bold font-mono-numbers text-rose-400">{row.confusion_matrix.fn}</span>
                                <span className="text-[10px] text-slate-500 block">Missed flares</span>
                              </div>
                              <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 shadow-sm">
                                <span className="text-slate-400 block text-[10px]">TRUE POSITIVES (TP)</span>
                                <span className="text-lg font-bold font-mono-numbers text-emerald-400">{row.confusion_matrix.tp}</span>
                                <span className="text-[10px] text-slate-500 block">Flares caught</span>
                              </div>
                            </div>
                          </div>
                        </td>
                      </tr>
                    )}
                  </React.Fragment>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
