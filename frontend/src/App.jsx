import React, { useState, useEffect, useCallback, useRef } from 'react';
import { 
  Sun, 
  Activity, 
  BarChart3, 
  RefreshCw, 
  AlertTriangle, 
  X
} from 'lucide-react';

import apiService from './services/api';
import RiskBanner from './components/RiskBanner';
import FeatureDashboard from './components/FeatureDashboard';
import MetricsMatrix from './components/MetricsMatrix';

export default function App() {
  // Navigation State with localStorage persistence
  const [activeTab, setActiveTab] = useState(() => {
    return localStorage.getItem('heliocast_tab') || 'forecast';
  });

  // Controls State with localStorage persistence
  const [selectedModel, setSelectedModel] = useState(() => {
    return localStorage.getItem('heliocast_model') || 'lstm';
  });
  const [autoRun, setAutoRun] = useState(() => {
    const saved = localStorage.getItem('heliocast_autorun');
    return saved !== null ? saved === 'true' : true;
  });

  // Metadata & Timestamps
  const [summary, setSummary] = useState(null);
  const [models, setModels] = useState([]);
  const [metrics, setMetrics] = useState([]);
  const [timestamps, setTimestamps] = useState([]);
  const [selectedTimestamp, setSelectedTimestamp] = useState('');
  const [isLatest, setIsLatest] = useState(true);

  // Predictions & Telemetry Data
  const [prediction, setPrediction] = useState(null);
  const [telemetry, setTelemetry] = useState(null);

  // Status & Health
  const [backendHealthy, setBackendHealthy] = useState(true);
  const [initialLoading, setInitialLoading] = useState(true);
  const [inferring, setInferring] = useState(false);
  const [toastMessage, setToastMessage] = useState(null);

  // Ref to prevent initial duplicate trigger in auto-run
  const isFirstRender = useRef(true);

  // Sync to localStorage
  useEffect(() => {
    localStorage.setItem('heliocast_tab', activeTab);
  }, [activeTab]);

  useEffect(() => {
    localStorage.setItem('heliocast_model', selectedModel);
  }, [selectedModel]);

  useEffect(() => {
    localStorage.setItem('heliocast_autorun', String(autoRun));
  }, [autoRun]);

  // Check health periodically
  const verifyHealth = useCallback(async () => {
    try {
      const res = await apiService.checkHealth();
      setBackendHealthy(res.status === 'healthy');
    } catch {
      setBackendHealthy(false);
    }
  }, []);

  // Primary inference runner
  const executeInference = useCallback(async (modelId, timestampStr) => {
    setInferring(true);
    setToastMessage(null);
    try {
      const [predRes, teleRes] = await Promise.all([
        apiService.runPrediction({
          model_name: modelId,
          timestamp: timestampStr,
        }),
        apiService.fetchTelemetryWindow(timestampStr),
      ]);
      setPrediction(predRes);
      setTelemetry(teleRes);
      setBackendHealthy(true);
    } catch (err) {
      console.error('Inference error:', err);
      setToastMessage({
        type: 'error',
        title: 'Inference Unsuccessful',
        text: err.message || 'Failed to execute model forecast.',
      });
      setBackendHealthy(false);
    } finally {
      setInferring(false);
    }
  }, []);

  // Initial load
  const loadInitialData = useCallback(async () => {
    setInitialLoading(true);
    setToastMessage(null);
    try {
      await verifyHealth();
      const [sumRes, modelRes, metricsRes, timeRes] = await Promise.all([
        apiService.fetchDatasetSummary(),
        apiService.fetchModels(),
        apiService.fetchMetrics(),
        apiService.fetchAvailableTimestamps(),
      ]);

      setSummary(sumRes);
      setModels(modelRes.models || []);
      setMetrics(metricsRes.models || []);
      
      const tsList = timeRes.timestamps || [];
      setTimestamps(tsList);

      const latestTs = timeRes.latest || tsList[tsList.length - 1];
      setSelectedTimestamp(latestTs);

      // Verify active model exists in models list
      const modelKeys = (modelRes.models || []).map(m => m.id);
      const activeM = modelKeys.includes(selectedModel) ? selectedModel : (modelKeys[0] || 'lstm');
      setSelectedModel(activeM);

      // Run initial forecast
      if (latestTs) {
        await executeInference(activeM, latestTs);
      }
    } catch (err) {
      console.error('Initialization error:', err);
      setToastMessage({
        type: 'error',
        title: 'Backend Connection Failed',
        text: `${err.message}. Please verify the FastAPI service is running.`,
      });
      setBackendHealthy(false);
    } finally {
      setInitialLoading(false);
      isFirstRender.current = false;
    }
  }, [verifyHealth, selectedModel, executeInference]);

  useEffect(() => {
    loadInitialData();
  }, [loadInitialData]);

  // Auto-run debouncer when model or timestamp changes
  useEffect(() => {
    if (isFirstRender.current || initialLoading) return;
    if (!autoRun) return;

    const handler = setTimeout(() => {
      const targetTs = isLatest ? (timestamps[timestamps.length - 1] || null) : selectedTimestamp;
      if (selectedModel && targetTs) {
        executeInference(selectedModel, targetTs);
      }
    }, 350);

    return () => clearTimeout(handler);
  }, [selectedModel, selectedTimestamp, isLatest, autoRun, executeInference, timestamps, initialLoading]);

  // Manual run button handler
  const handleManualRun = () => {
    const targetTs = isLatest ? (timestamps[timestamps.length - 1] || null) : selectedTimestamp;
    executeInference(selectedModel, targetTs);
  };

  return (
    <div className="h-screen w-screen overflow-hidden text-slate-100 flex flex-col font-sans selection:bg-teal-900 selection:text-white bg-[#070b12]">
      {/* =========================================================================
          COMPACT LANDSCAPE UNIFIED HEADER BAR
      ========================================================================= */}
      <header className="glass-header h-13 shrink-0 px-4 lg:px-6 flex items-center justify-between gap-4 z-40 border-b border-slate-800/80">
        {/* Left: App Title and Live Status Dot */}
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 rounded-lg bg-slate-800/90 border border-slate-700/80 flex items-center justify-center text-amber-400 shadow-xs">
            <Sun className="w-4 h-4" />
          </div>
          <div className="flex items-center space-x-2">
            <span className="text-sm font-bold tracking-tight text-slate-100 uppercase font-mono-numbers">
              HelioCast
            </span>
            <span className="text-xs font-mono text-slate-400 hidden xl:inline">
              Solar Flare Forecaster
            </span>
            {/* Live Status Dot */}
            <div 
              className={`flex items-center space-x-1.5 px-2 py-0.5 rounded-full text-[10px] font-mono border ${
                backendHealthy 
                  ? 'bg-emerald-950/40 text-emerald-300 border-emerald-800/60' 
                  : 'bg-rose-950/40 text-rose-300 border-rose-800/60'
              }`}
              title={backendHealthy ? 'Backend API connection active' : 'Backend API connection offline'}
            >
              <span className={`w-1.5 h-1.5 rounded-full ${backendHealthy ? 'bg-emerald-400 animate-pulse' : 'bg-rose-400'}`} />
              <span className="hidden sm:inline">{backendHealthy ? 'Live Stream' : 'Offline'}</span>
            </div>
          </div>
        </div>

        {/* Center: Tabs in Header Bar */}
        <nav className="flex items-center space-x-1.5" aria-label="Tabs">
          <button
            type="button"
            onClick={() => setActiveTab('forecast')}
            className={`px-3 py-1.5 text-xs font-medium rounded-lg border transition-all flex items-center space-x-1.5 cursor-pointer ${
              activeTab === 'forecast' ? 'glass-tab-active' : 'glass-tab-inactive'
            }`}
          >
            <Activity className="w-3.5 h-3.5 text-teal-400" />
            <span>Forecast</span>
          </button>

          <button
            type="button"
            onClick={() => setActiveTab('benchmarks')}
            className={`px-3 py-1.5 text-xs font-medium rounded-lg border transition-all flex items-center space-x-1.5 cursor-pointer ${
              activeTab === 'benchmarks' ? 'glass-tab-active' : 'glass-tab-inactive'
            }`}
          >
            <BarChart3 className="w-3.5 h-3.5 text-amber-400" />
            <span>Model Benchmarks</span>
          </button>

        </nav>

        {/* Right: Quick Action & Refresh */}
        <div className="flex items-center space-x-2">

          <button
            type="button"
            onClick={loadInitialData}
            title="Refresh telemetry"
            className="p-1.5 rounded-md bg-slate-800/70 hover:bg-slate-700 text-slate-300 border border-slate-700 transition-colors cursor-pointer"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${initialLoading ? 'animate-spin' : ''}`} />
          </button>
        </div>
      </header>

      {/* =========================================================================
          MAIN LANDSCAPE CONTAINER (PERFECT FIT AT 100% ZOOM)
      ========================================================================= */}
      <main className="flex-1 min-h-0 w-full px-4 lg:px-6 py-3 lg:py-3.5 overflow-hidden flex flex-col">
        {/* Floating Toast Notification on error */}
        {toastMessage && (
          <div className="mb-2 p-3 rounded-lg bg-rose-950/80 border border-rose-800 text-rose-200 text-xs flex items-center justify-between shadow-xl backdrop-blur-md animate-fadeIn shrink-0">
            <div className="flex items-center space-x-2.5">
              <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0" />
              <div>
                <strong className="text-slate-100">{toastMessage.title}: </strong>
                <span>{toastMessage.text}</span>
              </div>
            </div>
            <div className="flex items-center space-x-2">
              <button
                type="button"
                onClick={loadInitialData}
                className="px-2.5 py-0.5 rounded bg-rose-900/60 hover:bg-rose-800 text-rose-100 border border-rose-700 text-xs cursor-pointer"
              >
                Retry
              </button>
              <button
                type="button"
                onClick={() => setToastMessage(null)}
                className="p-0.5 text-rose-400 hover:text-white cursor-pointer"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>
        )}

        {/* TAB 1: OPERATIONAL FORECAST (RISK BANNER + ALL-FEATURE BAR GRID, VIEWPORT FIT) */}
        {activeTab === 'forecast' && (
          <div className="h-full min-h-0 flex flex-col gap-3 overflow-y-auto lg:overflow-hidden animate-fadeIn">
            {initialLoading ? (
              <>
                <div className="h-24 shrink-0 rounded-xl border border-slate-800 bg-slate-900/40 animate-pulse" />
                <div className="flex-1 min-h-0 grid grid-cols-2 lg:grid-cols-4 lg:grid-rows-3 gap-2.5">
                  {Array.from({ length: 12 }).map((_, i) => (
                    <div key={i} className="min-h-[160px] lg:min-h-0 rounded-xl border border-slate-800 bg-slate-900/40 animate-pulse" />
                  ))}
                </div>
              </>
            ) : (
              <>
                <RiskBanner
                  models={models}
                  timestamps={timestamps}
                  selectedModel={selectedModel}
                  setSelectedModel={setSelectedModel}
                  selectedTimestamp={selectedTimestamp}
                  setSelectedTimestamp={setSelectedTimestamp}
                  isLatest={isLatest}
                  setIsLatest={setIsLatest}
                  prediction={prediction}
                  loading={inferring}
                  onExecuteInference={handleManualRun}
                  autoRun={autoRun}
                  setAutoRun={setAutoRun}
                />
                <FeatureDashboard
                  telemetry={telemetry}
                  featureNames={summary?.feature_names || []}
                />
              </>
            )}
          </div>
        )}

        {/* TAB 2: MODEL BENCHMARKS (Full Width Scrollable View) */}
        {activeTab === 'benchmarks' && (
          <div className="h-full overflow-y-auto pr-1 pb-4 animate-fadeIn">
            <MetricsMatrix metrics={metrics} />
          </div>
        )}
      </main>

      {/* =========================================================================
          SLIM ENGINEERING STATUS FOOTER (LANDSCAPE ZERO-OVERFLOW)
      ========================================================================= */}
      <footer className="h-7 shrink-0 border-t border-slate-800/80 bg-slate-950/90 px-4 lg:px-6 flex items-center justify-between text-[11px] font-mono text-slate-500">
        <div>
          HelioCast Space Weather Mission Operations Console
        </div>
        <div className="flex items-center space-x-4">
          <span>Cadence: 1.0h</span>
          <span>Forecast Horizon: 24h</span>
          <span className="hidden sm:inline">Zero Lookahead Leakage</span>
        </div>
      </footer>
    </div>
  );
}
