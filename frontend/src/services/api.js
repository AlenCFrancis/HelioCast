/**
 * Scientific API client wrapper for the Solar Flare Forecasting backend.
 * Reads base URL dynamically from VITE_API_BASE_URL or falls back to relative /api/v1.
 */

const BASE_URL = (import.meta.env.VITE_API_BASE_URL || '/api/v1').replace(/\/+$/, '');

async function handleResponse(response) {
  if (!response.ok) {
    let errorDetail = `HTTP Error ${response.status}: ${response.statusText}`;
    try {
      const errorJson = await response.json();
      if (errorJson && errorJson.detail) {
        errorDetail = errorJson.detail;
      }
    } catch (_) {
      // Ignore json parse error on non-json error responses
    }
    throw new Error(errorDetail);
  }
  return response.json();
}

export const apiService = {
  /**
   * Health and service readiness check.
   */
  async checkHealth() {
    const res = await fetch(`${BASE_URL}/health`);
    return handleResponse(res);
  },

  /**
   * Fetch dataset metadata and operational observation parameters.
   */
  async fetchDatasetSummary() {
    const res = await fetch(`${BASE_URL}/summary`);
    return handleResponse(res);
  },

  /**
   * Fetch available model architectures and threshold configurations.
   */
  async fetchModels() {
    const res = await fetch(`${BASE_URL}/models`);
    return handleResponse(res);
  },

  /**
   * Fetch offline test-set validation benchmark matrix.
   */
  async fetchMetrics() {
    const res = await fetch(`${BASE_URL}/metrics`);
    return handleResponse(res);
  },

  /**
   * Fetch list of available observation timestamps with complete 24h history.
   */
  async fetchAvailableTimestamps() {
    const res = await fetch(`${BASE_URL}/timestamps`);
    return handleResponse(res);
  },

  /**
   * Fetch continuous 24-hour sequence of SHARP telemetry for a target timestamp.
   * @param {string|null} timestamp - UTC timestamp string (YYYY-MM-DD HH:MM:SS) or null for latest.
   */
  async fetchTelemetryWindow(timestamp = null) {
    const query = timestamp ? `?timestamp=${encodeURIComponent(timestamp)}` : '';
    const res = await fetch(`${BASE_URL}/telemetry${query}`);
    return handleResponse(res);
  },

  /**
   * Execute real-time inference on a selected model and target timestamp.
   * @param {Object} params
   * @param {string} params.model_name
   * @param {string|null} params.timestamp
   * @param {Array|null} params.custom_window
   */
  async runPrediction({ model_name, timestamp = null, custom_window = null }) {
    const res = await fetch(`${BASE_URL}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model_name,
        timestamp,
        custom_window,
      }),
    });
    return handleResponse(res);
  },

  /**
   * Compare all 4 model architectures side-by-side for a target timestamp.
   * @param {string|null} timestamp
   */
  async compareAllModels(timestamp = null) {
    const modelKeys = ['lstm', 'random_forest', 'xgboost', 'logistic_regression'];
    const promises = modelKeys.map(key => 
      this.runPrediction({ model_name: key, timestamp })
        .then(result => ({ status: 'fulfilled', modelKey: key, data: result }))
        .catch(err => ({ status: 'rejected', modelKey: key, error: err.message }))
    );
    const outcomes = await Promise.all(promises);
    return outcomes;
  }
};

export default apiService;
