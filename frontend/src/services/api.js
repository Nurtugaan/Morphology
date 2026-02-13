/**
 * API client for Morphology Analyzer backend
 */

const API_BASE = '/api';
const TIMEOUT_MS = 10000;

/**
 * Fetch with timeout
 */
async function fetchWithTimeout(url, options = {}) {
    const controller = new AbortController();
    const id = setTimeout(() => controller.abort(), TIMEOUT_MS);
    try {
        const response = await fetch(url, { ...options, signal: controller.signal });
        return response;
    } catch (err) {
        if (err.name === 'AbortError') {
            throw new Error('Request timed out. Is the backend running?');
        }
        throw err;
    } finally {
        clearTimeout(id);
    }
}

/**
 * Fetch available models
 * @returns {Promise<{models: Array, default_model: string}>}
 */
export async function getModels() {
    const response = await fetchWithTimeout(`${API_BASE}/models`);
    if (!response.ok) {
        throw new Error(`Failed to fetch models: ${response.statusText}`);
    }
    return response.json();
}

/**
 * Analyze text with specified model
 * @param {string} text - Text to analyze
 * @param {string} modelName - Model to use
 * @returns {Promise<{model_name: string, text: string, tokens: Array, processing_time_ms: number}>}
 */
export async function analyzeText(text, modelName) {
    const response = await fetchWithTimeout(`${API_BASE}/analyze`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            text: text,
            model_name: modelName
        }),
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(error.detail || 'Analysis failed');
    }

    return response.json();
}
