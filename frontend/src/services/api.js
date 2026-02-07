/**
 * API client for Morphology Analyzer backend
 */

const API_BASE = '/api';

/**
 * Fetch available models
 * @returns {Promise<{models: Array, default_model: string}>}
 */
export async function getModels() {
    const response = await fetch(`${API_BASE}/models`);
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
    const response = await fetch(`${API_BASE}/analyze`, {
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
