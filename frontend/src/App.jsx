import React, { useState, useEffect } from 'react';
import { Sparkles, ChevronDown, ArrowUp, Loader2 } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import AnalysisResult from './components/AnalysisResult';
import ModelInfoCard from './components/ModelInfoCard';
import { getModels, analyzeText } from './services/api';

/**
 * Main Application Component
 */
export default function App() {
    // State — полностью сохранена оригинальная логика
    const [models, setModels] = useState([]);
    const [selectedModel, setSelectedModel] = useState('distilbert');
    const [text, setText] = useState('');
    const [result, setResult] = useState(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);
    const [isInitializing, setIsInitializing] = useState(true);

    // Load available models
    const loadModels = async () => {
        setIsInitializing(true);
        setError(null);
        try {
            const data = await getModels();
            setModels(data.models);
            setSelectedModel(data.default_model);
        } catch (err) {
            setError(`Failed to load models: ${err.message}`);
        } finally {
            setIsInitializing(false);
        }
    };

    // Load on mount
    useEffect(() => {
        loadModels();
    }, []);

    // Handle analysis submission
    const handleAnalyze = async () => {
        if (!text.trim()) return;

        setIsLoading(true);
        setError(null);

        try {
            const analysisResult = await analyzeText(text, selectedModel);
            setResult(analysisResult);

            // Refresh models to update "loaded" status
            const modelsData = await getModels();
            setModels(modelsData.models);
        } catch (err) {
            setError(err.message);
            setResult(null);
        } finally {
            setIsLoading(false);
        }
    };

    // Submit on Enter (without Shift)
    const handleKeyDown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleAnalyze();
        }
    };

    // Auto-resize textarea
    const handleTextareaInput = (e) => {
        const target = e.target;
        target.style.height = 'auto';
        target.style.height = Math.min(target.scrollHeight, 120) + 'px';
    };

    // Show loading / error during initialization
    if (isInitializing) {
        return (
            <div className="init-screen">
                <Sparkles className="header-icon" style={{ width: '3rem', height: '3rem' }} />
                <h1 className="header-title">Morphology Analyzer</h1>
                <Loader2 className="init-spinner" />
                <p className="init-text">Loading models...</p>
            </div>
        );
    }

    if (models.length === 0 && error) {
        return (
            <div className="init-screen">
                <Sparkles className="header-icon" style={{ width: '3rem', height: '3rem' }} />
                <h1 className="header-title">Morphology Analyzer</h1>
                <p className="init-text" style={{ color: '#fca5a5' }}>❌ {error}</p>
                <button className="send-button" style={{ width: 'auto', borderRadius: '9999px', padding: '0.5rem 1.5rem', fontSize: '0.875rem' }} onClick={loadModels}>
                    Retry
                </button>
            </div>
        );
    }

    return (
        <div style={{ minHeight: '100vh' }}>
            {/* Header */}
            <header className="app-header">
                <div className="header-inner">
                    <div className="header-title-row">
                        <Sparkles className="header-icon" />
                        <h1 className="header-title">Morphology Analyzer</h1>
                    </div>
                    <div className="header-glow-line" />
                </div>
            </header>

            <main className="app-main">
                {/* Model Info Card */}
                <ModelInfoCard model={models.find(m => m.id === selectedModel)} />

                {/* Input Bar — ChatGPT-style pill */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="input-bar"
                >
                    <div className="input-bar-inner">
                        {/* Model Selector */}
                        <div className="model-select-wrapper">
                            <select
                                value={selectedModel}
                                onChange={(e) => setSelectedModel(e.target.value)}
                                className="model-select"
                                disabled={isLoading}
                            >
                                {models.map((model) => (
                                    <option key={model.id} value={model.id}>
                                        {model.name}{model.loaded ? ' ✓' : ''}
                                    </option>
                                ))}
                            </select>
                            <ChevronDown className="model-select-chevron" />
                        </div>

                        {/* Text Input */}
                        <textarea
                            value={text}
                            onChange={(e) => setText(e.target.value)}
                            onKeyDown={handleKeyDown}
                            onInput={handleTextareaInput}
                            placeholder="Enter English text for morphological analysis..."
                            rows={1}
                            className="input-textarea"
                            disabled={isLoading}
                        />

                        {/* Analyze Button */}
                        <button
                            onClick={handleAnalyze}
                            disabled={isLoading || !text.trim()}
                            className="send-button"
                        >
                            {isLoading ? (
                                <Loader2 className="spin" />
                            ) : (
                                <ArrowUp />
                            )}
                        </button>
                    </div>
                </motion.div>

                {/* Error */}
                {error && (
                    <div className="error-message">
                        ❌ {error}
                    </div>
                )}

                {/* Results */}
                <AnimatePresence mode="wait">
                    {result && (
                        <motion.div
                            key="results"
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -20 }}
                            transition={{ duration: 0.4 }}
                        >
                            <AnalysisResult result={result} />
                        </motion.div>
                    )}
                </AnimatePresence>

                {/* Empty State */}
                {!result && !isLoading && (
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="empty-state"
                    >
                        <Sparkles className="empty-state-icon" />
                        <p>Enter text and click analyze to begin morphological analysis</p>
                    </motion.div>
                )}
            </main>
        </div>
    );
}
