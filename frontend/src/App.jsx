import React, { useState, useEffect } from 'react';
import ModelSelector from './components/ModelSelector';
import TextInput from './components/TextInput';
import AnalysisResult from './components/AnalysisResult';
import { getModels, analyzeText } from './services/api';

/**
 * Main Application Component
 */
export default function App() {
    // State
    const [models, setModels] = useState([]);
    const [selectedModel, setSelectedModel] = useState('distilbert');
    const [text, setText] = useState('');
    const [result, setResult] = useState(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);
    const [isInitializing, setIsInitializing] = useState(true);

    // Load available models on mount
    useEffect(() => {
        async function loadModels() {
            try {
                const data = await getModels();
                setModels(data.models);
                setSelectedModel(data.default_model);
            } catch (err) {
                setError(`Не удалось загрузить модели: ${err.message}`);
            } finally {
                setIsInitializing(false);
            }
        }
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

    // Show loading during initialization
    if (isInitializing) {
        return (
            <div className="container">
                <div className="header">
                    <h1>Morphology Analyzer</h1>
                    <p>Загрузка...</p>
                </div>
            </div>
        );
    }

    return (
        <div className="container">
            <header className="header">
                <h1>Morphology Analyzer</h1>
                <p>Интерактивный морфологический анализ текста с использованием трансформеров</p>
            </header>

            <main>
                <div className="card">
                    <ModelSelector
                        models={models}
                        selectedModel={selectedModel}
                        onModelChange={setSelectedModel}
                        disabled={isLoading}
                    />

                    <TextInput
                        text={text}
                        onTextChange={setText}
                        onSubmit={handleAnalyze}
                        isLoading={isLoading}
                    />

                    {error && (
                        <div className="error-message">
                            ❌ {error}
                        </div>
                    )}
                </div>

                <div className="card">
                    <AnalysisResult result={result} />
                </div>
            </main>

            <footer style={{ textAlign: 'center', padding: '2rem 0', color: 'var(--text-muted)' }}>
                <p>Дипломный проект · Морфологический анализ · 2026</p>
            </footer>
        </div>
    );
}
