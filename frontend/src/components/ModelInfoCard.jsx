import React, { useState } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { Cpu, Clock, BarChart3, Target, Crosshair, Repeat, ChevronDown, Info } from 'lucide-react';

/**
 * ModelInfoCard - Displays evaluation metrics for the selected model
 * Shows between header and input bar with animation on model change
 * Includes collapsible "Additional Information" panel with training hyperparameters
 */
export default function ModelInfoCard({ model }) {
    const [showDetails, setShowDetails] = useState(false);

    if (!model) return null;

    const metrics = [
        { label: 'Accuracy', value: model.accuracy, icon: Target, format: 'percent', color: '#4ade80' },
        { label: 'Macro-F1', value: model.f1, icon: BarChart3, format: 'percent', color: '#0ea5e9' },
        { label: 'Precision', value: model.precision, icon: Crosshair, format: 'percent', color: '#a78bfa' },
        { label: 'Recall', value: model.recall, icon: Repeat, format: 'percent', color: '#facc15' },
        { label: 'Parameters', value: model.parameters, icon: Cpu, format: 'text', color: '#f472b6' },
        { label: 'Train Time', value: model.training_time, icon: Clock, format: 'text', color: '#fb923c' },
    ];

    const formatValue = (value, format) => {
        if (value == null) return '—';
        if (format === 'percent') return (value * 100).toFixed(2) + '%';
        return value;
    };

    const config = model.training_config;

    // Training hyperparameters to display
    const hyperparams = config ? [
        { label: 'Epochs', value: config.epochs },
        { label: 'Learning Rate', value: config.learning_rate },
        { label: 'Batch Size', value: config.batch_size },
        { label: 'Max Length', value: config.max_length },
        { label: 'Warmup Ratio', value: config.warmup_ratio },
        { label: 'Weight Decay', value: config.weight_decay },
        { label: 'Dropout', value: config.dropout },
        { label: 'Dataset', value: config.dataset, wide: true },
    ] : [];

    return (
        <AnimatePresence mode="wait">
            <motion.div
                key={model.id}
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 10 }}
                transition={{ duration: 0.3 }}
                className="model-info-card"
            >
                {/* Model Header */}
                <div className="model-info-header">
                    <div>
                        <h2 className="model-info-name">{model.name}</h2>
                        <p className="model-info-description">{model.description}</p>
                    </div>
                    {model.architecture && (
                        <span className="model-info-arch">{model.architecture}</span>
                    )}
                </div>

                {/* Metrics Grid */}
                <div className="model-info-metrics">
                    {metrics.map((metric) => {
                        const Icon = metric.icon;
                        return (
                            <div key={metric.label} className="metric-item">
                                <div className="metric-icon-row">
                                    <Icon style={{ width: '0.875rem', height: '0.875rem', color: metric.color }} />
                                    <span className="metric-label">{metric.label}</span>
                                </div>
                                <span className="metric-value" style={{ color: metric.color }}>
                                    {formatValue(metric.value, metric.format)}
                                </span>
                            </div>
                        );
                    })}
                </div>

                {/* Additional Information Toggle */}
                {config && (
                    <>
                        <button
                            className="details-toggle"
                            onClick={() => setShowDetails(!showDetails)}
                        >
                            <Info style={{ width: '0.875rem', height: '0.875rem' }} />
                            <span>Training Hyperparameters</span>
                            <ChevronDown
                                className={`details-chevron ${showDetails ? 'details-chevron--open' : ''}`}
                            />
                        </button>

                        <AnimatePresence>
                            {showDetails && (
                                <motion.div
                                    initial={{ height: 0, opacity: 0 }}
                                    animate={{ height: 'auto', opacity: 1 }}
                                    exit={{ height: 0, opacity: 0 }}
                                    transition={{ duration: 0.25 }}
                                    style={{ overflow: 'hidden' }}
                                >
                                    <div className="details-panel">
                                        {hyperparams.map((param) => (
                                            <div
                                                key={param.label}
                                                className={`detail-row ${param.wide ? 'detail-row--wide' : ''}`}
                                            >
                                                <span className="detail-label">{param.label}</span>
                                                <span className="detail-value">
                                                    {param.value != null ? String(param.value) : '—'}
                                                </span>
                                            </div>
                                        ))}
                                    </div>
                                </motion.div>
                            )}
                        </AnimatePresence>
                    </>
                )}
            </motion.div>
        </AnimatePresence>
    );
}
