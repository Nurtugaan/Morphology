import React from 'react';
import { Clock, Sparkles } from 'lucide-react';
import { motion } from 'motion/react';

/**
 * AnalysisResult - Animated table displaying morphological analysis results
 * Preserves the same API response format: { model_name, text, tokens, processing_time_ms }
 */
export default function AnalysisResult({ result }) {
    if (!result) {
        return null;
    }

    const { model_name, tokens, processing_time_ms } = result;

    /**
     * Get POS tag CSS class for colored pill badges
     */
    const getPosClass = (upos) => {
        const knownTags = ['NOUN', 'VERB', 'ADJ', 'ADV', 'DET', 'PRON', 'ADP', 'PUNCT', 'AUX', 'CCONJ', 'SCONJ'];
        return knownTags.includes(upos) ? `pos-${upos}` : 'pos-default';
    };

    /**
     * Render features as glass-style badges
     */
    const renderFeatures = (features) => {
        if (!features || Object.keys(features).length === 0) {
            return <span style={{ color: 'var(--text-muted)' }}>—</span>;
        }

        return (
            <div className="features-list">
                {Object.entries(features).map(([key, value]) => (
                    <span key={key} className="feature-badge">
                        {key}={value}
                    </span>
                ))}
            </div>
        );
    };

    return (
        <div>
            {/* Meta Information Badges */}
            <div className="result-meta">
                <div className="meta-badge meta-badge--time">
                    <Clock className="meta-icon" />
                    <span className="meta-label">Processing time:</span>
                    <span className="meta-value">{processing_time_ms.toFixed(1)}ms</span>
                </div>
                <div className="meta-badge meta-badge--model">
                    <Sparkles className="meta-icon" />
                    <span className="meta-label">Model:</span>
                    <span className="meta-value">{model_name}</span>
                </div>
            </div>

            {/* Results Table */}
            <div className="results-card">
                <div style={{ overflowX: 'auto' }}>
                    <table className="results-table">
                        <thead>
                            <tr>
                                <th>#</th>
                                <th>Token</th>
                                <th>POS Tag</th>
                                <th>Morphological Features</th>
                                <th>Full Label</th>
                            </tr>
                        </thead>
                        <tbody>
                            {tokens.map((token, idx) => (
                                <motion.tr
                                    key={idx}
                                    initial={{ opacity: 0, x: -20 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    transition={{ delay: idx * 0.05 }}
                                >
                                    <td style={{ color: 'var(--text-muted)' }}>{idx + 1}</td>
                                    <td>
                                        <span className="token-text">{token.token}</span>
                                    </td>
                                    <td>
                                        <span className={`pos-pill ${getPosClass(token.upos)}`}>
                                            {token.upos}
                                        </span>
                                    </td>
                                    <td>{renderFeatures(token.features)}</td>
                                    <td>
                                        <span className="label-text">{token.label}</span>
                                    </td>
                                </motion.tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    );
}
