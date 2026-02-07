import React from 'react';

/**
 * AnalysisResult - Table displaying morphological analysis results
 */
export default function AnalysisResult({ result }) {
    if (!result) {
        return (
            <div className="empty-state">
                <div className="empty-state-icon">📝</div>
                <p>Введите текст и нажмите "Анализировать" для получения результатов</p>
            </div>
        );
    }

    const { model_name, text, tokens, processing_time_ms } = result;

    /**
     * Get CSS class for POS tag
     */
    const getPosClass = (upos) => {
        const knownTags = ['NOUN', 'VERB', 'ADJ', 'ADV', 'DET', 'PRON', 'ADP', 'PUNCT', 'AUX', 'CCONJ', 'SCONJ'];
        return knownTags.includes(upos) ? `pos-${upos}` : 'pos-default';
    };

    /**
     * Format features object as badges
     */
    const renderFeatures = (features) => {
        if (!features || Object.keys(features).length === 0) {
            return <span style={{ color: 'var(--text-muted)' }}>—</span>;
        }

        return (
            <div className="features">
                {Object.entries(features).map(([key, value]) => (
                    <span key={key} className="feature-badge">
                        {key}={value}
                    </span>
                ))}
            </div>
        );
    };

    return (
        <div className="results-section">
            <div className="results-header">
                <h2>Результаты анализа</h2>
                <div className="results-meta">
                    <span className="model-badge">{model_name}</span>
                    <span style={{ marginLeft: '1rem' }}>
                        ⏱ {processing_time_ms.toFixed(1)} ms
                    </span>
                </div>
            </div>

            <table className="results-table">
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Токен</th>
                        <th>POS</th>
                        <th>Признаки</th>
                        <th>Полная метка</th>
                    </tr>
                </thead>
                <tbody>
                    {tokens.map((token, idx) => (
                        <tr key={idx}>
                            <td style={{ color: 'var(--text-muted)' }}>{idx + 1}</td>
                            <td style={{ fontWeight: 500 }}>{token.token}</td>
                            <td>
                                <span className={`pos-tag ${getPosClass(token.upos)}`}>
                                    {token.upos}
                                </span>
                            </td>
                            <td>{renderFeatures(token.features)}</td>
                            <td style={{ fontFamily: 'monospace', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                                {token.label}
                            </td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
}
