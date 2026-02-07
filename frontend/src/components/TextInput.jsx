import React from 'react';

/**
 * TextInput - Textarea for entering text to analyze
 */
export default function TextInput({ text, onTextChange, onSubmit, isLoading }) {
    const handleKeyDown = (e) => {
        // Submit on Ctrl+Enter
        if (e.ctrlKey && e.key === 'Enter') {
            onSubmit();
        }
    };

    return (
        <div className="form-group">
            <label htmlFor="text-input">Введите текст для анализа</label>
            <textarea
                id="text-input"
                value={text}
                onChange={(e) => onTextChange(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Enter English text for morphological analysis...&#10;&#10;Example: The quick brown fox jumps over the lazy dog."
                disabled={isLoading}
            />
            <button
                className="btn btn-primary btn-block"
                onClick={onSubmit}
                disabled={isLoading || !text.trim()}
                style={{ marginTop: '1rem' }}
            >
                {isLoading ? (
                    <>
                        <span className="spinner"></span>
                        Анализ...
                    </>
                ) : (
                    <>
                        🔍 Анализировать
                    </>
                )}
            </button>
            <p style={{ marginTop: '0.5rem', fontSize: '0.875rem', color: 'var(--text-muted)' }}>
                Ctrl+Enter для быстрой отправки
            </p>
        </div>
    );
}
