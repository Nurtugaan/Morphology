import React from 'react';

/**
 * ModelSelector - Dropdown for selecting morphological analysis model
 */
export default function ModelSelector({ models, selectedModel, onModelChange, disabled }) {
    return (
        <div className="form-group">
            <label htmlFor="model-select">Выберите модель</label>
            <div className="select-wrapper">
                <select
                    id="model-select"
                    value={selectedModel}
                    onChange={(e) => onModelChange(e.target.value)}
                    disabled={disabled}
                >
                    {models.map((model) => (
                        <option key={model.id} value={model.id}>
                            {model.name} — {model.description}
                            {model.loaded ? ' ✓' : ''}
                        </option>
                    ))}
                </select>
            </div>
        </div>
    );
}
