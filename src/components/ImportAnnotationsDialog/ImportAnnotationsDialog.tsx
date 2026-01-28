import { useState } from 'react';
import { X, Check, AlertCircle, ArrowUp, Copy, Minus } from 'lucide-react';
import type { Follicle } from '../../types';
import type { AugmentableAnnotation } from '../../utils/export-utils';
import './ImportAnnotationsDialog.css';

export interface ImportAnalysis {
  newAnnotations: Follicle[];
  augmentable: AugmentableAnnotation[];
  alreadyAugmented: Follicle[];
  duplicates: Follicle[];
  totalImported: number;
}

export interface ImportOptions {
  importNew: boolean;
  updateExisting: boolean;
  importAlreadyAugmented: boolean;
  importDuplicates: boolean;
}

interface ImportAnnotationsDialogProps {
  analysis: ImportAnalysis;
  onImport: (options: ImportOptions) => void;
  onCancel: () => void;
}

export function ImportAnnotationsDialog({
  analysis,
  onImport,
  onCancel,
}: ImportAnnotationsDialogProps) {
  const { newAnnotations, augmentable, alreadyAugmented, duplicates, totalImported } = analysis;

  // Default options: import new, update existing, skip redundant
  const [options, setOptions] = useState<ImportOptions>({
    importNew: true,
    updateExisting: augmentable.length > 0,
    importAlreadyAugmented: false,
    importDuplicates: false,
  });

  const handleOptionChange = (key: keyof ImportOptions, value: boolean) => {
    setOptions(prev => ({ ...prev, [key]: value }));
  };

  const handleImport = () => {
    onImport(options);
  };

  // Calculate what will happen
  const willImportCount = (options.importNew ? newAnnotations.length : 0) +
    (options.importAlreadyAugmented ? alreadyAugmented.length : 0) +
    (options.importDuplicates ? duplicates.length : 0);
  const willUpdateCount = options.updateExisting ? augmentable.length : 0;
  const totalActions = willImportCount + willUpdateCount;

  return (
    <div className="import-dialog-overlay" onClick={onCancel}>
      <div className="import-dialog" onClick={e => e.stopPropagation()}>
        <div className="dialog-header">
          <h2>Import Annotations</h2>
          <button className="close-button" onClick={onCancel}>
            <X size={18} />
          </button>
        </div>

        <div className="dialog-content">
          <div className="import-summary">
            <span className="summary-total">{totalImported} annotations found in file</span>
          </div>

          {/* New Annotations */}
          {newAnnotations.length > 0 && (
            <section className="import-section">
              <label className="import-option">
                <input
                  type="checkbox"
                  checked={options.importNew}
                  onChange={e => handleOptionChange('importNew', e.target.checked)}
                />
                <div className="option-icon new">
                  <Check size={14} />
                </div>
                <div className="option-content">
                  <span className="option-title">
                    Import {newAnnotations.length} new annotation{newAnnotations.length !== 1 ? 's' : ''}
                  </span>
                  <span className="option-description">
                    These annotations don't exist in the current image and will be added.
                  </span>
                </div>
              </label>
            </section>
          )}

          {/* Augmentable - can update existing */}
          {augmentable.length > 0 && (
            <section className="import-section">
              <label className="import-option">
                <input
                  type="checkbox"
                  checked={options.updateExisting}
                  onChange={e => handleOptionChange('updateExisting', e.target.checked)}
                />
                <div className="option-icon augmentable">
                  <ArrowUp size={14} />
                </div>
                <div className="option-content">
                  <span className="option-title">
                    Update {augmentable.length} existing annotation{augmentable.length !== 1 ? 's' : ''} with origin data
                  </span>
                  <span className="option-description">
                    These match existing annotations but have origin/direction data that the existing ones lack.
                    Checking this will add the origin data to the existing annotations.
                  </span>
                </div>
              </label>
            </section>
          )}

          {/* Already Augmented - existing has origin, imported doesn't */}
          {alreadyAugmented.length > 0 && (
            <section className="import-section">
              <label className="import-option">
                <input
                  type="checkbox"
                  checked={options.importAlreadyAugmented}
                  onChange={e => handleOptionChange('importAlreadyAugmented', e.target.checked)}
                />
                <div className="option-icon already-augmented">
                  <Minus size={14} />
                </div>
                <div className="option-content">
                  <span className="option-title">
                    Import {alreadyAugmented.length} annotation{alreadyAugmented.length !== 1 ? 's' : ''} as new (existing already has origin)
                  </span>
                  <span className="option-description">
                    These match existing annotations that already have origin/direction data.
                    The imported versions don't have origin data, so they would be redundant.
                    <strong> Skip recommended.</strong>
                  </span>
                </div>
              </label>
            </section>
          )}

          {/* Exact Duplicates */}
          {duplicates.length > 0 && (
            <section className="import-section">
              <label className="import-option">
                <input
                  type="checkbox"
                  checked={options.importDuplicates}
                  onChange={e => handleOptionChange('importDuplicates', e.target.checked)}
                />
                <div className="option-icon duplicate">
                  <Copy size={14} />
                </div>
                <div className="option-content">
                  <span className="option-title">
                    Import {duplicates.length} duplicate{duplicates.length !== 1 ? 's' : ''} as new annotations
                  </span>
                  <span className="option-description">
                    These exactly match existing annotations (same position and origin state).
                    Importing will create duplicate annotations.
                    <strong> Skip recommended.</strong>
                  </span>
                </div>
              </label>
            </section>
          )}

          {/* Nothing useful to import */}
          {newAnnotations.length === 0 && augmentable.length === 0 && (
            <div className="import-warning">
              <AlertCircle size={16} />
              <span>No new data found. All annotations already exist in the current image.</span>
            </div>
          )}
        </div>

        <div className="dialog-footer">
          <div className="import-preview">
            {totalActions > 0 ? (
              <span>
                {willImportCount > 0 && `${willImportCount} to import`}
                {willImportCount > 0 && willUpdateCount > 0 && ', '}
                {willUpdateCount > 0 && `${willUpdateCount} to update`}
              </span>
            ) : (
              <span className="no-actions">No actions selected</span>
            )}
          </div>
          <div className="footer-actions">
            <button className="button-secondary" onClick={onCancel}>
              Cancel
            </button>
            <button
              className="button-primary"
              onClick={handleImport}
              disabled={totalActions === 0}
            >
              Import
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
