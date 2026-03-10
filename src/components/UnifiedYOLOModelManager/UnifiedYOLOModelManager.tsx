import { useState, useEffect } from 'react';
import { X } from 'lucide-react';
import { DetectionModelsTab } from './DetectionModelsTab';
import { OriginModelsTab } from './OriginModelsTab';
import './UnifiedYOLOModelManager.css';

interface UnifiedYOLOModelManagerProps {
  onClose: () => void;
  initialTab?: 'detection' | 'origin';  // Open to specific tab
  selectedDetectionModelId?: string | null;  // Currently selected detection model
  selectedKeypointModelId?: string | null;   // Currently selected keypoint model
  onModelSelected?: (modelType: 'detection' | 'keypoint', modelId: string | null, modelName: string | null) => void;
}

type TabType = 'detection' | 'origin';

export function UnifiedYOLOModelManager({
  onClose,
  initialTab,
  selectedDetectionModelId,
  selectedKeypointModelId,
  onModelSelected,
}: UnifiedYOLOModelManagerProps) {
  const [activeTab, setActiveTab] = useState<TabType>(initialTab || 'detection');

  // Update active tab when initialTab changes
  useEffect(() => {
    if (initialTab) {
      setActiveTab(initialTab);
    }
  }, [initialTab]);

  return (
    <div className="unified-yolo-model-manager-overlay" onClick={(e) => e.target === e.currentTarget && onClose()}>
      <div className="unified-yolo-model-manager-dialog">
        <div className="dialog-header">
          <h2>Model Library</h2>
          <button className="close-button" onClick={onClose}>
            <X size={20} />
          </button>
        </div>

        <div className="tab-bar">
          <button
            className={`tab-button ${activeTab === 'detection' ? 'active' : ''}`}
            onClick={() => setActiveTab('detection')}
          >
            Area Detection
          </button>
          <button
            className={`tab-button ${activeTab === 'origin' ? 'active' : ''}`}
            onClick={() => setActiveTab('origin')}
          >
            Origin Prediction
          </button>
        </div>

        <div className="tab-content">
          {activeTab === 'detection' ? (
            <DetectionModelsTab
              selectedModelId={selectedDetectionModelId}
              onModelSelected={onModelSelected ? (modelId, modelName) => onModelSelected('detection', modelId, modelName) : undefined}
            />
          ) : (
            <OriginModelsTab
              selectedModelId={selectedKeypointModelId}
              onModelSelected={onModelSelected ? (modelId, modelName) => onModelSelected('keypoint', modelId, modelName) : undefined}
            />
          )}
        </div>

        <div className="dialog-footer">
          <button className="close-dialog-button" onClick={onClose}>
            Close
          </button>
        </div>
      </div>
    </div>
  );
}

export default UnifiedYOLOModelManager;
