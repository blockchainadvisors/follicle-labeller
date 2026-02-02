import { useState } from 'react';
import { X } from 'lucide-react';
import { DetectionModelsTab } from './DetectionModelsTab';
import { OriginModelsTab } from './OriginModelsTab';
import './UnifiedYOLOModelManager.css';

interface UnifiedYOLOModelManagerProps {
  onClose: () => void;
  onModelLoaded?: (modelPath: string) => void;
}

type TabType = 'detection' | 'origin';

export function UnifiedYOLOModelManager({ onClose, onModelLoaded }: UnifiedYOLOModelManagerProps) {
  const [activeTab, setActiveTab] = useState<TabType>('detection');

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
            <DetectionModelsTab onModelLoaded={onModelLoaded} />
          ) : (
            <OriginModelsTab onModelLoaded={onModelLoaded} />
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
