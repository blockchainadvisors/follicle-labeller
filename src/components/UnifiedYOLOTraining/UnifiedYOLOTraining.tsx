import { useState } from 'react';
import { X } from 'lucide-react';
import { DetectionTab } from './DetectionTab';
import { OriginTab } from './OriginTab';
import './UnifiedYOLOTraining.css';

interface UnifiedYOLOTrainingProps {
  onClose: () => void;
}

type TabType = 'detection' | 'origin';

export function UnifiedYOLOTraining({ onClose }: UnifiedYOLOTrainingProps) {
  const [activeTab, setActiveTab] = useState<TabType>('detection');

  return (
    <div className="unified-yolo-training-overlay" onClick={(e) => e.target === e.currentTarget && onClose()}>
      <div className="unified-yolo-training-dialog">
        <div className="dialog-header">
          <h2>Model Training</h2>
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
          {activeTab === 'detection' ? <DetectionTab /> : <OriginTab />}
        </div>
      </div>
    </div>
  );
}

export default UnifiedYOLOTraining;
