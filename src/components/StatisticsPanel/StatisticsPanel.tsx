import { useMemo } from 'react';
import { X, BarChart3 } from 'lucide-react';
import { useFollicleStore } from '../../store/follicleStore';
import { useProjectStore } from '../../store/projectStore';
import { Follicle, isCircle, isRectangle, isLinear } from '../../types';
import './StatisticsPanel.css';

interface StatisticsPanelProps {
  onClose: () => void;
}

interface AnnotationStats {
  count: number;
  avgWidth: number;
  avgHeight: number;
  avgArea: number;
  minWidth: number;
  maxWidth: number;
  minHeight: number;
  maxHeight: number;
  widthDistribution: number[];
  heightDistribution: number[];
  shapeBreakdown: {
    circles: number;
    rectangles: number;
    linears: number;
  };
}

/**
 * Get bounding box dimensions for any annotation type.
 */
function getAnnotationSize(f: Follicle): { width: number; height: number } {
  if (isCircle(f)) {
    const diameter = f.radius * 2;
    return { width: diameter, height: diameter };
  } else if (isRectangle(f)) {
    return { width: f.width, height: f.height };
  } else if (isLinear(f)) {
    const dx = f.endPoint.x - f.startPoint.x;
    const dy = f.endPoint.y - f.startPoint.y;
    const length = Math.sqrt(dx * dx + dy * dy);
    return { width: length, height: f.halfWidth * 2 };
  }
  return { width: 0, height: 0 };
}

/**
 * Calculate statistics for a set of annotations.
 */
function calculateStats(follicles: Follicle[]): AnnotationStats | null {
  if (follicles.length === 0) {
    return null;
  }

  const sizes = follicles.map(getAnnotationSize);
  const widths = sizes.map(s => s.width);
  const heights = sizes.map(s => s.height);
  const areas = sizes.map(s => s.width * s.height);

  // Calculate averages
  const avgWidth = widths.reduce((a, b) => a + b, 0) / widths.length;
  const avgHeight = heights.reduce((a, b) => a + b, 0) / heights.length;
  const avgArea = areas.reduce((a, b) => a + b, 0) / areas.length;

  // Calculate min/max
  const minWidth = Math.min(...widths);
  const maxWidth = Math.max(...widths);
  const minHeight = Math.min(...heights);
  const maxHeight = Math.max(...heights);

  // Create distribution histograms (10 bins)
  const numBins = 10;
  const widthBinSize = (maxWidth - minWidth) / numBins || 1;
  const heightBinSize = (maxHeight - minHeight) / numBins || 1;

  const widthDistribution = new Array(numBins).fill(0);
  const heightDistribution = new Array(numBins).fill(0);

  for (const w of widths) {
    const bin = Math.min(Math.floor((w - minWidth) / widthBinSize), numBins - 1);
    widthDistribution[bin]++;
  }

  for (const h of heights) {
    const bin = Math.min(Math.floor((h - minHeight) / heightBinSize), numBins - 1);
    heightDistribution[bin]++;
  }

  // Shape breakdown
  const shapeBreakdown = {
    circles: follicles.filter(isCircle).length,
    rectangles: follicles.filter(isRectangle).length,
    linears: follicles.filter(isLinear).length,
  };

  return {
    count: follicles.length,
    avgWidth,
    avgHeight,
    avgArea,
    minWidth,
    maxWidth,
    minHeight,
    maxHeight,
    widthDistribution,
    heightDistribution,
    shapeBreakdown,
  };
}

/**
 * Mini bar chart component for distribution display.
 */
function MiniBarChart({ data, label }: { data: number[]; label: string }) {
  const maxValue = Math.max(...data, 1);

  return (
    <div className="mini-chart">
      <div className="chart-label">{label}</div>
      <div className="chart-bars">
        {data.map((value, i) => (
          <div
            key={i}
            className="chart-bar"
            style={{ height: `${(value / maxValue) * 100}%` }}
            title={`${value} annotations`}
          />
        ))}
      </div>
    </div>
  );
}

export function StatisticsPanel({ onClose }: StatisticsPanelProps) {
  const follicles = useFollicleStore(state => state.follicles);
  const activeImageId = useProjectStore(state => state.activeImageId);
  const images = useProjectStore(state => state.images);

  // Filter follicles for current image
  const currentImageFollicles = useMemo(() => {
    if (!activeImageId) return [];
    return follicles.filter(f => f.imageId === activeImageId);
  }, [follicles, activeImageId]);

  // Calculate stats for current image and all images
  const currentImageStats = useMemo(
    () => calculateStats(currentImageFollicles),
    [currentImageFollicles]
  );

  const allStats = useMemo(
    () => calculateStats(follicles),
    [follicles]
  );

  const activeImage = activeImageId ? images.get(activeImageId) : null;

  return (
    <div className="statistics-panel">
      <div className="statistics-header">
        <div className="statistics-title">
          <BarChart3 size={18} />
          <span>Statistics</span>
        </div>
        <button className="close-button" onClick={onClose} title="Close">
          <X size={18} />
        </button>
      </div>

      <div className="statistics-content">
        {/* Current Image Stats */}
        <div className="stats-section">
          <h3>Current Image</h3>
          {activeImage && currentImageStats ? (
            <>
              <div className="stat-row">
                <span className="stat-label">Image:</span>
                <span className="stat-value">{activeImage.fileName}</span>
              </div>
              <div className="stat-row">
                <span className="stat-label">Dimensions:</span>
                <span className="stat-value">{activeImage.width} x {activeImage.height}</span>
              </div>
              <div className="stat-row">
                <span className="stat-label">Annotations:</span>
                <span className="stat-value">{currentImageStats.count}</span>
              </div>
              <div className="stat-row">
                <span className="stat-label">Avg Size:</span>
                <span className="stat-value">
                  {currentImageStats.avgWidth.toFixed(1)} x {currentImageStats.avgHeight.toFixed(1)} px
                </span>
              </div>
              <div className="stat-row">
                <span className="stat-label">Size Range:</span>
                <span className="stat-value">
                  {currentImageStats.minWidth.toFixed(0)}-{currentImageStats.maxWidth.toFixed(0)} x{' '}
                  {currentImageStats.minHeight.toFixed(0)}-{currentImageStats.maxHeight.toFixed(0)} px
                </span>
              </div>
              <div className="stat-row">
                <span className="stat-label">Avg Area:</span>
                <span className="stat-value">{Math.round(currentImageStats.avgArea)} px²</span>
              </div>

              {/* Shape breakdown */}
              <div className="shape-breakdown">
                <span className="stat-label">Shapes:</span>
                <div className="shape-counts">
                  {currentImageStats.shapeBreakdown.circles > 0 && (
                    <span className="shape-count">
                      {currentImageStats.shapeBreakdown.circles} circles
                    </span>
                  )}
                  {currentImageStats.shapeBreakdown.rectangles > 0 && (
                    <span className="shape-count">
                      {currentImageStats.shapeBreakdown.rectangles} rectangles
                    </span>
                  )}
                  {currentImageStats.shapeBreakdown.linears > 0 && (
                    <span className="shape-count">
                      {currentImageStats.shapeBreakdown.linears} linears
                    </span>
                  )}
                </div>
              </div>

              {/* Distribution charts */}
              <div className="distribution-charts">
                <MiniBarChart
                  data={currentImageStats.widthDistribution}
                  label="Width Distribution"
                />
                <MiniBarChart
                  data={currentImageStats.heightDistribution}
                  label="Height Distribution"
                />
              </div>
            </>
          ) : (
            <p className="no-data">No annotations on current image</p>
          )}
        </div>

        {/* Session Stats (All Images) */}
        {images.size > 1 && (
          <div className="stats-section">
            <h3>All Images</h3>
            {allStats ? (
              <>
                <div className="stat-row">
                  <span className="stat-label">Total Images:</span>
                  <span className="stat-value">{images.size}</span>
                </div>
                <div className="stat-row">
                  <span className="stat-label">Total Annotations:</span>
                  <span className="stat-value">{allStats.count}</span>
                </div>
                <div className="stat-row">
                  <span className="stat-label">Avg Size:</span>
                  <span className="stat-value">
                    {allStats.avgWidth.toFixed(1)} x {allStats.avgHeight.toFixed(1)} px
                  </span>
                </div>
                <div className="stat-row">
                  <span className="stat-label">Size Range:</span>
                  <span className="stat-value">
                    {allStats.minWidth.toFixed(0)}-{allStats.maxWidth.toFixed(0)} x{' '}
                    {allStats.minHeight.toFixed(0)}-{allStats.maxHeight.toFixed(0)} px
                  </span>
                </div>

                {/* Distribution charts */}
                <div className="distribution-charts">
                  <MiniBarChart
                    data={allStats.widthDistribution}
                    label="Width Distribution"
                  />
                  <MiniBarChart
                    data={allStats.heightDistribution}
                    label="Height Distribution"
                  />
                </div>
              </>
            ) : (
              <p className="no-data">No annotations in project</p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default StatisticsPanel;
