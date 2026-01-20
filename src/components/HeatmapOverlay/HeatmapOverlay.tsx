import { useEffect, useRef, useState, useMemo } from 'react';
import { useCanvasStore } from '../../store/canvasStore';
import { useFollicleStore } from '../../store/follicleStore';
import { useProjectStore } from '../../store/projectStore';
import {
  generateHeatmap,
  getFollicleCenters,
} from '../../services/heatmapGenerator';

interface HeatmapOverlayProps {
  canvasWidth: number;
  canvasHeight: number;
}

/**
 * HeatmapOverlay renders a Gaussian density heatmap over the canvas
 * based on the current annotations.
 */
export function HeatmapOverlay({ canvasWidth, canvasHeight }: HeatmapOverlayProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [imageBitmap, setImageBitmap] = useState<ImageBitmap | null>(null);

  const showHeatmap = useCanvasStore(state => state.showHeatmap);
  const heatmapOptions = useCanvasStore(state => state.heatmapOptions);

  const follicles = useFollicleStore(state => state.follicles);
  const activeImageId = useProjectStore(state => state.activeImageId);
  const activeImage = useProjectStore(state =>
    state.activeImageId ? state.images.get(state.activeImageId) : null
  );

  // Filter follicles for current image
  const currentFollicles = useMemo(() => {
    if (!activeImageId) return [];
    return follicles.filter(f => f.imageId === activeImageId);
  }, [follicles, activeImageId]);

  // Get centers for heatmap
  const centers = useMemo(() => {
    return getFollicleCenters(currentFollicles);
  }, [currentFollicles]);

  // Generate heatmap when dependencies change
  useEffect(() => {
    if (!showHeatmap || !activeImage || centers.length === 0) {
      setImageBitmap(null);
      return;
    }

    const { width, height } = activeImage;

    // Generate heatmap on a separate task to not block UI
    const generateAsync = async () => {
      const imageData = generateHeatmap(centers, width, height, heatmapOptions);
      const bitmap = await createImageBitmap(imageData);
      setImageBitmap(bitmap);
    };

    generateAsync();

    return () => {
      // Cleanup
      if (imageBitmap) {
        imageBitmap.close();
      }
    };
  }, [showHeatmap, activeImage, centers, heatmapOptions]);

  // Render heatmap to canvas
  useEffect(() => {
    if (!canvasRef.current || !activeImage) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvasWidth, canvasHeight);

    if (!showHeatmap || !imageBitmap) return;

    // Get viewport for transformation
    const { viewport } = activeImage;
    const { offsetX, offsetY, scale } = viewport;

    // Save context state
    ctx.save();

    // Apply viewport transformation
    ctx.translate(offsetX, offsetY);
    ctx.scale(scale, scale);

    // Draw heatmap
    ctx.drawImage(imageBitmap, 0, 0);

    // Restore context state
    ctx.restore();
  }, [imageBitmap, activeImage, canvasWidth, canvasHeight, showHeatmap]);

  if (!showHeatmap) {
    return null;
  }

  return (
    <canvas
      ref={canvasRef}
      width={canvasWidth}
      height={canvasHeight}
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        pointerEvents: 'none',
        zIndex: 10,
      }}
    />
  );
}

export default HeatmapOverlay;
