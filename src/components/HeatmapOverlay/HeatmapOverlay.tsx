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

// Maximum heatmap resolution to prevent memory/GPU issues
const MAX_HEATMAP_SIZE = 800;

/**
 * HeatmapOverlay renders a Gaussian density heatmap over the canvas
 * based on the current annotations.
 */
export function HeatmapOverlay({ canvasWidth, canvasHeight }: HeatmapOverlayProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [heatmapCanvas, setHeatmapCanvas] = useState<OffscreenCanvas | null>(null);

  const showHeatmap = useCanvasStore(state => state.showHeatmap);
  const heatmapOptions = useCanvasStore(state => state.heatmapOptions);

  const follicles = useFollicleStore(state => state.follicles);
  const activeImageId = useProjectStore(state => state.activeImageId);
  const activeImage = useProjectStore(state =>
    state.activeImageId ? state.images.get(state.activeImageId) : null
  );

  // Extract stable image dimensions (don't change on viewport updates)
  const imageWidth = activeImage?.width ?? 0;
  const imageHeight = activeImage?.height ?? 0;

  // Filter follicles for current image
  const currentFollicles = useMemo(() => {
    if (!activeImageId) return [];
    return follicles.filter(f => f.imageId === activeImageId);
  }, [follicles, activeImageId]);

  // Get centers for heatmap
  const centers = useMemo(() => {
    return getFollicleCenters(currentFollicles);
  }, [currentFollicles]);

  // Generate heatmap when dependencies change (NOT on viewport/zoom changes)
  useEffect(() => {
    if (!showHeatmap || imageWidth <= 0 || imageHeight <= 0) {
      setHeatmapCanvas(null);
      return;
    }

    // If no annotations, just clear and return (no heatmap to show)
    if (centers.length === 0) {
      setHeatmapCanvas(null);
      return;
    }

    // Calculate scaled dimensions to prevent memory issues with large images
    const scale = Math.min(1, MAX_HEATMAP_SIZE / Math.max(imageWidth, imageHeight));
    const scaledWidth = Math.max(1, Math.round(imageWidth * scale));
    const scaledHeight = Math.max(1, Math.round(imageHeight * scale));

    // Scale centers to match the scaled heatmap
    const scaledCenters = centers.map(c => ({
      x: c.x * scale,
      y: c.y * scale,
    }));

    // Scale sigma proportionally (minimum of 1 to avoid zero sigma)
    const scaledOptions = {
      ...heatmapOptions,
      sigma: Math.max(1, heatmapOptions.sigma * scale),
    };

    // Track if this effect is still current
    let isCancelled = false;

    // Generate heatmap asynchronously using setTimeout to not block UI
    const generateAsync = async () => {
      try {
        // Use setTimeout to yield to the main thread
        await new Promise(resolve => setTimeout(resolve, 0));

        if (isCancelled) return;

        const imageData = generateHeatmap(scaledCenters, scaledWidth, scaledHeight, scaledOptions);

        if (isCancelled) return;

        // Use OffscreenCanvas instead of createImageBitmap to avoid GPU crashes
        const offscreen = new OffscreenCanvas(scaledWidth, scaledHeight);
        const ctx = offscreen.getContext('2d');
        if (!ctx) {
          console.error('Failed to get 2d context for heatmap');
          return;
        }
        ctx.putImageData(imageData, 0, 0);

        if (isCancelled) return;

        setHeatmapCanvas(offscreen);
      } catch (error) {
        console.error('Failed to generate heatmap:', error);
        setHeatmapCanvas(null);
      }
    };

    generateAsync();

    return () => {
      isCancelled = true;
    };
  }, [showHeatmap, imageWidth, imageHeight, centers, heatmapOptions]);

  // Render heatmap to canvas (runs on viewport changes for smooth zoom/pan)
  useEffect(() => {
    if (!canvasRef.current || !activeImage) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvasWidth, canvasHeight);

    if (!showHeatmap || !heatmapCanvas) return;

    // Get viewport for transformation
    const { viewport } = activeImage;
    const { offsetX, offsetY, scale } = viewport;

    // Save context state
    ctx.save();

    // Apply viewport transformation
    ctx.translate(offsetX, offsetY);
    ctx.scale(scale, scale);

    // Draw heatmap scaled to original image dimensions
    // (the offscreen canvas may be smaller due to MAX_HEATMAP_SIZE optimization)
    ctx.drawImage(heatmapCanvas, 0, 0, imageWidth, imageHeight);

    // Restore context state
    ctx.restore();
  }, [heatmapCanvas, activeImage, canvasWidth, canvasHeight, showHeatmap]);

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
