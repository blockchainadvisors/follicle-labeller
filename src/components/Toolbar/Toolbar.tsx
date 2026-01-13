import React, { useEffect, useCallback } from 'react';
import {
  Pencil,
  MousePointer2,
  Hand,
  Circle,
  Square,
  Minus,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Camera,
  HelpCircle,
  Eye,
  EyeOff,
  Tag,
} from 'lucide-react';
import { useCanvasStore } from '../../store/canvasStore';
import { useProjectStore, generateImageId } from '../../store/projectStore';
import { useFollicleStore, useTemporalStore } from '../../store/follicleStore';
import { generateExportV2, parseImportV2 } from '../../utils/export-utils';
import { ProjectImage } from '../../types';

// Reusable icon button component
interface IconButtonProps {
  icon: React.ReactNode;
  tooltip: string;
  shortcut?: string;
  onClick: () => void;
  disabled?: boolean;
  active?: boolean;
}

const IconButton: React.FC<IconButtonProps> = ({
  icon,
  tooltip,
  shortcut,
  onClick,
  disabled = false,
  active = false,
}) => {
  const tooltipText = shortcut ? `${tooltip} (${shortcut})` : tooltip;

  return (
    <button
      className={`icon-button ${active ? 'active' : ''}`}
      onClick={onClick}
      disabled={disabled}
      title={tooltipText}
      aria-label={tooltip}
    >
      {icon}
    </button>
  );
};

export const Toolbar: React.FC = () => {
  const mode = useCanvasStore(state => state.mode);
  const setMode = useCanvasStore(state => state.setMode);
  const showLabels = useCanvasStore(state => state.showLabels);
  const toggleLabels = useCanvasStore(state => state.toggleLabels);
  const showShapes = useCanvasStore(state => state.showShapes);
  const toggleShapes = useCanvasStore(state => state.toggleShapes);
  const currentShapeType = useCanvasStore(state => state.currentShapeType);
  const setShapeType = useCanvasStore(state => state.setShapeType);
  const showHelp = useCanvasStore(state => state.showHelp);
  const toggleHelp = useCanvasStore(state => state.toggleHelp);
  const canvasRef = useCanvasStore(state => state.canvasRef);

  // Project store for multi-image support
  const images = useProjectStore(state => state.images);
  const imageOrder = useProjectStore(state => state.imageOrder);
  const activeImageId = useProjectStore(state => state.activeImageId);
  const addImage = useProjectStore(state => state.addImage);
  const clearProject = useProjectStore(state => state.clearProject);
  const zoom = useProjectStore(state => state.zoom);
  const resetZoom = useProjectStore(state => state.resetZoom);

  // Get active image info
  const activeImage = activeImageId ? images.get(activeImageId) : null;
  const imageLoaded = activeImage !== null;
  const viewport = activeImage?.viewport ?? { offsetX: 0, offsetY: 0, scale: 1 };

  const follicles = useFollicleStore(state => state.follicles);
  const importFollicles = useFollicleStore(state => state.importFollicles);
  const clearAll = useFollicleStore(state => state.clearAll);

  const temporalStore = useTemporalStore();

  // Get annotation count for active image only
  const activeImageAnnotationCount = activeImageId
    ? follicles.filter(f => f.imageId === activeImageId).length
    : 0;

  // Handler functions
  const handleOpenImage = useCallback(async () => {
    try {
      const result = await window.electronAPI.openImageDialog();
      if (result) {
        const blob = new Blob([result.data]);
        const url = URL.createObjectURL(blob);

        // Create pre-decoded ImageBitmap for smooth rendering
        const bitmap = await createImageBitmap(blob);

        const newImage: ProjectImage = {
          id: generateImageId(),
          fileName: result.fileName,
          width: bitmap.width,
          height: bitmap.height,
          imageData: result.data,
          imageBitmap: bitmap,
          imageSrc: url,
          viewport: { offsetX: 0, offsetY: 0, scale: 1 },
          createdAt: Date.now(),
          sortOrder: imageOrder.length,
        };

        addImage(newImage);
      }
    } catch (error) {
      console.error('Failed to open image:', error);
    }
  }, [addImage, imageOrder.length]);

  const handleSave = useCallback(async () => {
    if (images.size === 0) return;

    try {
      const { manifest, annotations, imageList } = generateExportV2(
        Array.from(images.values()),
        follicles
      );

      const saved = await window.electronAPI.saveProjectV2(
        imageList,
        JSON.stringify(manifest, null, 2),
        JSON.stringify(annotations, null, 2)
      );

      if (saved) {
        console.log('Project saved successfully');
      }
    } catch (error) {
      console.error('Failed to save project:', error);
    }
  }, [images, follicles]);

  const handleLoad = useCallback(async () => {
    try {
      const result = await window.electronAPI.loadProjectV2();
      if (!result) return;

      // Clear existing project
      clearProject();
      clearAll();

      const { loadedImages, loadedFollicles } = await parseImportV2(result);

      // Add all loaded images
      for (const image of loadedImages) {
        addImage(image);
      }

      // Import all annotations
      importFollicles(loadedFollicles);
    } catch (error) {
      console.error('Failed to load project:', error);
      alert('Failed to load project file. Please check the file format.');
    }
  }, [clearProject, clearAll, addImage, importFollicles]);

  const handleUndo = useCallback(() => {
    temporalStore.getState().undo();
  }, [temporalStore]);

  const handleRedo = useCallback(() => {
    temporalStore.getState().redo();
  }, [temporalStore]);

  const handleScreenshot = useCallback(async () => {
    if (!canvasRef) return;

    try {
      const blob = await new Promise<Blob | null>((resolve) => {
        canvasRef.toBlob(resolve, 'image/png');
      });

      if (!blob) {
        console.error('Failed to create screenshot blob');
        return;
      }

      const arrayBuffer = await blob.arrayBuffer();
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
      const suggestedName = `screenshot-${timestamp}.png`;

      const saved = await window.electronAPI.saveScreenshot(arrayBuffer, suggestedName);
      if (saved) {
        console.log('Screenshot saved successfully');
      }
    } catch (error) {
      console.error('Failed to save screenshot:', error);
    }
  }, [canvasRef]);

  const handleZoomIn = useCallback(() => zoom(0.2), [zoom]);
  const handleZoomOut = useCallback(() => zoom(-0.2), [zoom]);

  // Register menu event listeners
  useEffect(() => {
    const cleanups = [
      window.electronAPI.onMenuOpenImage(handleOpenImage),
      window.electronAPI.onMenuLoadProject(handleLoad),
      window.electronAPI.onMenuSaveProject(handleSave),
      window.electronAPI.onMenuUndo(handleUndo),
      window.electronAPI.onMenuRedo(handleRedo),
      window.electronAPI.onMenuClearAll(clearAll),
      window.electronAPI.onMenuToggleShapes(toggleShapes),
      window.electronAPI.onMenuToggleLabels(toggleLabels),
      window.electronAPI.onMenuZoomIn(handleZoomIn),
      window.electronAPI.onMenuZoomOut(handleZoomOut),
      window.electronAPI.onMenuResetZoom(resetZoom),
      window.electronAPI.onMenuShowHelp(toggleHelp),
    ];

    return () => cleanups.forEach(cleanup => cleanup());
  }, [
    handleOpenImage,
    handleLoad,
    handleSave,
    handleUndo,
    handleRedo,
    clearAll,
    toggleShapes,
    toggleLabels,
    handleZoomIn,
    handleZoomOut,
    resetZoom,
    toggleHelp,
  ]);

  const zoomPercent = Math.round(viewport.scale * 100);

  return (
    <div className="toolbar">
      {/* Mode tools */}
      <div className="toolbar-group" role="group" aria-label="Interaction modes">
        <IconButton
          icon={<Pencil size={18} />}
          tooltip="Create Mode"
          shortcut="C"
          onClick={() => setMode('create')}
          active={mode === 'create'}
        />
        <IconButton
          icon={<MousePointer2 size={18} />}
          tooltip="Select Mode"
          shortcut="V"
          onClick={() => setMode('select')}
          active={mode === 'select'}
        />
        <IconButton
          icon={<Hand size={18} />}
          tooltip="Pan Mode"
          shortcut="H"
          onClick={() => setMode('pan')}
          active={mode === 'pan'}
        />
      </div>

      <div className="toolbar-divider" />

      {/* Shape tools */}
      <div className="toolbar-group" role="group" aria-label="Shape types">
        <IconButton
          icon={<Circle size={18} />}
          tooltip="Circle Shape"
          shortcut="1"
          onClick={() => setShapeType('circle')}
          active={currentShapeType === 'circle'}
        />
        <IconButton
          icon={<Square size={18} />}
          tooltip="Rectangle Shape"
          shortcut="2"
          onClick={() => setShapeType('rectangle')}
          active={currentShapeType === 'rectangle'}
        />
        <IconButton
          icon={<Minus size={18} strokeWidth={3} />}
          tooltip="Linear Shape"
          shortcut="3"
          onClick={() => setShapeType('linear')}
          active={currentShapeType === 'linear'}
        />
      </div>

      <div className="toolbar-divider" />

      {/* Zoom controls */}
      <div className="toolbar-group" role="group" aria-label="Zoom controls">
        <IconButton
          icon={<ZoomOut size={18} />}
          tooltip="Zoom Out"
          shortcut="Ctrl+-"
          onClick={handleZoomOut}
        />
        <span className="zoom-display">{zoomPercent}%</span>
        <IconButton
          icon={<ZoomIn size={18} />}
          tooltip="Zoom In"
          shortcut="Ctrl+="
          onClick={handleZoomIn}
        />
        <IconButton
          icon={<RotateCcw size={18} />}
          tooltip="Reset Zoom"
          shortcut="Ctrl+0"
          onClick={resetZoom}
        />
      </div>

      <div className="toolbar-divider" />

      {/* View toggles */}
      <div className="toolbar-group" role="group" aria-label="View options">
        <IconButton
          icon={showShapes ? <Eye size={18} /> : <EyeOff size={18} />}
          tooltip="Toggle Shapes"
          shortcut="O"
          onClick={toggleShapes}
          active={showShapes}
        />
        <IconButton
          icon={<Tag size={18} />}
          tooltip="Toggle Labels"
          shortcut="L"
          onClick={toggleLabels}
          disabled={!showShapes}
          active={showLabels}
        />
      </div>

      <div className="toolbar-divider" />

      {/* Screenshot and Help */}
      <div className="toolbar-group" role="group" aria-label="Actions">
        <IconButton
          icon={<Camera size={18} />}
          tooltip="Save Screenshot"
          onClick={handleScreenshot}
          disabled={!imageLoaded}
        />
        <IconButton
          icon={<HelpCircle size={18} />}
          tooltip="User Guide"
          shortcut="?"
          onClick={toggleHelp}
          active={showHelp}
        />
      </div>

      {/* Status display */}
      <div className="toolbar-spacer" />
      <div className="toolbar-status">
        {imageLoaded && activeImage ? (
          <>
            {images.size > 1 && (
              <span className="image-count">{images.size} images</span>
            )}
            <span className="file-name">{activeImage.fileName}</span>
            <span className="image-size">{activeImage.width} x {activeImage.height}</span>
            <span className="follicle-count">{activeImageAnnotationCount} annotations</span>
          </>
        ) : (
          <span className="no-image">No image loaded</span>
        )}
      </div>
    </div>
  );
};
