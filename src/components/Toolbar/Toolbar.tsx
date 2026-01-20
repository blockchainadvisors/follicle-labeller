import React, { useEffect, useCallback, useRef } from 'react';
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
  HelpCircle,
  Eye,
  EyeOff,
  Tag,
  ImagePlus,
  Download,
  BoxSelect,
  Lasso,
} from 'lucide-react';
import { useCanvasStore } from '../../store/canvasStore';
import { useProjectStore, generateImageId } from '../../store/projectStore';
import { useFollicleStore, useTemporalStore } from '../../store/follicleStore';
import { generateExportV2, parseImportV2 } from '../../utils/export-utils';
import { extractAllFolliclesToZip, extractSelectedFolliclesToZip, extractImageFolliclesToZip, downloadBlob } from '../../utils/follicle-extract';
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
  const selectionToolType = useCanvasStore(state => state.selectionToolType);
  const setSelectionToolType = useCanvasStore(state => state.setSelectionToolType);
  const showHelp = useCanvasStore(state => state.showHelp);
  const toggleHelp = useCanvasStore(state => state.toggleHelp);

  // Project store for multi-image support
  const images = useProjectStore(state => state.images);
  const imageOrder = useProjectStore(state => state.imageOrder);
  const activeImageId = useProjectStore(state => state.activeImageId);
  const addImage = useProjectStore(state => state.addImage);
  const clearProject = useProjectStore(state => state.clearProject);
  const zoom = useProjectStore(state => state.zoom);
  const resetZoom = useProjectStore(state => state.resetZoom);
  const currentProjectPath = useProjectStore(state => state.currentProjectPath);
  const setCurrentProjectPath = useProjectStore(state => state.setCurrentProjectPath);
  const isDirty = useProjectStore(state => state.isDirty);
  const setDirty = useProjectStore(state => state.setDirty);
  const markClean = useProjectStore(state => state.markClean);

  // Get active image info
  const activeImage = activeImageId ? images.get(activeImageId) : null;
  const imageLoaded = activeImage !== null;
  const viewport = activeImage?.viewport ?? { offsetX: 0, offsetY: 0, scale: 1 };

  const follicles = useFollicleStore(state => state.follicles);
  const selectedIds = useFollicleStore(state => state.selectedIds);
  const importFollicles = useFollicleStore(state => state.importFollicles);
  const clearAll = useFollicleStore(state => state.clearAll);

  const temporalStore = useTemporalStore();

  // Get annotation count for active image only
  const activeImageAnnotationCount = activeImageId
    ? follicles.filter(f => f.imageId === activeImageId).length
    : 0;

  // Update menu state when project changes
  useEffect(() => {
    const hasProject = images.size > 0;
    window.electronAPI.setProjectState(hasProject);
  }, [images.size]);

  // Track follicle changes to mark project as dirty
  const prevFolliclesRef = useRef(follicles);
  useEffect(() => {
    if (prevFolliclesRef.current !== follicles && images.size > 0) {
      setDirty(true);
    }
    prevFolliclesRef.current = follicles;
  }, [follicles, images.size, setDirty]);

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

  const handleSave = useCallback(async (): Promise<boolean> => {
    if (images.size === 0) return false;

    try {
      const { manifest, annotations, imageList } = generateExportV2(
        Array.from(images.values()),
        follicles
      );

      let result: { success: boolean; filePath?: string };

      if (currentProjectPath) {
        // Silent save to existing path
        result = await window.electronAPI.saveProjectV2ToPath(
          currentProjectPath,
          imageList,
          JSON.stringify(manifest, null, 2),
          JSON.stringify(annotations, null, 2)
        );
      } else {
        // Show save dialog
        result = await window.electronAPI.saveProjectV2(
          imageList,
          JSON.stringify(manifest, null, 2),
          JSON.stringify(annotations, null, 2)
        );
      }

      if (result.success && result.filePath) {
        setCurrentProjectPath(result.filePath);
        markClean();
        console.log('Project saved successfully to:', result.filePath);
      }
      return result.success;
    } catch (error) {
      console.error('Failed to save project:', error);
      return false;
    }
  }, [images, follicles, currentProjectPath, setCurrentProjectPath, markClean]);

  const handleSaveAs = useCallback(async (): Promise<boolean> => {
    if (images.size === 0) return false;

    try {
      const { manifest, annotations, imageList } = generateExportV2(
        Array.from(images.values()),
        follicles
      );

      // Always show save dialog
      const result = await window.electronAPI.saveProjectV2(
        imageList,
        JSON.stringify(manifest, null, 2),
        JSON.stringify(annotations, null, 2),
        currentProjectPath || undefined
      );

      if (result.success && result.filePath) {
        setCurrentProjectPath(result.filePath);
        markClean();
        console.log('Project saved successfully to:', result.filePath);
      }
      return result.success;
    } catch (error) {
      console.error('Failed to save project:', error);
      return false;
    }
  }, [images, follicles, currentProjectPath, setCurrentProjectPath, markClean]);

  // Check for unsaved changes and prompt user
  // Returns true if safe to proceed, false if user cancelled
  const checkUnsavedChanges = useCallback(async (): Promise<boolean> => {
    if (!isDirty) return true;

    const response = await window.electronAPI.showUnsavedChangesDialog();

    if (response === 'save') {
      const saved = await handleSave();
      return saved;
    } else if (response === 'discard') {
      return true;
    } else {
      // Cancel
      return false;
    }
  }, [isDirty, handleSave]);

  // Load project from parsed result (shared logic)
  const loadProjectFromResult = useCallback(async (result: Awaited<ReturnType<typeof window.electronAPI.loadProjectV2>>) => {
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

    // Set the current project path
    setCurrentProjectPath(result.filePath);
  }, [clearProject, clearAll, addImage, importFollicles, setCurrentProjectPath]);

  const handleLoad = useCallback(async () => {
    // Check for unsaved changes first
    const canProceed = await checkUnsavedChanges();
    if (!canProceed) return;

    try {
      const result = await window.electronAPI.loadProjectV2();
      await loadProjectFromResult(result);
    } catch (error) {
      console.error('Failed to load project:', error);
      alert('Failed to load project file. Please check the file format.');
    }
  }, [loadProjectFromResult, checkUnsavedChanges]);

  const handleCloseProject = useCallback(async () => {
    // Check for unsaved changes first
    const canProceed = await checkUnsavedChanges();
    if (!canProceed) return;

    clearProject();
    clearAll();
  }, [clearProject, clearAll, checkUnsavedChanges]);

  const handleUndo = useCallback(() => {
    temporalStore.getState().undo();
  }, [temporalStore]);

  const handleRedo = useCallback(() => {
    temporalStore.getState().redo();
  }, [temporalStore]);

  const handleZoomIn = useCallback(() => zoom(0.2), [zoom]);
  const handleZoomOut = useCallback(() => zoom(-0.2), [zoom]);

  // Download extracted follicle images
  const handleDownloadFollicles = useCallback(async () => {
    if (images.size === 0 || follicles.length === 0) return;

    try {
      // Calculate counts for dialog
      const selectedCount = selectedIds.size;
      const currentImageCount = activeImageId
        ? follicles.filter(f => f.imageId === activeImageId).length
        : 0;
      const totalCount = follicles.length;

      // Generate base filename
      const baseName = currentProjectPath
        ? currentProjectPath.replace(/\.[^/.]+$/, '').split(/[/\\]/).pop()
        : activeImage?.fileName.replace(/\.[^/.]+$/, '') ?? 'follicles';

      let zipBlob: Blob;
      let suffix: string;

      // If there's a selection, show options dialog
      if (selectedCount > 0) {
        const choice = await window.electronAPI.showDownloadOptionsDialog(
          selectedCount,
          currentImageCount,
          totalCount
        );

        if (choice === 'cancel') return;

        if (choice === 'selected') {
          zipBlob = await extractSelectedFolliclesToZip(images, follicles, selectedIds);
          suffix = '_selected';
        } else if (choice === 'currentImage' && activeImage) {
          zipBlob = await extractImageFolliclesToZip(activeImage, follicles);
          suffix = `_${activeImage.fileName.replace(/\.[^/.]+$/, '')}`;
        } else {
          // 'all'
          zipBlob = await extractAllFolliclesToZip(images, follicles);
          suffix = '_all';
        }
      } else {
        // No selection - download all
        zipBlob = await extractAllFolliclesToZip(images, follicles);
        suffix = '_follicles';
      }

      downloadBlob(zipBlob, `${baseName}${suffix}.zip`);
    } catch (error) {
      console.error('Failed to extract follicles:', error);
      alert('Failed to extract follicle images. Please ensure there are annotations to extract.');
    }
  }, [images, follicles, selectedIds, activeImageId, activeImage, currentProjectPath]);

  // Register menu event listeners
  useEffect(() => {
    const cleanups = [
      window.electronAPI.onMenuOpenImage(handleOpenImage),
      window.electronAPI.onMenuLoadProject(handleLoad),
      window.electronAPI.onMenuSaveProject(handleSave),
      window.electronAPI.onMenuSaveProjectAs(handleSaveAs),
      window.electronAPI.onMenuCloseProject(handleCloseProject),
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
    handleSaveAs,
    handleCloseProject,
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

  // Handle file open from file association (double-click .fol file)
  useEffect(() => {
    const loadFromPath = async (filePath: string) => {
      try {
        const result = await window.electronAPI.loadProjectFromPath(filePath);
        await loadProjectFromResult(result);
      } catch (error) {
        console.error('Failed to load project from file association:', error);
        alert('Failed to load project file. Please check the file format.');
      }
    };

    // Check for file to open on startup
    window.electronAPI.getFileToOpen().then((filePath) => {
      if (filePath) {
        loadFromPath(filePath);
      }
    });

    // Listen for file open while app is running
    const cleanup = window.electronAPI.onFileOpen((filePath) => {
      loadFromPath(filePath);
    });

    return cleanup;
  }, [loadProjectFromResult]);

  // Handle app close - check for unsaved changes
  useEffect(() => {
    const handleCheckUnsavedChanges = async () => {
      const canClose = await checkUnsavedChanges();
      window.electronAPI.confirmClose(canClose);
    };

    const cleanup = window.electronAPI.onCheckUnsavedChanges(handleCheckUnsavedChanges);
    return cleanup;
  }, [checkUnsavedChanges]);

  // Handle system suspend (sleep/hibernate) - auto-save to prevent data loss
  useEffect(() => {
    const handleSystemSuspend = async () => {
      // Only auto-save if there's a project with unsaved changes AND an existing save path
      // (we can't show a dialog during suspend - no time for user interaction)
      if (isDirty && currentProjectPath && images.size > 0) {
        console.log('System suspending - auto-saving project to:', currentProjectPath);
        try {
          const { manifest, annotations, imageList } = generateExportV2(
            Array.from(images.values()),
            follicles
          );

          const result = await window.electronAPI.saveProjectV2ToPath(
            currentProjectPath,
            imageList,
            JSON.stringify(manifest, null, 2),
            JSON.stringify(annotations, null, 2)
          );

          if (result.success) {
            markClean();
            console.log('Auto-save before suspend completed successfully');
          }
        } catch (error) {
          console.error('Auto-save before suspend failed:', error);
        }
      }
    };

    const cleanup = window.electronAPI.onSystemSuspend(handleSystemSuspend);
    return cleanup;
  }, [isDirty, currentProjectPath, images, follicles, markClean]);

  const zoomPercent = Math.round(viewport.scale * 100);

  return (
    <div className="toolbar">
      {/* Add Image */}
      <div className="toolbar-group" role="group" aria-label="Add image">
        <IconButton
          icon={<ImagePlus size={18} />}
          tooltip="Add Image"
          shortcut="Ctrl+O"
          onClick={handleOpenImage}
        />
      </div>

      <div className="toolbar-divider" />

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

      {/* Selection tools (visible only in select mode) */}
      {mode === 'select' && (
        <>
          <div className="toolbar-group" role="group" aria-label="Selection tools">
            <IconButton
              icon={<BoxSelect size={18} />}
              tooltip="Marquee Select"
              shortcut="M"
              onClick={() => setSelectionToolType('marquee')}
              active={selectionToolType === 'marquee'}
            />
            <IconButton
              icon={<Lasso size={18} />}
              tooltip="Lasso Select"
              shortcut="F"
              onClick={() => setSelectionToolType('lasso')}
              active={selectionToolType === 'lasso'}
            />
          </div>
          <div className="toolbar-divider" />
        </>
      )}

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

      {/* Export follicle images */}
      <div className="toolbar-group" role="group" aria-label="Export">
        <IconButton
          icon={<Download size={18} />}
          tooltip="Download Follicle Images"
          onClick={handleDownloadFollicles}
          disabled={!imageLoaded || follicles.length === 0}
        />
      </div>

      <div className="toolbar-divider" />

      {/* Help */}
      <div className="toolbar-group" role="group" aria-label="Help">
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
