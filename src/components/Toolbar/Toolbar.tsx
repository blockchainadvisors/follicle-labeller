import React from 'react';
import { useCanvasStore } from '../../store/canvasStore';
import { useFollicleStore, useTemporalStore } from '../../store/follicleStore';
import { generateExport, parseImport } from '../../utils/export-utils';

export const Toolbar: React.FC = () => {
  const mode = useCanvasStore(state => state.mode);
  const setMode = useCanvasStore(state => state.setMode);
  const viewport = useCanvasStore(state => state.viewport);
  const zoom = useCanvasStore(state => state.zoom);
  const resetZoom = useCanvasStore(state => state.resetZoom);
  const setImage = useCanvasStore(state => state.setImage);
  const imageLoaded = useCanvasStore(state => state.imageLoaded);
  const imageData = useCanvasStore(state => state.imageData);
  const fileName = useCanvasStore(state => state.fileName);
  const imageWidth = useCanvasStore(state => state.imageWidth);
  const imageHeight = useCanvasStore(state => state.imageHeight);
  const showLabels = useCanvasStore(state => state.showLabels);
  const toggleLabels = useCanvasStore(state => state.toggleLabels);
  const showCircles = useCanvasStore(state => state.showCircles);
  const toggleCircles = useCanvasStore(state => state.toggleCircles);

  const follicles = useFollicleStore(state => state.follicles);
  const importFollicles = useFollicleStore(state => state.importFollicles);
  const clearAll = useFollicleStore(state => state.clearAll);

  const temporalStore = useTemporalStore();

  const handleOpenImage = async () => {
    try {
      const result = await window.electronAPI.openImageDialog();
      if (result) {
        const blob = new Blob([result.data]);
        const url = URL.createObjectURL(blob);

        const img = new Image();
        img.onload = () => {
          setImage(url, img.width, img.height, result.fileName, result.data);
        };
        img.src = url;
      }
    } catch (error) {
      console.error('Failed to open image:', error);
    }
  };

  const handleSave = async () => {
    if (!imageData || !fileName) return;

    try {
      const exportData = generateExport(
        follicles,
        fileName,
        imageWidth,
        imageHeight
      );
      const json = JSON.stringify(exportData, null, 2);
      const saved = await window.electronAPI.saveProject(imageData, fileName, json);
      if (saved) {
        console.log('Project saved successfully');
      }
    } catch (error) {
      console.error('Failed to save project:', error);
    }
  };

  const handleLoad = async () => {
    try {
      const result = await window.electronAPI.loadProject();
      if (result) {
        // Load image
        const blob = new Blob([result.imageData]);
        const url = URL.createObjectURL(blob);

        const img = new Image();
        img.onload = () => {
          setImage(url, img.width, img.height, result.imageFileName, result.imageData);

          // Load annotations
          const imported = parseImport(result.jsonData);
          importFollicles(imported);
        };
        img.src = url;
      }
    } catch (error) {
      console.error('Failed to load project:', error);
      alert('Failed to load project file. Please check the file format.');
    }
  };

  const handleUndo = () => {
    temporalStore.getState().undo();
  };

  const handleRedo = () => {
    temporalStore.getState().redo();
  };

  const zoomPercent = Math.round(viewport.scale * 100);

  return (
    <div className="toolbar">
      {/* File operations */}
      <div className="toolbar-group">
        <button onClick={handleOpenImage} title="Open Image">
          Open Image
        </button>
        <button onClick={handleLoad} title="Load Project (.fol)">
          Load
        </button>
        <button
          onClick={handleSave}
          disabled={!imageLoaded}
          title="Save Project (.fol)"
        >
          Save
        </button>
      </div>

      <div className="toolbar-divider" />

      {/* Mode selection */}
      <div className="toolbar-group">
        <button
          className={mode === 'create' ? 'active' : ''}
          onClick={() => setMode('create')}
          title="Create Mode (C)"
        >
          Create
        </button>
        <button
          className={mode === 'select' ? 'active' : ''}
          onClick={() => setMode('select')}
          title="Select Mode (V)"
        >
          Select
        </button>
        <button
          className={mode === 'pan' ? 'active' : ''}
          onClick={() => setMode('pan')}
          title="Pan Mode (H)"
        >
          Pan
        </button>
      </div>

      <div className="toolbar-divider" />

      {/* Zoom controls */}
      <div className="toolbar-group">
        <button onClick={() => zoom(-0.2)} title="Zoom Out">
          -
        </button>
        <span className="zoom-display">{zoomPercent}%</span>
        <button onClick={() => zoom(0.2)} title="Zoom In">
          +
        </button>
        <button onClick={resetZoom} title="Reset Zoom">
          Reset
        </button>
      </div>

      <div className="toolbar-divider" />

      {/* Undo/Redo */}
      <div className="toolbar-group">
        <button onClick={handleUndo} title="Undo (Ctrl+Z)">
          Undo
        </button>
        <button onClick={handleRedo} title="Redo (Ctrl+Shift+Z)">
          Redo
        </button>
      </div>

      <div className="toolbar-divider" />

      {/* View options */}
      <div className="toolbar-group">
        <button
          className={showCircles ? 'active' : ''}
          onClick={toggleCircles}
          title="Toggle Circles (O)"
        >
          Circles
        </button>
        <button
          className={showLabels ? 'active' : ''}
          onClick={toggleLabels}
          disabled={!showCircles}
          title="Toggle Labels (L)"
        >
          Labels
        </button>
      </div>

      <div className="toolbar-divider" />

      {/* Clear all */}
      <div className="toolbar-group">
        <button
          onClick={clearAll}
          disabled={follicles.length === 0}
          className="danger"
          title="Clear All Selections"
        >
          Clear All
        </button>
      </div>

      {/* Status */}
      <div className="toolbar-spacer" />
      <div className="toolbar-status">
        {imageLoaded ? (
          <>
            <span className="file-name">{fileName}</span>
            <span className="image-size">{imageWidth} x {imageHeight}</span>
            <span className="follicle-count">{follicles.length} follicles</span>
          </>
        ) : (
          <span className="no-image">No image loaded</span>
        )}
      </div>
    </div>
  );
};
