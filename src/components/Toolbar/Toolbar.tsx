import React, { useEffect, useCallback, useRef, useState } from "react";
import {
  Pencil,
  MousePointer2,
  Hand,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  HelpCircle,
  Eye,
  EyeOff,
  Tag,
  ImagePlus,
  BoxSelect,
  Lasso,
  Sparkles,
  Loader2,
  FolderOpen,
  Save,
  Flame,
  BarChart3,
  Settings,
  GraduationCap,
  Brain,
  Database,
  FileUp,
  Scan,
  FolderCog,
} from "lucide-react";
import { useCanvasStore } from "../../store/canvasStore";
import { useProjectStore, generateImageId } from "../../store/projectStore";
import { useFollicleStore, useTemporalStore } from "../../store/follicleStore";
import {
  generateExportV2,
  parseImportV2,
  exportYOLODataset,
  exportToCSV,
  exportYOLOKeypointDatasetZip,
  exportSelectedAnnotationsJSON,
  importAnnotationsFromJSONWithDuplicateCheck,
} from "../../utils/export-utils";
import {
  extractAllFolliclesToZip,
  extractSelectedFolliclesToZip,
  extractImageFolliclesToZip,
  downloadBlob,
} from "../../utils/follicle-extract";
import { blobService } from "../../services/blobService";
import {
  DetectionSettingsDialog,
  DetectionSettings,
  DEFAULT_DETECTION_SETTINGS,
  GPUInstallState,
} from "../DetectionSettingsDialog/DetectionSettingsDialog";
import {
  LearnedDetectionDialog,
  LearnedDetectionSettings,
  DEFAULT_LEARNED_SETTINGS,
} from "../LearnedDetectionDialog/LearnedDetectionDialog";
import { ExportMenu, ExportType } from "../ExportMenu/ExportMenu";
import { ShapeToolDropdown } from "../ShapeToolDropdown/ShapeToolDropdown";
import { ThemePicker } from "../ThemePicker/ThemePicker";
import { YOLOTrainingDialog } from "../YOLOTrainingDialog";
import { YOLOModelManager } from "../YOLOModelManager";
import { YOLODetectionTrainingDialog } from "../YOLODetectionTrainingDialog";
import { YOLODetectionModelManager } from "../YOLODetectionModelManager";
import {
  ImportAnnotationsDialog,
  ImportAnalysis,
  ImportOptions,
} from "../ImportAnnotationsDialog/ImportAnnotationsDialog";
import { ProjectImage, RectangleAnnotation, FollicleOrigin, DetectionPrediction } from "../../types";
import type { BlobDetection } from "../../services/blobService";
import { yoloDetectionService } from "../../services/yoloDetectionService";
import { generateId } from "../../utils/id-generator";

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
      className={`icon-button ${active ? "active" : ""}`}
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
  const mode = useCanvasStore((state) => state.mode);
  const setMode = useCanvasStore((state) => state.setMode);
  const showLabels = useCanvasStore((state) => state.showLabels);
  const toggleLabels = useCanvasStore((state) => state.toggleLabels);
  const showShapes = useCanvasStore((state) => state.showShapes);
  const toggleShapes = useCanvasStore((state) => state.toggleShapes);
  const selectionToolType = useCanvasStore((state) => state.selectionToolType);
  const setSelectionToolType = useCanvasStore(
    (state) => state.setSelectionToolType,
  );
  const showHelp = useCanvasStore((state) => state.showHelp);
  const toggleHelp = useCanvasStore((state) => state.toggleHelp);
  const showHeatmap = useCanvasStore((state) => state.showHeatmap);
  const toggleHeatmap = useCanvasStore((state) => state.toggleHeatmap);
  const showStatistics = useCanvasStore((state) => state.showStatistics);
  const toggleStatistics = useCanvasStore((state) => state.toggleStatistics);

  // Project store for multi-image support
  const images = useProjectStore((state) => state.images);
  const imageOrder = useProjectStore((state) => state.imageOrder);
  const activeImageId = useProjectStore((state) => state.activeImageId);
  const addImage = useProjectStore((state) => state.addImage);
  const clearProject = useProjectStore((state) => state.clearProject);
  const zoom = useProjectStore((state) => state.zoom);
  const resetZoom = useProjectStore((state) => state.resetZoom);
  const currentProjectPath = useProjectStore(
    (state) => state.currentProjectPath,
  );
  const setCurrentProjectPath = useProjectStore(
    (state) => state.setCurrentProjectPath,
  );
  const isDirty = useProjectStore((state) => state.isDirty);
  const setDirty = useProjectStore((state) => state.setDirty);
  const markClean = useProjectStore((state) => state.markClean);

  // Get active image info
  const activeImage = activeImageId ? images.get(activeImageId) : null;
  const imageLoaded = activeImage !== null;
  const viewport = activeImage?.viewport ?? {
    offsetX: 0,
    offsetY: 0,
    scale: 1,
  };

  const follicles = useFollicleStore((state) => state.follicles);
  const selectedIds = useFollicleStore((state) => state.selectedIds);
  const importFollicles = useFollicleStore((state) => state.importFollicles);
  const updateFollicle = useFollicleStore((state) => state.updateFollicle);
  const clearAll = useFollicleStore((state) => state.clearAll);

  const temporalStore = useTemporalStore();

  // State for auto-detection loading
  const [isDetecting, setIsDetecting] = useState(false);

  // State for BLOB server
  const [blobServerConnected, setBlobServerConnected] = useState(false);
  const [blobSessionId, setBlobSessionId] = useState<string | null>(null);
  const [annotationCount, setAnnotationCount] = useState(0);
  const [canDetect, setCanDetect] = useState(false);
  const [serverStarting, setServerStarting] = useState(false);
  const [setupStatus, setSetupStatus] = useState<string>("");
  const MIN_ANNOTATIONS = 3;

  // Refs to prevent duplicate operations
  const isCreatingSession = useRef(false);
  const isSyncingAnnotations = useRef(false);
  const lastSessionImageId = useRef<string | null>(null);

  // State for detection settings dialog (manual settings mode)
  const [detectionSettings, setDetectionSettings] = useState<DetectionSettings>(
    DEFAULT_DETECTION_SETTINGS,
  );
  const [showDetectionSettings, setShowDetectionSettings] = useState(false);

  // Handler for when the BLOB server restarts (e.g., after GPU install)
  // Clears session state to force recreation
  const handleServerRestarted = () => {
    setBlobSessionId(null);
    lastSessionImageId.current = null;
    setAnnotationCount(0);
    setCanDetect(false);
  };

  // State for learned detection dialog
  const [learnedSettings, setLearnedSettings] =
    useState<LearnedDetectionSettings>(DEFAULT_LEARNED_SETTINGS);
  const [showLearnedDetection, setShowLearnedDetection] = useState(false);
  const [isLearnedDetecting, setIsLearnedDetecting] = useState(false);

  // State for GPU package installation (persisted across dialog open/close)
  const [gpuInstallState, setGpuInstallState] = useState<GPUInstallState>({
    isInstalling: false,
    progress: '',
    error: null,
  });

  // State for YOLO training dialogs
  const [showYOLOTraining, setShowYOLOTraining] = useState(false);
  const [showYOLOModelManager, setShowYOLOModelManager] = useState(false);
  const [showYOLODetectionTraining, setShowYOLODetectionTraining] = useState(false);
  const [showYOLODetectionModelManager, setShowYOLODetectionModelManager] = useState(false);
  const [showImportDialog, setShowImportDialog] = useState(false);
  const [importAnalysis, setImportAnalysis] = useState<ImportAnalysis | null>(null);

  // Get annotation count for active image only
  const activeImageAnnotationCount = activeImageId
    ? follicles.filter((f) => f.imageId === activeImageId).length
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

  // Ref to prevent duplicate server starts (React StrictMode runs effects twice)
  const serverStartAttempted = useRef(false);

  // Start BLOB server on mount
  useEffect(() => {
    // Prevent duplicate starts from StrictMode
    if (serverStartAttempted.current) return;
    serverStartAttempted.current = true;

    // Listen for setup progress events
    const cleanupProgress = window.electronAPI.blob.onSetupProgress(
      (status) => {
        setSetupStatus(status);
      }
    );

    const startServer = async () => {
      setServerStarting(true);
      setSetupStatus("Checking server status...");
      try {
        // Check if server is already running
        const isRunning = await blobService.isAvailable();
        if (isRunning) {
          setBlobServerConnected(true);
          setServerStarting(false);
          setSetupStatus("");
          return;
        }

        // Start the server via Electron IPC
        const result = await window.electronAPI.blob.startServer();
        if (result.success) {
          // Wait a bit for server to be fully ready
          await new Promise((resolve) => setTimeout(resolve, 1000));
          const available = await blobService.isAvailable();
          setBlobServerConnected(available);
          setSetupStatus("");
        } else {
          console.error("Failed to start BLOB server:", result.error);
          setBlobServerConnected(false);
          setSetupStatus(`Error: ${result.error}`);
        }
      } catch (error) {
        console.error("Error starting BLOB server:", error);
        setBlobServerConnected(false);
        setSetupStatus(`Error: ${error instanceof Error ? error.message : "Unknown error"}`);
      } finally {
        setServerStarting(false);
      }
    };

    startServer();

    // Cleanup on unmount
    return () => {
      cleanupProgress();
      if (blobSessionId) {
        blobService.clearSession(blobSessionId);
      }
    };
  }, []);

  // Create session when active image changes
  useEffect(() => {
    const createSession = async () => {
      // Skip if no image, server not connected, or already creating
      if (!activeImage || !blobServerConnected) {
        setBlobSessionId(null);
        setAnnotationCount(0);
        setCanDetect(false);
        lastSessionImageId.current = null;
        return;
      }

      // Skip if we already have a session for this image
      if (lastSessionImageId.current === activeImageId && blobSessionId) {
        return;
      }

      // Prevent concurrent session creation
      if (isCreatingSession.current) {
        return;
      }

      isCreatingSession.current = true;

      try {
        // Clear previous session
        if (blobSessionId) {
          await blobService.clearSession(blobSessionId);
        }

        // Create new session with image
        const result = await blobService.setImage(activeImage.imageData);
        setBlobSessionId(result.sessionId);
        lastSessionImageId.current = activeImageId;
        setAnnotationCount(0);
        setCanDetect(false);

        // Sync existing annotations for this image
        const imageAnnotations = follicles.filter(
          (f) => f.imageId === activeImageId,
        );
        if (imageAnnotations.length > 0) {
          const boxes = imageAnnotations
            .map((ann) => {
              if (ann.shape === "rectangle") {
                return {
                  x: ann.x,
                  y: ann.y,
                  width: ann.width,
                  height: ann.height,
                };
              } else if (ann.shape === "circle") {
                const r = ann.radius;
                return {
                  x: ann.center.x - r,
                  y: ann.center.y - r,
                  width: r * 2,
                  height: r * 2,
                };
              }
              return null;
            })
            .filter(
              (
                b,
              ): b is { x: number; y: number; width: number; height: number } =>
                b !== null,
            );

          if (boxes.length > 0) {
            const syncResult = await blobService.syncAnnotations(
              result.sessionId,
              boxes,
            );
            setAnnotationCount(syncResult.annotationCount);
            setCanDetect(syncResult.canDetect);
          }
        }
      } catch (error) {
        console.error("Failed to create BLOB session:", error);
        setBlobSessionId(null);
        lastSessionImageId.current = null;
      } finally {
        isCreatingSession.current = false;
      }
    };

    createSession();
  }, [activeImageId, blobServerConnected]); // Only depend on ID, not full image object

  // Get annotation count for the active image to use as a stable dependency
  const activeImageFollicleCount = follicles.filter(
    (f) => f.imageId === activeImageId,
  ).length;

  // Sync annotations when follicles change for active image (debounced)
  useEffect(() => {
    if (!blobSessionId || !activeImageId || !blobServerConnected) return;

    // Prevent concurrent syncs
    if (isSyncingAnnotations.current) return;

    // Debounce sync to avoid rapid calls
    const timeoutId = setTimeout(async () => {
      if (isSyncingAnnotations.current) return;
      isSyncingAnnotations.current = true;

      try {
        const imageAnnotations = follicles.filter(
          (f) => f.imageId === activeImageId,
        );
        const boxes = imageAnnotations
          .map((ann) => {
            if (ann.shape === "rectangle") {
              return {
                x: ann.x,
                y: ann.y,
                width: ann.width,
                height: ann.height,
              };
            } else if (ann.shape === "circle") {
              const r = ann.radius;
              return {
                x: ann.center.x - r,
                y: ann.center.y - r,
                width: r * 2,
                height: r * 2,
              };
            }
            return null;
          })
          .filter(
            (b): b is { x: number; y: number; width: number; height: number } =>
              b !== null,
          );

        const syncResult = await blobService.syncAnnotations(
          blobSessionId,
          boxes,
        );
        setAnnotationCount(syncResult.annotationCount);
        setCanDetect(syncResult.canDetect);
      } catch (error) {
        console.error("Failed to sync annotations:", error);
      } finally {
        isSyncingAnnotations.current = false;
      }
    }, 300); // 300ms debounce

    return () => clearTimeout(timeoutId);
  }, [
    activeImageFollicleCount,
    activeImageId,
    blobSessionId,
    blobServerConnected,
  ]);

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
      console.error("Failed to open image:", error);
    }
  }, [addImage, imageOrder.length]);

  const handleSave = useCallback(async (): Promise<boolean> => {
    if (images.size === 0) return false;

    try {
      const { manifest, annotations, imageList } = generateExportV2(
        Array.from(images.values()),
        follicles,
      );

      let result: { success: boolean; filePath?: string };

      if (currentProjectPath) {
        // Silent save to existing path
        result = await window.electronAPI.saveProjectV2ToPath(
          currentProjectPath,
          imageList,
          JSON.stringify(manifest, null, 2),
          JSON.stringify(annotations, null, 2),
        );
      } else {
        // Show save dialog
        result = await window.electronAPI.saveProjectV2(
          imageList,
          JSON.stringify(manifest, null, 2),
          JSON.stringify(annotations, null, 2),
        );
      }

      if (result.success && result.filePath) {
        setCurrentProjectPath(result.filePath);
        markClean();
        console.log("Project saved successfully to:", result.filePath);
      }
      return result.success;
    } catch (error) {
      console.error("Failed to save project:", error);
      return false;
    }
  }, [images, follicles, currentProjectPath, setCurrentProjectPath, markClean]);

  const handleSaveAs = useCallback(async (): Promise<boolean> => {
    if (images.size === 0) return false;

    try {
      const { manifest, annotations, imageList } = generateExportV2(
        Array.from(images.values()),
        follicles,
      );

      // Always show save dialog
      const result = await window.electronAPI.saveProjectV2(
        imageList,
        JSON.stringify(manifest, null, 2),
        JSON.stringify(annotations, null, 2),
        currentProjectPath || undefined,
      );

      if (result.success && result.filePath) {
        setCurrentProjectPath(result.filePath);
        markClean();
        console.log("Project saved successfully to:", result.filePath);
      }
      return result.success;
    } catch (error) {
      console.error("Failed to save project:", error);
      return false;
    }
  }, [images, follicles, currentProjectPath, setCurrentProjectPath, markClean]);

  // Check for unsaved changes and prompt user
  // Returns true if safe to proceed, false if user cancelled
  const checkUnsavedChanges = useCallback(async (): Promise<boolean> => {
    if (!isDirty) return true;

    const response = await window.electronAPI.showUnsavedChangesDialog();

    if (response === "save") {
      const saved = await handleSave();
      return saved;
    } else if (response === "discard") {
      return true;
    } else {
      // Cancel
      return false;
    }
  }, [isDirty, handleSave]);

  // Load project from parsed result (shared logic)
  const loadProjectFromResult = useCallback(
    async (
      result: Awaited<ReturnType<typeof window.electronAPI.loadProjectV2>>,
    ) => {
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

      // Mark as clean after loading - use setTimeout to ensure it runs after
      // dirty-tracking effects have fired (addImage sets isDirty, useEffect also sets it)
      setTimeout(markClean, 0);
    },
    [
      clearProject,
      clearAll,
      addImage,
      importFollicles,
      setCurrentProjectPath,
      markClean,
    ],
  );

  const handleLoad = useCallback(async () => {
    // Check for unsaved changes first
    const canProceed = await checkUnsavedChanges();
    if (!canProceed) return;

    try {
      const result = await window.electronAPI.loadProjectV2();
      await loadProjectFromResult(result);
    } catch (error) {
      console.error("Failed to load project:", error);
      alert("Failed to load project file. Please check the file format.");
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
        ? follicles.filter((f) => f.imageId === activeImageId).length
        : 0;
      const totalCount = follicles.length;

      // Generate base filename
      const baseName = currentProjectPath
        ? currentProjectPath
            .replace(/\.[^/.]+$/, "")
            .split(/[/\\]/)
            .pop()
        : (activeImage?.fileName.replace(/\.[^/.]+$/, "") ?? "follicles");

      let zipBlob: Blob;
      let suffix: string;

      // If there's a selection, show options dialog
      if (selectedCount > 0) {
        const choice = await window.electronAPI.showDownloadOptionsDialog(
          selectedCount,
          currentImageCount,
          totalCount,
        );

        if (choice === "cancel") return;

        if (choice === "selected") {
          zipBlob = await extractSelectedFolliclesToZip(
            images,
            follicles,
            selectedIds,
          );
          suffix = "_selected";
        } else if (choice === "currentImage" && activeImage) {
          zipBlob = await extractImageFolliclesToZip(activeImage, follicles);
          suffix = `_${activeImage.fileName.replace(/\.[^/.]+$/, "")}`;
        } else {
          // 'all'
          zipBlob = await extractAllFolliclesToZip(images, follicles);
          suffix = "_all";
        }
      } else {
        // No selection - download all
        zipBlob = await extractAllFolliclesToZip(images, follicles);
        suffix = "_follicles";
      }

      downloadBlob(zipBlob, `${baseName}${suffix}.zip`);
    } catch (error) {
      console.error("Failed to extract follicles:", error);
      alert(
        "Failed to extract follicle images. Please ensure there are annotations to extract.",
      );
    }
  }, [
    images,
    follicles,
    selectedIds,
    activeImageId,
    activeImage,
    currentProjectPath,
  ]);

  // Export to YOLO dataset format
  const handleExportYOLO = useCallback(async () => {
    if (images.size === 0 || follicles.length === 0) return;

    try {
      const { files, dataYaml } = exportYOLODataset(
        Array.from(images.values()),
        follicles,
      );

      // Create a ZIP file with YOLO dataset structure
      const JSZip = (await import("jszip")).default;
      const zip = new JSZip();

      // Add data.yaml
      zip.file("data.yaml", dataYaml);

      // Add images and labels folders
      const imagesFolder = zip.folder("images");
      const labelsFolder = zip.folder("labels");

      for (const file of files) {
        imagesFolder?.file(file.imageName, file.imageData);
        labelsFolder?.file(file.labelName, file.labelContent);
      }

      // Generate and download ZIP
      const zipBlob = await zip.generateAsync({ type: "blob" });
      const baseName = currentProjectPath
        ? currentProjectPath
            .replace(/\.[^/.]+$/, "")
            .split(/[/\\]/)
            .pop()
        : "yolo_dataset";
      downloadBlob(zipBlob, `${baseName}_yolo.zip`);

      console.log(`Exported YOLO dataset with ${files.length} images`);
    } catch (error) {
      console.error("Failed to export YOLO dataset:", error);
      alert("Failed to export YOLO dataset. Please try again.");
    }
  }, [images, follicles, currentProjectPath]);

  // Export to CSV format
  const handleExportCSV = useCallback(() => {
    if (images.size === 0 || follicles.length === 0) return;

    try {
      // Combine CSV for all images (using first image dimensions as reference,
      // normalized coordinates will still be correct per-annotation)
      const allImages = Array.from(images.values());
      let csvContent = "";
      let isFirst = true;

      for (const image of allImages) {
        const imageFollicles = follicles.filter((f) => f.imageId === image.id);
        if (imageFollicles.length === 0) continue;

        const csv = exportToCSV(imageFollicles, image.width, image.height);
        if (isFirst) {
          csvContent = csv;
          isFirst = false;
        } else {
          // Skip header row for subsequent images
          const lines = csv.split("\n");
          csvContent += "\n" + lines.slice(1).join("\n");
        }
      }

      // Create and download blob
      const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8" });
      const baseName = currentProjectPath
        ? currentProjectPath
            .replace(/\.[^/.]+$/, "")
            .split(/[/\\]/)
            .pop()
        : "annotations";
      downloadBlob(blob, `${baseName}_annotations.csv`);

      console.log(`Exported ${follicles.length} annotations to CSV`);
    } catch (error) {
      console.error("Failed to export CSV:", error);
      alert("Failed to export CSV. Please try again.");
    }
  }, [images, follicles, currentProjectPath]);

  // Export selected annotations as JSON
  const handleExportSelectedJSON = useCallback(() => {
    if (selectedIds.size === 0) {
      alert("No annotations selected. Please select annotations to export.");
      return;
    }

    if (!activeImage) {
      alert("No image loaded.");
      return;
    }

    try {
      const selectedFollicles = follicles.filter(f => selectedIds.has(f.id));
      const json = exportSelectedAnnotationsJSON(
        selectedFollicles,
        selectedIds,
        { width: activeImage.width, height: activeImage.height }
      );

      // Create and download JSON file
      const blob = new Blob([json], { type: "application/json" });
      const baseName = activeImage.fileName.replace(/\.[^/.]+$/, "");
      downloadBlob(blob, `${baseName}_selected_annotations.json`);

      console.log(`Exported ${selectedIds.size} selected annotations`);
    } catch (error) {
      console.error("Failed to export selected annotations:", error);
      alert("Failed to export selected annotations. Please try again.");
    }
  }, [selectedIds, follicles, activeImage]);

  // Import annotations from JSON - opens file dialog and shows import dialog
  const handleImportAnnotations = useCallback(async () => {
    if (!activeImageId || !activeImage) {
      alert("Please load an image first before importing annotations.");
      return;
    }

    try {
      // Open file dialog for JSON
      const result = await window.electronAPI.openFileDialog({
        filters: [{ name: "JSON Files", extensions: ["json"] }],
        title: "Import Annotations",
      });

      if (!result) return;

      const text = new TextDecoder().decode(result.data);
      const analysis = importAnnotationsFromJSONWithDuplicateCheck(text, activeImageId, follicles);

      if (analysis.totalImported === 0) {
        alert("No annotations found in the file.");
        return;
      }

      // Show the import dialog
      setImportAnalysis(analysis);
      setShowImportDialog(true);
    } catch (error) {
      console.error("Failed to import annotations:", error);
      alert(`Failed to import annotations: ${error instanceof Error ? error.message : "Unknown error"}`);
    }
  }, [activeImageId, activeImage, follicles]);

  // Handle import confirmation from dialog
  const handleImportConfirm = useCallback((options: ImportOptions) => {
    if (!importAnalysis) return;

    const { newAnnotations, augmentable, alreadyAugmented, duplicates } = importAnalysis;

    // Update existing annotations with origin data
    if (options.updateExisting && augmentable.length > 0) {
      for (const { imported, existingId } of augmentable) {
        updateFollicle(existingId, { origin: imported.origin });
      }
      console.log(`Updated ${augmentable.length} existing annotations with origin data`);
    }

    // Build list of annotations to import
    const annotationsToImport: typeof newAnnotations = [];

    if (options.importNew) {
      annotationsToImport.push(...newAnnotations);
    }
    if (options.importAlreadyAugmented) {
      annotationsToImport.push(...alreadyAugmented);
    }
    if (options.importDuplicates) {
      annotationsToImport.push(...duplicates);
    }

    // Import new annotations
    if (annotationsToImport.length > 0) {
      const allFollicles = [...follicles, ...annotationsToImport];
      importFollicles(allFollicles);
      console.log(`Imported ${annotationsToImport.length} new annotations`);
    }

    // Close dialog
    setShowImportDialog(false);
    setImportAnalysis(null);
  }, [importAnalysis, follicles, importFollicles, updateFollicle]);

  // Handle import cancel
  const handleImportCancel = useCallback(() => {
    setShowImportDialog(false);
    setImportAnalysis(null);
  }, []);

  // Export to YOLO Keypoint dataset format (for pose/keypoint training)
  const handleExportYOLOKeypoint = useCallback(async () => {
    if (images.size === 0 || follicles.length === 0) return;

    try {
      const { blob, stats } = await exportYOLOKeypointDatasetZip(
        images,
        follicles
      );

      if (stats.annotationsWithOrigin === 0) {
        alert(
          "No annotations with origin data found. Please set follicle origins (entry point and direction) before exporting the keypoint dataset."
        );
        return;
      }

      // Download the ZIP file
      const baseName = currentProjectPath
        ? currentProjectPath
            .replace(/\.[^/.]+$/, "")
            .split(/[/\\]/)
            .pop()
        : "yolo_keypoint_dataset";
      downloadBlob(blob, `${baseName}_keypoint.zip`);

      console.log(
        `Exported YOLO Keypoint dataset: ${stats.trainCount} train, ${stats.valCount} val (${stats.skippedNoOrigin} skipped without origin)`
      );
    } catch (error) {
      console.error("Failed to export YOLO Keypoint dataset:", error);
      alert("Failed to export YOLO Keypoint dataset. Please try again.");
    }
  }, [images, follicles, currentProjectPath]);

  // Handle export menu selection
  const handleExport = useCallback(
    (type: ExportType) => {
      switch (type) {
        case "images":
          handleDownloadFollicles();
          break;
        case "yolo":
          handleExportYOLO();
          break;
        case "yolo-keypoint":
          handleExportYOLOKeypoint();
          break;
        case "csv":
          handleExportCSV();
          break;
        case "selected-json":
          handleExportSelectedJSON();
          break;
      }
    },
    [handleDownloadFollicles, handleExportYOLO, handleExportYOLOKeypoint, handleExportCSV, handleExportSelectedJSON],
  );

  // Colors for auto-detected annotations (cycles through)
  const ANNOTATION_COLORS = [
    "#FF6B6B",
    "#4ECDC4",
    "#45B7D1",
    "#96CEB4",
    "#FFEAA7",
    "#DDA0DD",
    "#98D8C8",
    "#F7DC6F",
    "#74B9FF",
    "#A29BFE",
    "#FD79A8",
    "#00CEC9",
  ];

  // Helper to get image as base64
  const getImageBase64 = useCallback(async (image: ProjectImage): Promise<string> => {
    // Convert ArrayBuffer to base64
    const bytes = new Uint8Array(image.imageData);
    let binary = '';
    for (let i = 0; i < bytes.byteLength; i++) {
      binary += String.fromCharCode(bytes[i]);
    }
    const base64 = btoa(binary);

    // Determine mime type from filename
    const ext = image.fileName.toLowerCase().split('.').pop();
    let mimeType = 'image/jpeg';
    if (ext === 'png') mimeType = 'image/png';
    else if (ext === 'webp') mimeType = 'image/webp';
    else if (ext === 'tiff' || ext === 'tif') mimeType = 'image/tiff';
    else if (ext === 'bmp') mimeType = 'image/bmp';

    return `data:${mimeType};base64,${base64}`;
  }, []);

  // Auto-detect follicles using YOLO detection
  const handleYoloDetect = useCallback(async () => {
    if (!activeImage || !activeImageId || isDetecting) return;

    setIsDetecting(true);

    try {
      // Load model if needed
      if (detectionSettings.yoloModelId) {
        // Load custom trained model
        const models = await yoloDetectionService.listModels();
        const selectedModel = models.find(m => m.id === detectionSettings.yoloModelId);
        if (selectedModel) {
          const loaded = await yoloDetectionService.loadModel(selectedModel.path);
          if (!loaded) {
            throw new Error('Failed to load selected YOLO model');
          }
        }
      }
      // If no model ID is set, the pre-trained model will be downloaded/used automatically

      // Get image as base64
      const imageBase64 = await getImageBase64(activeImage);

      // Run YOLO detection
      const predictions = await yoloDetectionService.predict(
        imageBase64,
        detectionSettings.yoloConfidenceThreshold
      );

      if (predictions.length === 0) {
        console.log("No follicles detected by YOLO");
        alert("No follicles detected. Try lowering the confidence threshold.");
        setIsDetecting(false);
        return;
      }

      // Convert predictions to RECTANGLE annotations
      const existingCount = follicles.filter(
        (f) => f.imageId === activeImageId,
      ).length;
      const now = Date.now();

      const newFollicles: RectangleAnnotation[] = predictions.map(
        (detection: DetectionPrediction, i: number) => ({
          id: generateId(),
          imageId: activeImageId,
          shape: "rectangle" as const,
          x: detection.x,
          y: detection.y,
          width: detection.width,
          height: detection.height,
          label: `YOLO ${existingCount + i + 1}`,
          notes: `YOLO detection (conf: ${(detection.confidence * 100).toFixed(0)}%)`,
          color: ANNOTATION_COLORS[(existingCount + i) % ANNOTATION_COLORS.length],
          createdAt: now,
          updatedAt: now,
        }),
      );

      // Import all at once (supports undo as single action)
      const allFollicles = [...follicles, ...newFollicles];
      importFollicles(allFollicles);

      console.log(`YOLO detected ${predictions.length} follicles`);
    } catch (error) {
      console.error("YOLO detection failed:", error);
      alert(
        `YOLO detection failed: ${error instanceof Error ? error.message : "Unknown error"}`,
      );
    } finally {
      setIsDetecting(false);
    }
  }, [
    activeImage,
    activeImageId,
    isDetecting,
    follicles,
    importFollicles,
    detectionSettings.yoloModelId,
    detectionSettings.yoloConfidenceThreshold,
    getImageBase64,
  ]);

  // Auto-detect follicles using manual settings (blob detection)
  const handleBlobDetect = useCallback(async () => {
    if (!activeImage || !activeImageId || isDetecting) return;

    // Check if server is connected
    if (!blobServerConnected || !blobSessionId) {
      alert(
        "BLOB detection server is not connected. Please wait for it to start.",
      );
      return;
    }

    // Check if manual settings are configured
    if (!(detectionSettings.minWidth > 0 && detectionSettings.maxWidth > 0)) {
      alert("Please configure size settings in Detection Settings first.");
      return;
    }

    setIsDetecting(true);

    try {
      const detectSettings = {
        minWidth: detectionSettings.minWidth,
        maxWidth: detectionSettings.maxWidth,
        minHeight: detectionSettings.minHeight,
        maxHeight: detectionSettings.maxHeight,
        darkBlobs: detectionSettings.darkBlobs,
        useCLAHE: detectionSettings.useCLAHE,
        claheClipLimit: detectionSettings.claheClipLimit,
        claheTileSize: detectionSettings.claheTileSize,
        forceCPU: detectionSettings.forceCPU,
      };

      // Use detectWithKeypoints if keypoint prediction is enabled
      let detections: BlobDetection[];
      let predictedOrigins: Map<string, FollicleOrigin> | null = null;
      let count = 0;

      if (detectionSettings.useKeypointPrediction) {
        const result = await blobService.detectWithKeypoints(blobSessionId, detectSettings);
        detections = result.detections;
        predictedOrigins = result.origins;
        count = result.count;
      } else {
        const result = await blobService.blobDetect(blobSessionId, detectSettings);
        detections = result.detections;
        count = result.count;
      }

      if (count === 0) {
        console.log("No follicles detected");
        alert("No follicles detected. Try adjusting the detection settings.");
        setIsDetecting(false);
        return;
      }

      // Convert detected blobs to RECTANGLE annotations
      const existingCount = follicles.filter(
        (f) => f.imageId === activeImageId,
      ).length;
      const now = Date.now();

      const newFollicles: RectangleAnnotation[] = detections.map(
        (detection, i) => {
          const detectionId = `det_${i}_${detection.x}_${detection.y}`;
          const origin = predictedOrigins?.get(detectionId);

          return {
            id: generateId(),
            imageId: activeImageId,
            shape: "rectangle" as const,
            x: detection.x,
            y: detection.y,
            width: detection.width,
            height: detection.height,
            label: `Settings ${existingCount + i + 1}`,
            notes: origin
              ? `Detected via ${detection.method} (conf: ${(detection.confidence * 100).toFixed(0)}%) + origin predicted`
              : `Detected via ${detection.method} (conf: ${(detection.confidence * 100).toFixed(0)}%)`,
            color:
              ANNOTATION_COLORS[(existingCount + i) % ANNOTATION_COLORS.length],
            createdAt: now,
            updatedAt: now,
            origin, // Include predicted origin if available
          };
        },
      );

      // Import all at once (supports undo as single action)
      const allFollicles = [...follicles, ...newFollicles];
      importFollicles(allFollicles);

      const originsCount = predictedOrigins?.size ?? 0;
      console.log(`Detected ${count} follicles using manual settings${originsCount > 0 ? ` (${originsCount} with predicted origins)` : ''}`);
    } catch (error) {
      console.error("Failed to detect follicles:", error);
      alert(
        `Failed to detect follicles: ${error instanceof Error ? error.message : "Unknown error"}`,
      );
    } finally {
      setIsDetecting(false);
    }
  }, [
    activeImage,
    activeImageId,
    isDetecting,
    follicles,
    importFollicles,
    blobServerConnected,
    blobSessionId,
    detectionSettings,
  ]);

  // Auto-detect follicles - routes to YOLO or blob based on settings
  const handleSettingsDetect = useCallback(async () => {
    if (detectionSettings.detectionMethod === 'yolo') {
      await handleYoloDetect();
    } else {
      await handleBlobDetect();
    }
  }, [detectionSettings.detectionMethod, handleYoloDetect, handleBlobDetect]);

  // Handle learned detection (from annotations)
  const handleLearnedDetect = useCallback(
    async (settings: LearnedDetectionSettings) => {
      if (!activeImage || !activeImageId || isLearnedDetecting) return;

      // Check if server is connected
      if (!blobServerConnected || !blobSessionId) {
        alert(
          "BLOB detection server is not connected. Please wait for it to start.",
        );
        return;
      }

      // Check if we have enough annotations
      if (!canDetect) {
        alert(
          `Please draw at least ${MIN_ANNOTATIONS} annotations first to learn from.`,
        );
        return;
      }

      setShowLearnedDetection(false);
      setIsLearnedDetecting(true);
      setLearnedSettings(settings);

      try {
        const learnedDetectSettings = {
          useLearnedStats: true,
          tolerance: settings.tolerance,
          darkBlobs: settings.darkBlobs,
          useCLAHE: true,
          claheClipLimit: 3.0,
          claheTileSize: 8,
          forceCPU: detectionSettings.forceCPU,
        };

        // Use detectWithKeypoints if keypoint prediction is enabled
        let detections: BlobDetection[];
        let predictedOrigins: Map<string, FollicleOrigin> | null = null;
        let count = 0;

        if (detectionSettings.useKeypointPrediction) {
          const result = await blobService.detectWithKeypoints(blobSessionId, learnedDetectSettings);
          detections = result.detections;
          predictedOrigins = result.origins;
          count = result.count;
        } else {
          const result = await blobService.blobDetect(blobSessionId, learnedDetectSettings);
          detections = result.detections;
          count = result.count;
        }

        if (count === 0) {
          console.log("No follicles detected");
          alert(
            "No follicles detected. Try adjusting the tolerance or drawing more diverse annotations.",
          );
          setIsLearnedDetecting(false);
          return;
        }

        // Convert detected blobs to RECTANGLE annotations
        const existingCount = follicles.filter(
          (f) => f.imageId === activeImageId,
        ).length;
        const now = Date.now();

        const newFollicles: RectangleAnnotation[] = detections.map(
          (detection, i) => {
            const detectionId = `det_${i}_${detection.x}_${detection.y}`;
            const origin = predictedOrigins?.get(detectionId);

            return {
              id: generateId(),
              imageId: activeImageId,
              shape: "rectangle" as const,
              x: detection.x,
              y: detection.y,
              width: detection.width,
              height: detection.height,
              label: `Learned ${existingCount + i + 1}`,
              notes: origin
                ? `Detected via ${detection.method} (conf: ${(detection.confidence * 100).toFixed(0)}%) + origin predicted`
                : `Detected via ${detection.method} (conf: ${(detection.confidence * 100).toFixed(0)}%)`,
              color:
                ANNOTATION_COLORS[(existingCount + i) % ANNOTATION_COLORS.length],
              createdAt: now,
              updatedAt: now,
              origin, // Include predicted origin if available
            };
          },
        );

        // Import all at once (supports undo as single action)
        const allFollicles = [...follicles, ...newFollicles];
        importFollicles(allFollicles);

        const originsCount = predictedOrigins?.size ?? 0;
        console.log(
          `Detected ${count} follicles using learned settings (tolerance: ${settings.tolerance}%)${originsCount > 0 ? ` (${originsCount} with predicted origins)` : ''}`,
        );
      } catch (error) {
        console.error("Failed to detect follicles:", error);
        alert(
          `Failed to detect follicles: ${error instanceof Error ? error.message : "Unknown error"}`,
        );
      } finally {
        setIsLearnedDetecting(false);
      }
    },
    [
      activeImage,
      activeImageId,
      isLearnedDetecting,
      follicles,
      importFollicles,
      blobServerConnected,
      blobSessionId,
      canDetect,
      detectionSettings,
    ],
  );

  // Open learned detection dialog
  const handleOpenLearnedDetection = useCallback(() => {
    if (!canDetect) {
      alert(
        `Please draw at least ${MIN_ANNOTATIONS} annotations first to learn from.`,
      );
      return;
    }
    setShowLearnedDetection(true);
  }, [canDetect]);

  // Listen for auto-detect trigger from ImageCanvas keyboard shortcut
  // Use learned detection if we have annotations, otherwise settings-based
  useEffect(() => {
    const handler = () => {
      if (canDetect) {
        handleOpenLearnedDetection();
      } else if (
        detectionSettings.minWidth > 0 &&
        detectionSettings.maxWidth > 0
      ) {
        handleSettingsDetect();
      }
    };
    window.addEventListener("triggerAutoDetect", handler);
    return () => window.removeEventListener("triggerAutoDetect", handler);
  }, [
    canDetect,
    handleOpenLearnedDetection,
    handleSettingsDetect,
    detectionSettings,
  ]);

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

    return () => cleanups.forEach((cleanup) => cleanup());
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
        console.error("Failed to load project from file association:", error);
        alert("Failed to load project file. Please check the file format.");
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

    const cleanup = window.electronAPI.onCheckUnsavedChanges(
      handleCheckUnsavedChanges,
    );
    return cleanup;
  }, [checkUnsavedChanges]);

  // Handle system suspend (sleep/hibernate) - auto-save to prevent data loss
  useEffect(() => {
    const handleSystemSuspend = async () => {
      // Only auto-save if there's a project with unsaved changes AND an existing save path
      // (we can't show a dialog during suspend - no time for user interaction)
      if (isDirty && currentProjectPath && images.size > 0) {
        console.log(
          "System suspending - auto-saving project to:",
          currentProjectPath,
        );
        try {
          const { manifest, annotations, imageList } = generateExportV2(
            Array.from(images.values()),
            follicles,
          );

          const result = await window.electronAPI.saveProjectV2ToPath(
            currentProjectPath,
            imageList,
            JSON.stringify(manifest, null, 2),
            JSON.stringify(annotations, null, 2),
          );

          if (result.success) {
            markClean();
            console.log("Auto-save before suspend completed successfully");
          }
        } catch (error) {
          console.error("Auto-save before suspend failed:", error);
        }
      }
    };

    const cleanup = window.electronAPI.onSystemSuspend(handleSystemSuspend);
    return cleanup;
  }, [isDirty, currentProjectPath, images, follicles, markClean]);

  const zoomPercent = Math.round(viewport.scale * 100);

  return (
    <div className="toolbar">
      {/* File Operations */}
      <div className="toolbar-group" role="group" aria-label="File operations">
        <IconButton
          icon={<ImagePlus size={18} />}
          tooltip="Add Image"
          shortcut="Ctrl+O"
          onClick={handleOpenImage}
        />
        <IconButton
          icon={<FolderOpen size={18} />}
          tooltip="Open Project"
          shortcut="Ctrl+Shift+O"
          onClick={handleLoad}
        />
        <IconButton
          icon={<Save size={18} />}
          tooltip="Save Project"
          shortcut="Ctrl+S"
          onClick={handleSave}
          disabled={!isDirty || images.size === 0}
        />
      </div>

      <div className="toolbar-divider" />

      {/* Mode tools */}
      <div
        className="toolbar-group"
        role="group"
        aria-label="Interaction modes"
      >
        <IconButton
          icon={<Pencil size={18} />}
          tooltip="Create Mode"
          shortcut="C"
          onClick={() => setMode("create")}
          active={mode === "create"}
        />
        <IconButton
          icon={<MousePointer2 size={18} />}
          tooltip="Select Mode"
          shortcut="V"
          onClick={() => setMode("select")}
          active={mode === "select"}
        />
        <IconButton
          icon={<Hand size={18} />}
          tooltip="Pan Mode"
          shortcut="H"
          onClick={() => setMode("pan")}
          active={mode === "pan"}
        />
      </div>

      <div className="toolbar-divider" />

      {/* Selection tools (visible only in select mode) */}
      {mode === "select" && (
        <>
          <div
            className="toolbar-group"
            role="group"
            aria-label="Selection tools"
          >
            <IconButton
              icon={<BoxSelect size={18} />}
              tooltip="Marquee Select"
              shortcut="M"
              onClick={() => setSelectionToolType("marquee")}
              active={selectionToolType === "marquee"}
            />
            <IconButton
              icon={<Lasso size={18} />}
              tooltip="Lasso Select"
              shortcut="F"
              onClick={() => setSelectionToolType("lasso")}
              active={selectionToolType === "lasso"}
            />
          </div>
          <div className="toolbar-divider" />
        </>
      )}

      {/* Shape tools (visible only in create mode) */}
      {mode === "create" && (
        <>
          <div className="toolbar-group" role="group" aria-label="Shape types">
            <ShapeToolDropdown />
          </div>
          <div className="toolbar-divider" />
        </>
      )}

      {/* Detection Tools - Learn from annotations or use manual settings */}
      <div className="toolbar-group" role="group" aria-label="Detection tools">
        <IconButton
          icon={
            isLearnedDetecting || serverStarting ? (
              <Loader2 size={18} className="animate-spin" />
            ) : (
              <GraduationCap size={18} />
            )
          }
          tooltip={
            !blobServerConnected
              ? "Starting detection server..."
              : !canDetect
                ? `Draw ${MIN_ANNOTATIONS - annotationCount} more annotation${MIN_ANNOTATIONS - annotationCount === 1 ? "" : "s"} to enable learning`
                : "Learn from Selection"
          }
          shortcut="L"
          onClick={handleOpenLearnedDetection}
          disabled={
            !imageLoaded ||
            isLearnedDetecting ||
            serverStarting ||
            !blobServerConnected ||
            !canDetect
          }
        />
        <IconButton
          icon={
            isDetecting || serverStarting ? (
              <Loader2 size={18} className="animate-spin" />
            ) : (
              <Sparkles size={18} />
            )
          }
          tooltip={
            !blobServerConnected
              ? "Starting detection server..."
              : detectionSettings.minWidth > 0 && detectionSettings.maxWidth > 0
                ? "Auto Detect"
                : "Configure settings first"
          }
          shortcut="D"
          onClick={handleSettingsDetect}
          disabled={
            !imageLoaded ||
            isDetecting ||
            serverStarting ||
            !blobServerConnected ||
            !(detectionSettings.minWidth > 0 && detectionSettings.maxWidth > 0)
          }
        />
        <IconButton
          icon={<Settings size={18} />}
          tooltip="Detection Settings"
          onClick={() => setShowDetectionSettings(true)}
          disabled={!imageLoaded}
        />
      </div>

      <div className="toolbar-divider" />

      {/* YOLO Keypoint Training controls */}
      <div className="toolbar-group" role="group" aria-label="YOLO Keypoint Training">
        <IconButton
          icon={<Brain size={18} />}
          tooltip="YOLO Keypoint Training"
          onClick={() => setShowYOLOTraining(true)}
        />
        <IconButton
          icon={<Database size={18} />}
          tooltip="Manage Keypoint Models"
          onClick={() => setShowYOLOModelManager(true)}
        />
      </div>

      <div className="toolbar-divider" />

      {/* YOLO Detection Training controls */}
      <div className="toolbar-group" role="group" aria-label="YOLO Detection Training">
        <IconButton
          icon={<Scan size={18} />}
          tooltip="YOLO Detection Training"
          onClick={() => setShowYOLODetectionTraining(true)}
        />
        <IconButton
          icon={<FolderCog size={18} />}
          tooltip="Manage Detection Models"
          onClick={() => setShowYOLODetectionModelManager(true)}
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
        <IconButton
          icon={<Flame size={18} />}
          tooltip="Toggle Heatmap"
          shortcut="H"
          onClick={toggleHeatmap}
          disabled={!imageLoaded}
          active={showHeatmap}
        />
        <IconButton
          icon={<BarChart3 size={18} />}
          tooltip="Toggle Statistics Panel"
          shortcut="S"
          onClick={toggleStatistics}
          disabled={!imageLoaded}
          active={showStatistics}
        />
      </div>

      <div className="toolbar-divider" />

      {/* Import/Export options */}
      <div className="toolbar-group" role="group" aria-label="Import/Export">
        <IconButton
          icon={<FileUp size={18} />}
          tooltip="Import Annotations"
          onClick={handleImportAnnotations}
          disabled={!imageLoaded}
        />
        <ExportMenu
          onExport={handleExport}
          disabled={!imageLoaded || follicles.length === 0}
          hasSelection={selectedIds.size > 0}
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

      <div className="toolbar-divider" />

      {/* Theme Picker */}
      <div className="toolbar-group" role="group" aria-label="Theme">
        <ThemePicker />
      </div>

      {/* Status display */}
      <div className="toolbar-spacer" />
      <div className="toolbar-status">
        {serverStarting && setupStatus ? (
          <span className="setup-status">{setupStatus}</span>
        ) : imageLoaded && activeImage ? (
          <>
            {images.size > 1 && (
              <span className="image-count">{images.size} images</span>
            )}
            <span className="file-name">{activeImage.fileName}</span>
            <span className="image-size">
              {activeImage.width} x {activeImage.height}
            </span>
            <span className="follicle-count">
              {activeImageAnnotationCount} annotations
            </span>
          </>
        ) : (
          <span className="no-image">No image loaded</span>
        )}
      </div>

      {/* Detection Settings Dialog (Manual Settings) */}
      {showDetectionSettings && (
        <DetectionSettingsDialog
          settings={detectionSettings}
          onSave={(settings) => {
            setDetectionSettings(settings);
            setShowDetectionSettings(false);
          }}
          onCancel={() => setShowDetectionSettings(false)}
          blobServerConnected={blobServerConnected}
          onServerRestarted={handleServerRestarted}
          installState={gpuInstallState}
          onInstallStateChange={setGpuInstallState}
        />
      )}

      {/* Learned Detection Dialog */}
      {showLearnedDetection && blobSessionId && (
        <LearnedDetectionDialog
          sessionId={blobSessionId}
          settings={learnedSettings}
          onRun={handleLearnedDetect}
          onCancel={() => setShowLearnedDetection(false)}
        />
      )}

      {/* YOLO Training Dialog */}
      {showYOLOTraining && (
        <YOLOTrainingDialog onClose={() => setShowYOLOTraining(false)} />
      )}

      {/* YOLO Model Manager Dialog */}
      {showYOLOModelManager && (
        <YOLOModelManager
          onClose={() => setShowYOLOModelManager(false)}
          onModelLoaded={(modelPath) => {
            console.log('Model loaded:', modelPath);
            setShowYOLOModelManager(false);
          }}
        />
      )}

      {/* YOLO Detection Training Dialog */}
      {showYOLODetectionTraining && (
        <YOLODetectionTrainingDialog onClose={() => setShowYOLODetectionTraining(false)} />
      )}

      {/* YOLO Detection Model Manager Dialog */}
      {showYOLODetectionModelManager && (
        <YOLODetectionModelManager
          onClose={() => setShowYOLODetectionModelManager(false)}
          onModelLoaded={() => {
            setShowYOLODetectionModelManager(false);
          }}
        />
      )}

      {/* Import Annotations Dialog */}
      {showImportDialog && importAnalysis && (
        <ImportAnnotationsDialog
          analysis={importAnalysis}
          onImport={handleImportConfirm}
          onCancel={handleImportCancel}
        />
      )}
    </div>
  );
};
