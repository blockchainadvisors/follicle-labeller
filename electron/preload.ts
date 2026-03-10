import { contextBridge, ipcRenderer, IpcRendererEvent } from "electron";

// Menu action callback type
type MenuCallback = () => void;

// Define the API exposed to the renderer
const electronAPI = {
  // Open image file dialog - returns file path and data as ArrayBuffer
  openImageDialog: (): Promise<{
    filePath: string;
    fileName: string;
    data: ArrayBuffer;
  } | null> => ipcRenderer.invoke("dialog:openImage"),

  // Open generic file dialog with filters - returns file path and data
  openFileDialog: (options: {
    filters?: Array<{ name: string; extensions: string[] }>;
    title?: string;
  }): Promise<{
    filePath: string;
    fileName: string;
    data: ArrayBuffer;
  } | null> => ipcRenderer.invoke("dialog:openFile", options),

  // Legacy V1: Save project as .fol archive (image + annotations)
  saveProject: (
    imageData: ArrayBuffer,
    imageFileName: string,
    jsonData: string,
  ): Promise<boolean> =>
    ipcRenderer.invoke(
      "dialog:saveProject",
      imageData,
      imageFileName,
      jsonData,
    ),

  // Legacy V1: Load project from .fol archive
  loadProject: (): Promise<{
    imageFileName: string;
    imageData: ArrayBuffer;
    jsonData: string;
  } | null> => ipcRenderer.invoke("dialog:loadProject"),

  // V2: Save project with multiple images (Save As - shows dialog)
  saveProjectV2: (
    images: Array<{ id: string; fileName: string; data: ArrayBuffer }>,
    manifest: string,
    annotations: string,
    defaultPath?: string,
  ): Promise<{ success: boolean; filePath?: string }> =>
    ipcRenderer.invoke(
      "dialog:saveProjectV2",
      images,
      manifest,
      annotations,
      defaultPath,
    ),

  // V2: Save project to specific path (silent save - no dialog)
  saveProjectV2ToPath: (
    filePath: string,
    images: Array<{ id: string; fileName: string; data: ArrayBuffer }>,
    manifest: string,
    annotations: string,
  ): Promise<{ success: boolean; filePath?: string }> =>
    ipcRenderer.invoke(
      "file:saveProjectV2",
      filePath,
      images,
      manifest,
      annotations,
    ),

  // Update menu state based on project
  setProjectState: (hasProject: boolean): void => {
    ipcRenderer.send("menu:setProjectState", hasProject);
  },

  // V2: Load project with support for V1 and V2 formats
  loadProjectV2: (): Promise<{
    version: "1.0" | "2.0";
    filePath: string;
    imageFileName?: string;
    imageData?: ArrayBuffer;
    jsonData?: string;
    manifest?: string;
    images?: Array<{ id: string; fileName: string; data: ArrayBuffer }>;
    annotations?: string;
  } | null> => ipcRenderer.invoke("dialog:loadProjectV2"),

  // Get file to open on startup (from file association)
  getFileToOpen: (): Promise<string | null> =>
    ipcRenderer.invoke("app:getFileToOpen"),

  // Load project from specific file path (for file association)
  loadProjectFromPath: (
    filePath: string,
  ): Promise<{
    version: "1.0" | "2.0";
    filePath: string;
    imageFileName?: string;
    imageData?: ArrayBuffer;
    jsonData?: string;
    manifest?: string;
    images?: Array<{ id: string; fileName: string; data: ArrayBuffer }>;
    annotations?: string;
  } | null> => ipcRenderer.invoke("file:loadProject", filePath),

  // Listen for file open events (when app is already running)
  onFileOpen: (callback: (filePath: string) => void) => {
    const handler = (_event: IpcRendererEvent, filePath: string) =>
      callback(filePath);
    ipcRenderer.on("file:open", handler);
    return () => ipcRenderer.removeListener("file:open", handler);
  },

  // Unsaved changes dialog - returns 'save' | 'discard' | 'cancel'
  showUnsavedChangesDialog: (): Promise<"save" | "discard" | "cancel"> =>
    ipcRenderer.invoke("dialog:unsavedChanges"),

  // Download options dialog - returns 'all' | 'currentImage' | 'selected' | 'cancel'
  showDownloadOptionsDialog: (
    selectedCount: number,
    currentImageCount: number,
    totalCount: number,
  ): Promise<"all" | "currentImage" | "selected" | "cancel"> =>
    ipcRenderer.invoke(
      "dialog:downloadOptions",
      selectedCount,
      currentImageCount,
      totalCount,
    ),

  // Listen for unsaved changes check request from main
  onCheckUnsavedChanges: (callback: () => void) => {
    const handler = (_event: IpcRendererEvent) => callback();
    ipcRenderer.on("app:checkUnsavedChanges", handler);
    return () => ipcRenderer.removeListener("app:checkUnsavedChanges", handler);
  },

  // Confirm close to main process
  confirmClose: (canClose: boolean): void => {
    ipcRenderer.send("app:confirmClose", canClose);
  },

  // Update download progress listener (for optional custom UI)
  onUpdateDownloadProgress: (
    callback: (progress: {
      percent: number;
      transferred: number;
      total: number;
      bytesPerSecond: number;
    }) => void,
  ) => {
    const handler = (
      _event: IpcRendererEvent,
      progress: {
        percent: number;
        transferred: number;
        total: number;
        bytesPerSecond: number;
      },
    ) => callback(progress);
    ipcRenderer.on("update:downloadProgress", handler);
    return () => ipcRenderer.removeListener("update:downloadProgress", handler);
  },

  // Menu event listeners (return cleanup function)
  onMenuOpenImage: (callback: MenuCallback) => {
    const handler = (_event: IpcRendererEvent) => callback();
    ipcRenderer.on("menu:openImage", handler);
    return () => ipcRenderer.removeListener("menu:openImage", handler);
  },
  onMenuLoadProject: (callback: MenuCallback) => {
    const handler = (_event: IpcRendererEvent) => callback();
    ipcRenderer.on("menu:loadProject", handler);
    return () => ipcRenderer.removeListener("menu:loadProject", handler);
  },
  onMenuSaveProject: (callback: MenuCallback) => {
    const handler = (_event: IpcRendererEvent) => callback();
    ipcRenderer.on("menu:saveProject", handler);
    return () => ipcRenderer.removeListener("menu:saveProject", handler);
  },
  onMenuSaveProjectAs: (callback: MenuCallback) => {
    const handler = (_event: IpcRendererEvent) => callback();
    ipcRenderer.on("menu:saveProjectAs", handler);
    return () => ipcRenderer.removeListener("menu:saveProjectAs", handler);
  },
  onMenuCloseProject: (callback: MenuCallback) => {
    const handler = (_event: IpcRendererEvent) => callback();
    ipcRenderer.on("menu:closeProject", handler);
    return () => ipcRenderer.removeListener("menu:closeProject", handler);
  },
  onMenuUndo: (callback: MenuCallback) => {
    const handler = (_event: IpcRendererEvent) => callback();
    ipcRenderer.on("menu:undo", handler);
    return () => ipcRenderer.removeListener("menu:undo", handler);
  },
  onMenuRedo: (callback: MenuCallback) => {
    const handler = (_event: IpcRendererEvent) => callback();
    ipcRenderer.on("menu:redo", handler);
    return () => ipcRenderer.removeListener("menu:redo", handler);
  },
  onMenuClearAll: (callback: MenuCallback) => {
    const handler = (_event: IpcRendererEvent) => callback();
    ipcRenderer.on("menu:clearAll", handler);
    return () => ipcRenderer.removeListener("menu:clearAll", handler);
  },
  onMenuToggleShapes: (callback: MenuCallback) => {
    const handler = (_event: IpcRendererEvent) => callback();
    ipcRenderer.on("menu:toggleShapes", handler);
    return () => ipcRenderer.removeListener("menu:toggleShapes", handler);
  },
  onMenuToggleLabels: (callback: MenuCallback) => {
    const handler = (_event: IpcRendererEvent) => callback();
    ipcRenderer.on("menu:toggleLabels", handler);
    return () => ipcRenderer.removeListener("menu:toggleLabels", handler);
  },
  onMenuZoomIn: (callback: MenuCallback) => {
    const handler = (_event: IpcRendererEvent) => callback();
    ipcRenderer.on("menu:zoomIn", handler);
    return () => ipcRenderer.removeListener("menu:zoomIn", handler);
  },
  onMenuZoomOut: (callback: MenuCallback) => {
    const handler = (_event: IpcRendererEvent) => callback();
    ipcRenderer.on("menu:zoomOut", handler);
    return () => ipcRenderer.removeListener("menu:zoomOut", handler);
  },
  onMenuResetZoom: (callback: MenuCallback) => {
    const handler = (_event: IpcRendererEvent) => callback();
    ipcRenderer.on("menu:resetZoom", handler);
    return () => ipcRenderer.removeListener("menu:resetZoom", handler);
  },
  onMenuShowHelp: (callback: MenuCallback) => {
    const handler = (_event: IpcRendererEvent) => callback();
    ipcRenderer.on("menu:showHelp", handler);
    return () => ipcRenderer.removeListener("menu:showHelp", handler);
  },

  // System power events - triggered before sleep/hibernate
  onSystemSuspend: (callback: MenuCallback) => {
    const handler = (_event: IpcRendererEvent) => callback();
    ipcRenderer.on("system:suspend", handler);
    return () => ipcRenderer.removeListener("system:suspend", handler);
  },

  // BLOB Detection Server API
  blob: {
    // Start the BLOB detection server
    startServer: (): Promise<{ success: boolean; error?: string; errorDetails?: string }> =>
      ipcRenderer.invoke("blob:startServer"),

    // Stop the BLOB detection server
    stopServer: (): Promise<{ success: boolean }> =>
      ipcRenderer.invoke("blob:stopServer"),

    // Check if BLOB server is available
    isAvailable: (): Promise<boolean> => ipcRenderer.invoke("blob:isAvailable"),

    // Check if Python is installed
    checkPython: (): Promise<{
      available: boolean;
      version?: string;
      error?: string;
    }> => ipcRenderer.invoke("blob:checkPython"),

    // Get BLOB server info
    getServerInfo: (): Promise<{
      port: number;
      running: boolean;
      scriptPath: string;
    }> => ipcRenderer.invoke("blob:getServerInfo"),

    // Get setup progress status
    getSetupStatus: (): Promise<string> =>
      ipcRenderer.invoke("blob:getSetupStatus"),

    // Listen for setup progress events (includes download percentage when downloading Python)
    onSetupProgress: (callback: (status: string, percent?: number) => void) => {
      const handler = (_event: IpcRendererEvent, status: string, percent?: number) =>
        callback(status, percent);
      ipcRenderer.on("blob:setupProgress", handler);
      return () => ipcRenderer.removeListener("blob:setupProgress", handler);
    },

    // Get GPU backend information
    getGPUInfo: (): Promise<{
      activeBackend: "cuda" | "mps" | "cpu";
      deviceName: string;
      memoryGB?: number;
      available: { cuda: boolean; mps: boolean };
    }> => ipcRenderer.invoke("blob:getGPUInfo"),

    // Restart the BLOB server (after GPU package installation)
    restartServer: (): Promise<{ success: boolean; error?: string; errorDetails?: string }> =>
      ipcRenderer.invoke("blob:restartServer"),
  },

  // GPU Hardware Detection & Package Installation API
  gpu: {
    // Get GPU hardware info (works before packages installed)
    getHardwareInfo: (): Promise<{
      hardware: {
        nvidia: { found: boolean; name?: string; driver_version?: string };
        apple_silicon: { found: boolean; chip?: string };
      };
      packages: { cupy: boolean; torch: boolean };
      canEnableGpu: boolean;
      gpuEnabled: boolean;
    }> => ipcRenderer.invoke("gpu:getHardwareInfo"),

    // Install GPU packages (cupy-cuda12x on Windows/Linux, torch on macOS)
    installPackages: (): Promise<{ success: boolean; error?: string }> =>
      ipcRenderer.invoke("gpu:installPackages"),

    // Listen for install progress events
    onInstallProgress: (
      callback: (data: { message: string; percent?: number }) => void
    ) => {
      const handler = (_event: IpcRendererEvent, data: { message: string; percent?: number }) =>
        callback(data);
      ipcRenderer.on("gpu:installProgress", handler);
      return () => ipcRenderer.removeListener("gpu:installProgress", handler);
    },
  },

  // File system utilities
  fileExists: (filePath: string): Promise<boolean> =>
    ipcRenderer.invoke("file:exists", filePath),

  // TensorRT Installation API
  tensorrt: {
    // Check TensorRT availability and CUDA support
    check: (): Promise<{
      available: boolean;
      version: string | null;
      canInstall: boolean;
    }> => ipcRenderer.invoke("tensorrt:check"),

    // Install TensorRT packages
    install: (): Promise<{ success: boolean; error?: string }> =>
      ipcRenderer.invoke("tensorrt:install"),

    // Listen for install progress events
    onInstallProgress: (
      callback: (data: { message: string; percent?: number }) => void
    ) => {
      const handler = (_event: IpcRendererEvent, data: { message: string; percent?: number }) =>
        callback(data);
      ipcRenderer.on("tensorrt:installProgress", handler);
      return () => ipcRenderer.removeListener("tensorrt:installProgress", handler);
    },
  },

  // YOLO Keypoint Training API
  yoloKeypoint: {
    // Check if YOLO dependencies are installed
    checkDependencies: (): Promise<{
      installed: boolean;
      missing: string[];
      estimatedSize: string;
    }> => ipcRenderer.invoke("yolo:checkDependencies"),

    // Install YOLO dependencies
    installDependencies: (): Promise<{ success: boolean; error?: string }> =>
      ipcRenderer.invoke("yolo:installDependencies"),

    // Upgrade PyTorch to CUDA version for GPU training
    upgradeToCUDA: (): Promise<{ success: boolean; error?: string }> =>
      ipcRenderer.invoke("yolo:upgradeToCUDA"),

    // Listen for install progress events
    onInstallProgress: (
      callback: (data: { message: string; percent?: number }) => void
    ) => {
      const handler = (_event: IpcRendererEvent, data: { message: string; percent?: number }) =>
        callback(data);
      ipcRenderer.on("yolo:installProgress", handler);
      return () => ipcRenderer.removeListener("yolo:installProgress", handler);
    },

    // Get service status
    getStatus: (): Promise<{
      available: boolean;
      sseAvailable: boolean;
      activeTrainingJobs: number;
    }> => ipcRenderer.invoke("yolo-keypoint:getStatus"),

    // Get system info (CPU/GPU, memory)
    getSystemInfo: (): Promise<{
      python_version: string;
      platform: string;
      cpu_count: number;
      cpu_percent: number;
      memory_total_gb: number;
      memory_used_gb: number;
      memory_percent: number;
      device: string;
      device_name: string;
      cuda_available: boolean;
      mps_available: boolean;
      torch_version: string | null;
      gpu_memory_total_gb: number | null;
      gpu_memory_used_gb: number | null;
    }> => ipcRenderer.invoke("yolo-keypoint:getSystemInfo"),

    // Validate dataset
    validateDataset: (
      datasetPath: string
    ): Promise<{
      valid: boolean;
      trainImages: number;
      valImages: number;
      trainLabels: number;
      valLabels: number;
      errors: string[];
      warnings: string[];
    }> => ipcRenderer.invoke("yolo-keypoint:validateDataset", datasetPath),

    // Start training
    startTraining: (
      datasetPath: string,
      config: {
        modelSize?: string;
        epochs?: number;
        imgSize?: number;
        batchSize?: number;
        patience?: number;
        device?: string;
      },
      modelName?: string
    ): Promise<{ jobId: string; status: string }> =>
      ipcRenderer.invoke("yolo-keypoint:startTraining", datasetPath, config, modelName),

    // Stop training
    stopTraining: (jobId: string): Promise<{ success: boolean }> =>
      ipcRenderer.invoke("yolo-keypoint:stopTraining", jobId),

    // Subscribe to training progress via IPC (main process proxies SSE)
    subscribeProgress: (
      jobId: string,
      onProgress: (progress: {
        status: string;
        epoch: number;
        totalEpochs: number;
        loss: number;
        boxLoss: number;
        poseLoss: number;
        kobjLoss: number;
        metrics: Record<string, number>;
        eta: string;
        message: string;
      }) => void,
      onError: (error: string) => void,
      onComplete: () => void
    ): (() => void) => {
      // Set up IPC listeners for progress events from main process
      const progressHandler = (
        _event: Electron.IpcRendererEvent,
        receivedJobId: string,
        progress: any
      ) => {
        if (receivedJobId !== jobId) return;
        // Map snake_case to camelCase
        onProgress({
          status: progress.status || "",
          epoch: progress.epoch || 0,
          totalEpochs: progress.total_epochs || 0,
          loss: progress.loss || 0,
          boxLoss: progress.box_loss || 0,
          poseLoss: progress.pose_loss || 0,
          kobjLoss: progress.kobj_loss || 0,
          metrics: progress.metrics || {},
          eta: progress.eta || "",
          message: progress.message || "",
        });
      };

      const errorHandler = (
        _event: Electron.IpcRendererEvent,
        receivedJobId: string,
        error: string
      ) => {
        if (receivedJobId !== jobId) return;
        onError(error);
      };

      const completeHandler = (
        _event: Electron.IpcRendererEvent,
        receivedJobId: string
      ) => {
        if (receivedJobId !== jobId) return;
        cleanup();
        onComplete();
      };

      // Register listeners
      ipcRenderer.on("yolo-keypoint:progress", progressHandler);
      ipcRenderer.on("yolo-keypoint:progress-error", errorHandler);
      ipcRenderer.on("yolo-keypoint:progress-complete", completeHandler);

      // Start the SSE subscription in main process
      ipcRenderer.invoke("yolo-keypoint:subscribeProgress", jobId);

      // Cleanup function
      const cleanup = () => {
        ipcRenderer.removeListener("yolo-keypoint:progress", progressHandler);
        ipcRenderer.removeListener(
          "yolo-keypoint:progress-error",
          errorHandler
        );
        ipcRenderer.removeListener(
          "yolo-keypoint:progress-complete",
          completeHandler
        );
        ipcRenderer.invoke("yolo-keypoint:unsubscribeProgress", jobId);
      };

      return cleanup;
    },

    // List trained models
    listModels: (): Promise<{
      models: Array<{
        id: string;
        name: string;
        path: string;
        createdAt: string;
        epochsTrained: number;
        imgSize: number;
        metrics: Record<string, number>;
      }>;
    }> => ipcRenderer.invoke("yolo-keypoint:listModels"),

    // Load model for inference
    loadModel: (modelPath: string): Promise<{ success: boolean }> =>
      ipcRenderer.invoke("yolo-keypoint:loadModel", modelPath),

    // Run prediction
    predict: (
      imageData: string
    ): Promise<{
      success: boolean;
      prediction?: {
        origin: { x: number; y: number };
        directionEndpoint: { x: number; y: number };
        confidence: number;
      };
      message?: string;
    }> => ipcRenderer.invoke("yolo-keypoint:predict", imageData),

    // Show save dialog for ONNX export
    showExportDialog: (
      defaultFileName: string
    ): Promise<{ canceled: boolean; filePath?: string }> =>
      ipcRenderer.invoke("yolo-keypoint:showExportDialog", defaultFileName),

    // Export to ONNX
    exportONNX: (
      modelPath: string,
      outputPath: string
    ): Promise<{ success: boolean; outputPath?: string }> =>
      ipcRenderer.invoke("yolo-keypoint:exportONNX", modelPath, outputPath),

    // Delete model
    deleteModel: (modelId: string): Promise<{ success: boolean }> =>
      ipcRenderer.invoke("yolo-keypoint:deleteModel", modelId),

    // Write dataset files to temp directory (for training from current project)
    writeDatasetToTemp: (
      files: Array<{ path: string; content: ArrayBuffer | string }>
    ): Promise<{ success: boolean; datasetPath?: string; error?: string }> =>
      ipcRenderer.invoke("yolo-keypoint:writeDatasetToTemp", files),

    // Check TensorRT availability for keypoint inference
    checkTensorRTAvailable: (): Promise<{ available: boolean; version: string | null }> =>
      ipcRenderer.invoke("yolo-keypoint:checkTensorRT"),

    // Export to TensorRT engine format
    exportToTensorRT: (
      modelPath: string,
      outputPath?: string,
      half?: boolean,
      imgsz?: number
    ): Promise<{ success: boolean; engine_path?: string; error?: string }> =>
      ipcRenderer.invoke("yolo-keypoint:exportTensorRT", modelPath, outputPath, half, imgsz),
  },

  // Model Export/Import API
  model: {
    // Export model as portable package (ZIP with model.pt + config.json)
    // suggestedFileName is optional - if not provided, filename is generated from config
    exportPackage: (
      modelId: string,
      modelPath: string,
      config: Record<string, unknown>,
      suggestedFileName?: string
    ): Promise<{ success: boolean; filePath?: string; canceled?: boolean; error?: string }> =>
      ipcRenderer.invoke("model:exportPackage", modelId, modelPath, config, suggestedFileName),

    // Preview model package before import (read config.json from ZIP)
    // Optionally pass expectedModelType to validate the package type
    previewPackage: (
      expectedModelType?: 'detection' | 'keypoint'
    ): Promise<{
      valid: boolean;
      filePath?: string;
      config?: Record<string, unknown>;
      modelType?: 'detection' | 'keypoint';
      hasEngine?: boolean;
      canceled?: boolean;
      error?: string;
    }> => ipcRenderer.invoke("model:previewPackage", expectedModelType),

    // Import model package
    importPackage: (
      filePath: string,
      newModelName?: string
    ): Promise<{
      success: boolean;
      modelId?: string;
      modelPath?: string;
      modelName?: string;
      modelType?: 'detection' | 'keypoint';
      error?: string;
    }> => ipcRenderer.invoke("model:importPackage", filePath, newModelName),
  },

  // YOLO Detection Training API (for follicle bounding box detection)
  yoloDetection: {
    // Get service status
    getStatus: (): Promise<{
      available: boolean;
      sseAvailable: boolean;
      activeTrainingJobs: number;
      loadedModel: string | null;
    }> => ipcRenderer.invoke("yolo-detection:getStatus"),

    // Validate dataset
    validateDataset: (
      datasetPath: string
    ): Promise<{
      valid: boolean;
      train_images: number;
      val_images: number;
      train_labels: number;
      val_labels: number;
      errors: string[];
      warnings: string[];
    }> => ipcRenderer.invoke("yolo-detection:validateDataset", datasetPath),

    // Start training
    startTraining: (
      datasetPath: string,
      config: {
        modelSize?: string;
        epochs?: number;
        imgSize?: number;
        batchSize?: number;
        patience?: number;
        device?: string;
      },
      modelName?: string
    ): Promise<{ jobId: string; status: string }> =>
      ipcRenderer.invoke("yolo-detection:startTraining", datasetPath, config, modelName),

    // Stop training
    stopTraining: (jobId: string): Promise<{ success: boolean }> =>
      ipcRenderer.invoke("yolo-detection:stopTraining", jobId),

    // Subscribe to training progress via IPC (main process proxies SSE)
    subscribeProgress: (
      jobId: string,
      onProgress: (progress: {
        status: string;
        epoch: number;
        totalEpochs: number;
        loss: number;
        boxLoss: number;
        clsLoss: number;
        dflLoss: number;
        metrics: Record<string, number>;
        eta: string;
        message: string;
      }) => void,
      onError: (error: string) => void,
      onComplete: () => void
    ): (() => void) => {
      const progressHandler = (
        _event: Electron.IpcRendererEvent,
        receivedJobId: string,
        progress: any
      ) => {
        if (receivedJobId !== jobId) return;
        // Map snake_case to camelCase
        onProgress({
          status: progress.status || "",
          epoch: progress.epoch || 0,
          totalEpochs: progress.total_epochs || 0,
          loss: progress.loss || 0,
          boxLoss: progress.box_loss || 0,
          clsLoss: progress.cls_loss || 0,
          dflLoss: progress.dfl_loss || 0,
          metrics: progress.metrics || {},
          eta: progress.eta || "",
          message: progress.message || "",
        });
      };

      const errorHandler = (
        _event: Electron.IpcRendererEvent,
        receivedJobId: string,
        error: string
      ) => {
        if (receivedJobId !== jobId) return;
        onError(error);
      };

      const completeHandler = (
        _event: Electron.IpcRendererEvent,
        receivedJobId: string
      ) => {
        if (receivedJobId !== jobId) return;
        cleanup();
        onComplete();
      };

      // Register listeners
      ipcRenderer.on("yolo-detection:progress", progressHandler);
      ipcRenderer.on("yolo-detection:progress-error", errorHandler);
      ipcRenderer.on("yolo-detection:progress-complete", completeHandler);

      // Start the SSE subscription in main process
      ipcRenderer.invoke("yolo-detection:subscribeProgress", jobId);

      // Cleanup function
      const cleanup = () => {
        ipcRenderer.removeListener("yolo-detection:progress", progressHandler);
        ipcRenderer.removeListener(
          "yolo-detection:progress-error",
          errorHandler
        );
        ipcRenderer.removeListener(
          "yolo-detection:progress-complete",
          completeHandler
        );
        ipcRenderer.invoke("yolo-detection:unsubscribeProgress", jobId);
      };

      return cleanup;
    },

    // List trained models
    listModels: (): Promise<{
      models: Array<{
        id: string;
        name: string;
        path: string;
        createdAt: string;
        epochsTrained: number;
        imgSize: number;
        metrics: Record<string, number>;
      }>;
    }> => ipcRenderer.invoke("yolo-detection:listModels"),

    // Get resumable models (incomplete training with last.pt)
    getResumableModels: (): Promise<{
      models: Array<{
        id: string;
        name: string;
        path: string;
        createdAt: string;
        epochsTrained: number;
        imgSize: number;
        metrics: Record<string, number>;
        epochsCompleted: number;
        totalEpochs: number;
        canResume: boolean;
      }>;
    }> => ipcRenderer.invoke("yolo-detection:getResumableModels"),

    // Load model for inference
    loadModel: (modelPath: string): Promise<{ success: boolean }> =>
      ipcRenderer.invoke("yolo-detection:loadModel", modelPath),

    // Run detection prediction on full image
    predict: (
      imageData: string,
      confidenceThreshold?: number
    ): Promise<{
      success: boolean;
      detections: Array<{
        x: number;
        y: number;
        width: number;
        height: number;
        confidence: number;
        classId: number;
        className: string;
      }>;
      count: number;
    }> => ipcRenderer.invoke("yolo-detection:predict", imageData, confidenceThreshold),

    // Run tiled detection prediction (for large images with small objects)
    predictTiled: (
      imageData: string,
      confidenceThreshold?: number,
      tileSize?: number,
      overlap?: number,
      nmsThreshold?: number,
      scaleFactor?: number
    ): Promise<{
      success: boolean;
      detections: Array<{
        x: number;
        y: number;
        width: number;
        height: number;
        confidence: number;
        classId: number;
        className: string;
      }>;
      count: number;
      method: string;
      tileSize: number;
      overlap: number;
      scaleFactor: number;
    }> => ipcRenderer.invoke(
      "yolo-detection:predictTiled",
      imageData,
      confidenceThreshold,
      tileSize,
      overlap,
      nmsThreshold,
      scaleFactor
    ),

    // Show save dialog for ONNX export
    showExportDialog: (
      defaultFileName: string
    ): Promise<{ canceled: boolean; filePath?: string }> =>
      ipcRenderer.invoke("yolo-detection:showExportDialog", defaultFileName),

    // Export to ONNX
    exportONNX: (
      modelPath: string,
      outputPath: string
    ): Promise<{ success: boolean; outputPath?: string }> =>
      ipcRenderer.invoke("yolo-detection:exportONNX", modelPath, outputPath),

    // Delete model
    deleteModel: (modelId: string): Promise<{ success: boolean }> =>
      ipcRenderer.invoke("yolo-detection:deleteModel", modelId),

    // Write dataset files to temp directory (for training from current project)
    writeDatasetToTemp: (
      files: Array<{ path: string; content: ArrayBuffer | string }>
    ): Promise<{ success: boolean; datasetPath?: string; error?: string }> =>
      ipcRenderer.invoke("yolo-detection:writeDatasetToTemp", files),

    // Check if TensorRT is available on this system
    checkTensorRTAvailable: (): Promise<{ available: boolean; version: string | null }> =>
      ipcRenderer.invoke("yolo-detection:checkTensorRT"),

    // Export model to TensorRT engine format
    exportToTensorRT: (
      modelPath: string,
      outputPath?: string,
      half?: boolean,
      imgsz?: number
    ): Promise<{ success: boolean; engine_path?: string; error?: string }> =>
      ipcRenderer.invoke("yolo-detection:exportTensorRT", modelPath, outputPath, half, imgsz),
  },
};

// Expose the API to the renderer process
contextBridge.exposeInMainWorld("electronAPI", electronAPI);

// Export type for TypeScript
export type ElectronAPI = typeof electronAPI;
