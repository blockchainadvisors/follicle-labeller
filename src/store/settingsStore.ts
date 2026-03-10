import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { ImageId, DetectionSettingsExport } from '../types';
import { DEFAULT_DETECTION_SETTINGS } from '../components/DetectionSettingsDialog/DetectionSettingsDialog';
import type { DetectionSettings } from '../components/DetectionSettingsDialog/DetectionSettingsDialog';

// Information about a missing model
export interface MissingModelInfo {
  modelId: string;
  modelName: string | null;
  modelSource: 'pretrained' | 'custom';
}

// Information about a missing keypoint model
export interface MissingKeypointModelInfo {
  modelId: string;
  modelName: string | null;
  modelSource: 'pretrained' | 'custom';
}

interface SettingsState {
  // Global settings (project default)
  globalDetectionSettings: DetectionSettings;

  // Per-image overrides (sparse map - only stores images with custom settings)
  imageSettingsOverrides: Map<ImageId, Partial<DetectionSettings>>;

  // Missing model tracking - set when a project references a model that doesn't exist
  missingModelInfo: MissingModelInfo | null;
  missingKeypointModelInfo: MissingKeypointModelInfo | null;

  // Actions
  setGlobalDetectionSettings: (settings: DetectionSettings) => void;
  setImageSettingsOverride: (imageId: ImageId, settings: Partial<DetectionSettings>) => void;
  clearImageSettingsOverride: (imageId: ImageId) => void;
  getEffectiveSettings: (imageId: ImageId | null) => DetectionSettings;

  // Persistence
  loadFromProject: (
    globalSettings?: DetectionSettingsExport,
    imageOverrides?: Map<string, Partial<DetectionSettingsExport>>
  ) => void;

  // Model validation (called after models list is loaded)
  validateModelAvailability: (availableModelIds: string[], availableKeypointModelIds?: string[]) => void;
  clearMissingModelWarning: () => void;
  clearMissingKeypointModelWarning: () => void;

  // Export helpers
  getGlobalSettingsForExport: () => DetectionSettingsExport;
  getImageOverridesForExport: () => Map<string, Partial<DetectionSettingsExport>>;

  // Reset
  clearAll: () => void;
}

export const useSettingsStore = create<SettingsState>()(
  persist(
    (set, get) => ({
  globalDetectionSettings: { ...DEFAULT_DETECTION_SETTINGS },
  imageSettingsOverrides: new Map(),
  missingModelInfo: null,
  missingKeypointModelInfo: null,

  setGlobalDetectionSettings: (settings) => {
    set({ globalDetectionSettings: settings });
  },

  setImageSettingsOverride: (imageId, settings) => {
    set((state) => {
      const newOverrides = new Map(state.imageSettingsOverrides);
      newOverrides.set(imageId, settings);
      return { imageSettingsOverrides: newOverrides };
    });
  },

  clearImageSettingsOverride: (imageId) => {
    set((state) => {
      const newOverrides = new Map(state.imageSettingsOverrides);
      newOverrides.delete(imageId);
      return { imageSettingsOverrides: newOverrides };
    });
  },

  getEffectiveSettings: (imageId) => {
    const state = get();
    if (!imageId) return state.globalDetectionSettings;
    const override = state.imageSettingsOverrides.get(imageId);
    if (!override) return state.globalDetectionSettings;
    return { ...state.globalDetectionSettings, ...override };
  },

  loadFromProject: (globalSettings, imageOverrides) => {
    // Merge with defaults for backward compatibility
    // Old files without settings will use all defaults
    // New fields (yoloModelName, yoloModelSource) will get defaults if missing
    const merged: DetectionSettings = globalSettings
      ? { ...DEFAULT_DETECTION_SETTINGS, ...globalSettings }
      : { ...DEFAULT_DETECTION_SETTINGS };

    // Convert image overrides (also merge with defaults to handle partial settings)
    const overridesMap = new Map<ImageId, Partial<DetectionSettings>>();
    if (imageOverrides) {
      for (const [imageId, override] of imageOverrides.entries()) {
        overridesMap.set(imageId, override);
      }
    }

    // Check if we have a custom model ID - we'll validate availability later
    // when the Toolbar component loads the models list
    let missingInfo: MissingModelInfo | null = null;
    if (merged.yoloModelId && merged.yoloModelSource === 'custom') {
      // Store potential missing model info - will be cleared if model is found
      missingInfo = {
        modelId: merged.yoloModelId,
        modelName: merged.yoloModelName,
        modelSource: merged.yoloModelSource,
      };
    }

    // Check for keypoint model
    let missingKeypointInfo: MissingKeypointModelInfo | null = null;
    if (merged.keypointModelId && merged.keypointModelSource === 'custom') {
      missingKeypointInfo = {
        modelId: merged.keypointModelId,
        modelName: merged.keypointModelName,
        modelSource: merged.keypointModelSource,
      };
    }

    set({
      globalDetectionSettings: merged,
      imageSettingsOverrides: overridesMap,
      missingModelInfo: missingInfo,
      missingKeypointModelInfo: missingKeypointInfo,
    });
  },

  validateModelAvailability: (availableModelIds, availableKeypointModelIds) => {
    const state = get();
    const { globalDetectionSettings } = state;

    // If there's a custom detection model referenced, check if it exists
    if (globalDetectionSettings.yoloModelId && globalDetectionSettings.yoloModelSource === 'custom') {
      const modelExists = availableModelIds.includes(globalDetectionSettings.yoloModelId);

      if (modelExists) {
        // Model found, clear any missing warning
        set({ missingModelInfo: null });
      } else {
        // Model not found, set missing info
        set({
          missingModelInfo: {
            modelId: globalDetectionSettings.yoloModelId,
            modelName: globalDetectionSettings.yoloModelName,
            modelSource: globalDetectionSettings.yoloModelSource,
          },
        });
      }
    } else {
      // No custom model, clear missing info
      set({ missingModelInfo: null });
    }

    // If there's a custom keypoint model referenced, check if it exists
    if (availableKeypointModelIds && globalDetectionSettings.keypointModelId && globalDetectionSettings.keypointModelSource === 'custom') {
      const keypointModelExists = availableKeypointModelIds.includes(globalDetectionSettings.keypointModelId);

      if (keypointModelExists) {
        set({ missingKeypointModelInfo: null });
      } else {
        set({
          missingKeypointModelInfo: {
            modelId: globalDetectionSettings.keypointModelId,
            modelName: globalDetectionSettings.keypointModelName,
            modelSource: globalDetectionSettings.keypointModelSource,
          },
        });
      }
    } else if (availableKeypointModelIds) {
      set({ missingKeypointModelInfo: null });
    }
  },

  clearMissingModelWarning: () => {
    set({ missingModelInfo: null });
  },

  clearMissingKeypointModelWarning: () => {
    set({ missingKeypointModelInfo: null });
  },

  getGlobalSettingsForExport: () => {
    const state = get();
    // Convert DetectionSettings to DetectionSettingsExport (they have the same shape)
    return state.globalDetectionSettings as DetectionSettingsExport;
  },

  getImageOverridesForExport: () => {
    const state = get();
    // Return a copy of the map with values cast to export format
    const exportMap = new Map<string, Partial<DetectionSettingsExport>>();
    for (const [imageId, override] of state.imageSettingsOverrides.entries()) {
      exportMap.set(imageId, override as Partial<DetectionSettingsExport>);
    }
    return exportMap;
  },

  clearAll: () => {
    set({
      globalDetectionSettings: { ...DEFAULT_DETECTION_SETTINGS },
      imageSettingsOverrides: new Map(),
      missingModelInfo: null,
      missingKeypointModelInfo: null,
    });
  },
    }),
    {
      name: 'follicle-labeller-settings',
      // Only persist global settings - image overrides are stored in project files
      partialize: (state) => ({
        globalDetectionSettings: state.globalDetectionSettings,
      }),
    }
  )
);
