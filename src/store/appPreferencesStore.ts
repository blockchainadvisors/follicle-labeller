import { create } from 'zustand';
import { persist } from 'zustand/middleware';

// App-level user preferences that persist across sessions but are not part
// of any single project. Distinct from settingsStore (which is detection /
// per-project) — those concerns happen to share the word "settings" but
// are owned separately.
interface AppPreferencesState {
  // Folder where tracking-session screen recordings are written. ``null``
  // means "use the OS Downloads folder", resolved at save time via the
  // electronAPI. We store ``null`` rather than the resolved path so that
  // moving the user's home directory or syncing prefs across machines
  // doesn't pin a stale path.
  screenRecordingFolder: string | null;

  setScreenRecordingFolder: (path: string | null) => void;
  clearAll: () => void;
}

export const useAppPreferencesStore = create<AppPreferencesState>()(
  persist(
    (set) => ({
      screenRecordingFolder: null,

      setScreenRecordingFolder: (path) => set({ screenRecordingFolder: path }),

      clearAll: () => set({ screenRecordingFolder: null }),
    }),
    {
      name: 'follicle-labeller-preferences',
      partialize: (state) => ({
        screenRecordingFolder: state.screenRecordingFolder,
      }),
    },
  ),
);
