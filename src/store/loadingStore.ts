import { create } from "zustand";

interface LoadingState {
  isLoading: boolean;
  message: string;
  canCancel: boolean;
  abortController: AbortController | null;

  // Actions
  startLoading: (message: string, canCancel?: boolean) => AbortController | null;
  stopLoading: () => void;
  cancelLoading: () => void;
}

export const useLoadingStore = create<LoadingState>((set, get) => ({
  isLoading: false,
  message: "",
  canCancel: false,
  abortController: null,

  startLoading: (message, canCancel = false) => {
    const controller = canCancel ? new AbortController() : null;
    set({
      isLoading: true,
      message,
      canCancel,
      abortController: controller,
    });
    return controller;
  },

  stopLoading: () => {
    set({
      isLoading: false,
      message: "",
      canCancel: false,
      abortController: null,
    });
  },

  cancelLoading: () => {
    const { abortController } = get();
    if (abortController) {
      abortController.abort();
    }
    set({
      isLoading: false,
      message: "",
      canCancel: false,
      abortController: null,
    });
  },
}));
