import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

/**
 * Vite configuration for Web build (no Electron)
 *
 * This builds the frontend as a static web app that communicates
 * directly with the backend server via HTTP (no IPC).
 */
export default defineConfig({
  plugins: [
    react(),
  ],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  define: {
    // Ensure window.electronAPI is undefined in web builds
    'window.electronAPI': 'undefined',
  },
  build: {
    outDir: 'dist-web',
    sourcemap: true,
  },
  server: {
    port: 5173,
    proxy: {
      // Proxy API requests to backend during development
      '/health': {
        target: 'http://127.0.0.1:5555',
        changeOrigin: true,
      },
      '/gpu-info': {
        target: 'http://127.0.0.1:5555',
        changeOrigin: true,
      },
      '/set-image': {
        target: 'http://127.0.0.1:5555',
        changeOrigin: true,
      },
      '/blob-detect': {
        target: 'http://127.0.0.1:5555',
        changeOrigin: true,
      },
      '/yolo-keypoint': {
        target: 'http://127.0.0.1:5555',
        changeOrigin: true,
      },
      '/yolo-detect': {
        target: 'http://127.0.0.1:5555',
        changeOrigin: true,
      },
      '/projects': {
        target: 'http://127.0.0.1:5555',
        changeOrigin: true,
      },
      '/model': {
        target: 'http://127.0.0.1:5555',
        changeOrigin: true,
      },
      '/files': {
        target: 'http://127.0.0.1:5555',
        changeOrigin: true,
      },
    },
  },
  preview: {
    port: 4173,
  },
});
