import React, { useEffect } from 'react';
import { ImageCanvas } from './components/Canvas/ImageCanvas';
import { Toolbar } from './components/Toolbar/Toolbar';
import { PropertyPanel } from './components/PropertyPanel/PropertyPanel';
import { ImageExplorer } from './components/ImageExplorer/ImageExplorer';
import { HelpPanel } from './components/HelpPanel/HelpPanel';
import { StatisticsPanel } from './components/StatisticsPanel/StatisticsPanel';
import { LoadingOverlay } from './components/LoadingOverlay';
import { useThemeStore } from './store/themeStore';
import { useCanvasStore } from './store/canvasStore';

const App: React.FC = () => {
  const initializeTheme = useThemeStore(state => state.initializeTheme);
  const showStatistics = useCanvasStore(state => state.showStatistics);
  const toggleStatistics = useCanvasStore(state => state.toggleStatistics);

  // Initialize theme on app startup
  useEffect(() => {
    initializeTheme();
  }, [initializeTheme]);

  return (
    <div className="app">
      <Toolbar />
      <div className="main-content">
        <ImageExplorer />
        <ImageCanvas />
        <PropertyPanel />
      </div>
      <HelpPanel />
      {showStatistics && <StatisticsPanel onClose={toggleStatistics} />}
      <LoadingOverlay />
    </div>
  );
};

export default App;
