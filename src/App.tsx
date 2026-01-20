import React, { useEffect } from 'react';
import { ImageCanvas } from './components/Canvas/ImageCanvas';
import { Toolbar } from './components/Toolbar/Toolbar';
import { PropertyPanel } from './components/PropertyPanel/PropertyPanel';
import { ImageExplorer } from './components/ImageExplorer/ImageExplorer';
import { HelpPanel } from './components/HelpPanel/HelpPanel';
import { useThemeStore } from './store/themeStore';

const App: React.FC = () => {
  const initializeTheme = useThemeStore(state => state.initializeTheme);

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
    </div>
  );
};

export default App;
