import React from 'react';
import { ImageCanvas } from './components/Canvas/ImageCanvas';
import { Toolbar } from './components/Toolbar/Toolbar';
import { PropertyPanel } from './components/PropertyPanel/PropertyPanel';
import { ImageExplorer } from './components/ImageExplorer/ImageExplorer';
import { HelpPanel } from './components/HelpPanel/HelpPanel';

const App: React.FC = () => {
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
