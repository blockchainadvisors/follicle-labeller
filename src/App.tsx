import React from 'react';
import { ImageCanvas } from './components/Canvas/ImageCanvas';
import { Toolbar } from './components/Toolbar/Toolbar';
import { PropertyPanel } from './components/PropertyPanel/PropertyPanel';

const App: React.FC = () => {
  return (
    <div className="app">
      <Toolbar />
      <div className="main-content">
        <ImageCanvas />
        <PropertyPanel />
      </div>
    </div>
  );
};

export default App;
