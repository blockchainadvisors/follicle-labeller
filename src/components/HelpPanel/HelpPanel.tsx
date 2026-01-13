import React from 'react';
import { useCanvasStore } from '../../store/canvasStore';

export const HelpPanel: React.FC = () => {
  const showHelp = useCanvasStore(state => state.showHelp);
  const toggleHelp = useCanvasStore(state => state.toggleHelp);

  if (!showHelp) return null;

  return (
    <div className="help-panel">
      <div className="help-header">
        <h3>User Guide</h3>
        <button onClick={toggleHelp} title="Close (?)">
          Close
        </button>
      </div>
      <div className="help-content">
        <h1>Follicle Labeller User Guide</h1>
        <p>Welcome to Follicle Labeller, a medical image annotation tool for labeling follicles and other structures in medical images.</p>

        <hr />

        <h2>Getting Started</h2>
        <h3>Opening an Image</h3>
        <ol>
          <li>Click the <strong>Open Image</strong> button in the toolbar</li>
          <li>Select an image file (PNG, JPG, TIFF, BMP, or WebP)</li>
          <li>The image will appear in the canvas area and automatically fit to view</li>
        </ol>
        <p><strong>Tip:</strong> You can add multiple images to a project. Each image maintains its own annotations and zoom/pan state.</p>

        <hr />

        <h2>Creating Annotations</h2>
        <p>First, make sure you're in <strong>Create</strong> mode by clicking the Create button or pressing <code>C</code>.</p>

        <h3>Circle Annotations</h3>
        <p>Best for round structures like follicles.</p>
        <ol>
          <li>Select <strong>Circle</strong> shape (or press <code>1</code>)</li>
          <li>Click on the center of the follicle</li>
          <li>Move the mouse outward to preview the radius</li>
          <li>Click again to create the annotation</li>
        </ol>

        <h3>Rectangle Annotations</h3>
        <p>Best for rectangular regions or bounding boxes.</p>
        <ol>
          <li>Select <strong>Rectangle</strong> shape (or press <code>2</code>)</li>
          <li>Click on one corner of the area</li>
          <li>Move the mouse to the opposite corner</li>
          <li>Click again to create the annotation</li>
        </ol>

        <h3>Linear Annotations</h3>
        <p>Best for elongated structures. Creates a rotated rectangle defined by its centerline.</p>
        <ol>
          <li>Select <strong>Linear</strong> shape (or press <code>3</code>)</li>
          <li>Click to set the start of the centerline</li>
          <li>Move the mouse and click to set the end of the centerline</li>
          <li>Move the mouse perpendicular to set the width</li>
          <li>Click to finalize the annotation</li>
        </ol>
        <p><strong>Tip:</strong> Press <code>Escape</code> at any time to cancel shape creation.</p>

        <hr />

        <h2>Editing Annotations</h2>

        <h3>Selecting an Annotation</h3>
        <ol>
          <li>Switch to <strong>Select</strong> mode (click Select or press <code>V</code>)</li>
          <li>Click on any annotation to select it</li>
          <li>The selected annotation shows resize handles</li>
        </ol>

        <h3>Moving Annotations</h3>
        <ul>
          <li>In Select mode, click and drag a selected annotation to move it</li>
          <li>The annotation will follow your mouse</li>
        </ul>

        <h3>Resizing Annotations</h3>
        <p>Each shape type has different resize handles:</p>
        <ul>
          <li><strong>Circle:</strong> Drag the edge handle to change the radius</li>
          <li><strong>Rectangle:</strong> Drag any corner handle to resize</li>
          <li><strong>Linear:</strong> Drag start/end points to change length, or drag the width handle</li>
        </ul>

        <h3>Editing Properties</h3>
        <p>With an annotation selected, use the <strong>Property Panel</strong> on the right to:</p>
        <ul>
          <li><strong>Label:</strong> Give the annotation a descriptive name</li>
          <li><strong>Notes:</strong> Add additional information</li>
          <li><strong>Color:</strong> Change the annotation color</li>
        </ul>

        <h3>Deleting Annotations</h3>
        <ul>
          <li>Select an annotation and press <code>Delete</code> or <code>Backspace</code></li>
          <li>Or click the <strong>Delete</strong> button in the Property Panel</li>
        </ul>

        <hr />

        <h2>Navigation</h2>

        <h3>Zooming</h3>
        <ul>
          <li><strong>Mouse wheel:</strong> Scroll to zoom in/out at the cursor position</li>
          <li><strong>Toolbar:</strong> Use the <code>-</code> and <code>+</code> buttons</li>
          <li><strong>Reset:</strong> Click Reset to fit the image to the view</li>
        </ul>
        <p>Zoom range: 10% to 1000%</p>

        <h3>Panning</h3>
        <ul>
          <li><strong>Pan mode:</strong> Press <code>H</code> or click Pan, then drag to move around</li>
          <li><strong>Middle mouse button:</strong> Drag with the middle button in any mode</li>
        </ul>

        <hr />

        <h2>Saving and Loading Projects</h2>

        <h3>Saving a Project</h3>
        <ol>
          <li>Click <strong>Save</strong> in the toolbar</li>
          <li>Choose a location and filename</li>
          <li>Projects are saved as <code>.fol</code> files</li>
        </ol>
        <p>The <code>.fol</code> format preserves:</p>
        <ul>
          <li>All images in the project</li>
          <li>All annotations with their properties</li>
          <li>Zoom and pan state for each image</li>
        </ul>

        <h3>Loading a Project</h3>
        <ol>
          <li>Click <strong>Load</strong> in the toolbar</li>
          <li>Select a <code>.fol</code> file</li>
          <li>All images and annotations will be restored</li>
        </ol>

        <hr />

        <h2>Keyboard Shortcuts</h2>
        <table>
          <thead>
            <tr>
              <th>Action</th>
              <th>Shortcut</th>
            </tr>
          </thead>
          <tbody>
            <tr><td>Create mode</td><td><code>C</code></td></tr>
            <tr><td>Select mode</td><td><code>V</code></td></tr>
            <tr><td>Pan mode</td><td><code>H</code></td></tr>
            <tr><td>Circle shape</td><td><code>1</code></td></tr>
            <tr><td>Rectangle shape</td><td><code>2</code></td></tr>
            <tr><td>Linear shape</td><td><code>3</code></td></tr>
            <tr><td>Toggle shapes</td><td><code>O</code></td></tr>
            <tr><td>Toggle labels</td><td><code>L</code></td></tr>
            <tr><td>Undo</td><td><code>Ctrl+Z</code></td></tr>
            <tr><td>Redo</td><td><code>Ctrl+Shift+Z</code></td></tr>
            <tr><td>Delete annotation</td><td><code>Delete</code> / <code>Backspace</code></td></tr>
            <tr><td>Cancel / Deselect</td><td><code>Escape</code></td></tr>
            <tr><td>Toggle help</td><td><code>?</code></td></tr>
            <tr><td>Next image</td><td><code>Ctrl+Tab</code></td></tr>
            <tr><td>Previous image</td><td><code>Ctrl+Shift+Tab</code></td></tr>
          </tbody>
        </table>

        <hr />

        <h2>Tips and Best Practices</h2>
        <ul>
          <li><strong>Use keyboard shortcuts</strong> for faster workflow - <code>C</code> for create, <code>V</code> for select, <code>1-3</code> for shapes</li>
          <li><strong>Middle-click and drag</strong> to pan while in any mode</li>
          <li><strong>Zoom to details</strong> with the mouse wheel before creating precise annotations</li>
          <li><strong>Add notes</strong> to annotations for additional context</li>
          <li><strong>Save frequently</strong> to avoid losing work</li>
          <li><strong>Use the Screenshot button</strong> to capture the current canvas view</li>
        </ul>

        <hr />

        <h2>Toolbar Overview</h2>
        <p>From left to right:</p>
        <ol>
          <li><strong>File Operations:</strong> Open Image, Load, Save</li>
          <li><strong>Modes:</strong> Create, Select, Pan</li>
          <li><strong>Shapes:</strong> Circle, Rectangle, Linear</li>
          <li><strong>Zoom:</strong> -, percentage, +, Reset</li>
          <li><strong>History:</strong> Undo, Redo</li>
          <li><strong>View:</strong> Shapes toggle, Labels toggle</li>
          <li><strong>Clear All:</strong> Remove all annotations (caution!)</li>
          <li><strong>Screenshot:</strong> Save current canvas view</li>
          <li><strong>Help:</strong> Open this guide</li>
        </ol>
      </div>
    </div>
  );
};
