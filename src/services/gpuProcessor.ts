/**
 * WebGL-accelerated image processing for BLOB detection.
 * Uses GPU shaders for fast grayscale conversion and thresholding.
 */

// Vertex shader - passes through texture coordinates
const VERTEX_SHADER_SOURCE = `
  attribute vec2 a_position;
  attribute vec2 a_texCoord;
  varying vec2 v_texCoord;

  void main() {
    gl_Position = vec4(a_position, 0.0, 1.0);
    v_texCoord = a_texCoord;
  }
`;

// Fragment shader for grayscale conversion
const GRAYSCALE_FRAGMENT_SHADER = `
  precision mediump float;
  uniform sampler2D u_image;
  varying vec2 v_texCoord;

  void main() {
    vec4 color = texture2D(u_image, v_texCoord);
    float gray = 0.299 * color.r + 0.587 * color.g + 0.114 * color.b;
    gl_FragColor = vec4(gray, gray, gray, 1.0);
  }
`;

// Fragment shader for thresholding
const THRESHOLD_FRAGMENT_SHADER = `
  precision mediump float;
  uniform sampler2D u_image;
  uniform float u_threshold;
  uniform bool u_darkBlobs;
  varying vec2 v_texCoord;

  void main() {
    vec4 color = texture2D(u_image, v_texCoord);
    float gray = 0.299 * color.r + 0.587 * color.g + 0.114 * color.b;
    float threshold = u_threshold / 255.0;

    float result;
    if (u_darkBlobs) {
      result = gray < threshold ? 1.0 : 0.0;
    } else {
      result = gray >= threshold ? 1.0 : 0.0;
    }

    gl_FragColor = vec4(result, result, result, 1.0);
  }
`;

/**
 * GPU Processor class for WebGL-accelerated image processing.
 */
export class GPUProcessor {
  private canvas: OffscreenCanvas | HTMLCanvasElement | null = null;
  private gl: WebGLRenderingContext | null = null;
  private grayscaleProgram: WebGLProgram | null = null;
  private thresholdProgram: WebGLProgram | null = null;
  private positionBuffer: WebGLBuffer | null = null;
  private texCoordBuffer: WebGLBuffer | null = null;
  private texture: WebGLTexture | null = null;
  private _isInitialized = false;

  /**
   * Check if WebGL is available.
   */
  static isAvailable(): boolean {
    try {
      // Try OffscreenCanvas first (preferred for performance)
      if (typeof OffscreenCanvas !== 'undefined') {
        const canvas = new OffscreenCanvas(1, 1);
        const gl = canvas.getContext('webgl');
        return gl !== null;
      }

      // Fallback to regular canvas
      const canvas = document.createElement('canvas');
      const gl = canvas.getContext('webgl');
      return gl !== null;
    } catch {
      return false;
    }
  }

  /**
   * Initialize WebGL context and shaders.
   */
  initialize(): boolean {
    if (this._isInitialized) return true;

    try {
      // Create canvas
      if (typeof OffscreenCanvas !== 'undefined') {
        this.canvas = new OffscreenCanvas(1, 1);
      } else {
        this.canvas = document.createElement('canvas');
      }

      // Get WebGL context
      this.gl = this.canvas.getContext('webgl', {
        preserveDrawingBuffer: true,
        antialias: false,
      }) as WebGLRenderingContext | null;

      if (!this.gl) {
        console.warn('WebGL not available');
        return false;
      }

      // Compile shaders and create programs
      this.grayscaleProgram = this.createProgram(
        VERTEX_SHADER_SOURCE,
        GRAYSCALE_FRAGMENT_SHADER
      );
      this.thresholdProgram = this.createProgram(
        VERTEX_SHADER_SOURCE,
        THRESHOLD_FRAGMENT_SHADER
      );

      if (!this.grayscaleProgram || !this.thresholdProgram) {
        return false;
      }

      // Create buffers
      this.createBuffers();

      this._isInitialized = true;
      return true;
    } catch (error) {
      console.error('Failed to initialize GPU processor:', error);
      return false;
    }
  }

  /**
   * Check if GPU processor is initialized.
   */
  get isInitialized(): boolean {
    return this._isInitialized;
  }

  /**
   * Create and compile a shader.
   */
  private compileShader(type: number, source: string): WebGLShader | null {
    const gl = this.gl!;
    const shader = gl.createShader(type);
    if (!shader) return null;

    gl.shaderSource(shader, source);
    gl.compileShader(shader);

    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      console.error('Shader compile error:', gl.getShaderInfoLog(shader));
      gl.deleteShader(shader);
      return null;
    }

    return shader;
  }

  /**
   * Create a shader program from vertex and fragment shaders.
   */
  private createProgram(
    vertexSource: string,
    fragmentSource: string
  ): WebGLProgram | null {
    const gl = this.gl!;

    const vertexShader = this.compileShader(gl.VERTEX_SHADER, vertexSource);
    const fragmentShader = this.compileShader(gl.FRAGMENT_SHADER, fragmentSource);

    if (!vertexShader || !fragmentShader) return null;

    const program = gl.createProgram();
    if (!program) return null;

    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      console.error('Program link error:', gl.getProgramInfoLog(program));
      gl.deleteProgram(program);
      return null;
    }

    // Clean up shaders (attached to program, no longer needed separately)
    gl.deleteShader(vertexShader);
    gl.deleteShader(fragmentShader);

    return program;
  }

  /**
   * Create vertex and texture coordinate buffers.
   */
  private createBuffers(): void {
    const gl = this.gl!;

    // Full-screen quad positions (two triangles)
    const positions = new Float32Array([
      -1, -1,
       1, -1,
      -1,  1,
      -1,  1,
       1, -1,
       1,  1,
    ]);

    this.positionBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);

    // Texture coordinates (flipped Y for image coordinates)
    const texCoords = new Float32Array([
      0, 1,
      1, 1,
      0, 0,
      0, 0,
      1, 1,
      1, 0,
    ]);

    this.texCoordBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.texCoordBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, texCoords, gl.STATIC_DRAW);
  }

  /**
   * Upload image data to GPU texture.
   */
  private uploadTexture(imageData: ImageData): void {
    const gl = this.gl!;

    if (!this.texture) {
      this.texture = gl.createTexture();
    }

    gl.bindTexture(gl.TEXTURE_2D, this.texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.RGBA,
      gl.RGBA,
      gl.UNSIGNED_BYTE,
      imageData
    );
  }

  /**
   * Set up vertex attributes for a program.
   */
  private setupAttributes(program: WebGLProgram): void {
    const gl = this.gl!;

    const positionLocation = gl.getAttribLocation(program, 'a_position');
    gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
    gl.enableVertexAttribArray(positionLocation);
    gl.vertexAttribPointer(positionLocation, 2, gl.FLOAT, false, 0, 0);

    const texCoordLocation = gl.getAttribLocation(program, 'a_texCoord');
    gl.bindBuffer(gl.ARRAY_BUFFER, this.texCoordBuffer);
    gl.enableVertexAttribArray(texCoordLocation);
    gl.vertexAttribPointer(texCoordLocation, 2, gl.FLOAT, false, 0, 0);
  }

  /**
   * Process image with threshold on GPU.
   * Returns binary image data (grayscale, where 255 = foreground, 0 = background).
   */
  processImage(
    imageData: ImageData,
    threshold: number,
    darkBlobs: boolean
  ): Uint8Array {
    if (!this._isInitialized || !this.gl || !this.thresholdProgram) {
      throw new Error('GPU processor not initialized');
    }

    const gl = this.gl;
    const canvas = this.canvas!;
    const width = imageData.width;
    const height = imageData.height;

    // Resize canvas to match image
    canvas.width = width;
    canvas.height = height;
    gl.viewport(0, 0, width, height);

    // Upload image to texture
    this.uploadTexture(imageData);

    // Use threshold program
    gl.useProgram(this.thresholdProgram);
    this.setupAttributes(this.thresholdProgram);

    // Set uniforms
    const imageLocation = gl.getUniformLocation(this.thresholdProgram, 'u_image');
    const thresholdLocation = gl.getUniformLocation(this.thresholdProgram, 'u_threshold');
    const darkBlobsLocation = gl.getUniformLocation(this.thresholdProgram, 'u_darkBlobs');

    gl.uniform1i(imageLocation, 0);
    gl.uniform1f(thresholdLocation, threshold);
    gl.uniform1i(darkBlobsLocation, darkBlobs ? 1 : 0);

    // Draw
    gl.drawArrays(gl.TRIANGLES, 0, 6);

    // Read pixels
    const pixels = new Uint8Array(width * height * 4);
    gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);

    // Extract grayscale channel (R channel since it's grayscale)
    const result = new Uint8Array(width * height);
    for (let i = 0; i < result.length; i++) {
      result[i] = pixels[i * 4];
    }

    return result;
  }

  /**
   * Convert image to grayscale on GPU.
   */
  toGrayscale(imageData: ImageData): Uint8Array {
    if (!this._isInitialized || !this.gl || !this.grayscaleProgram) {
      throw new Error('GPU processor not initialized');
    }

    const gl = this.gl;
    const canvas = this.canvas!;
    const width = imageData.width;
    const height = imageData.height;

    // Resize canvas to match image
    canvas.width = width;
    canvas.height = height;
    gl.viewport(0, 0, width, height);

    // Upload image to texture
    this.uploadTexture(imageData);

    // Use grayscale program
    gl.useProgram(this.grayscaleProgram);
    this.setupAttributes(this.grayscaleProgram);

    // Set uniforms
    const imageLocation = gl.getUniformLocation(this.grayscaleProgram, 'u_image');
    gl.uniform1i(imageLocation, 0);

    // Draw
    gl.drawArrays(gl.TRIANGLES, 0, 6);

    // Read pixels
    const pixels = new Uint8Array(width * height * 4);
    gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);

    // Extract grayscale channel
    const result = new Uint8Array(width * height);
    for (let i = 0; i < result.length; i++) {
      result[i] = pixels[i * 4];
    }

    return result;
  }

  /**
   * Clean up GPU resources.
   */
  dispose(): void {
    if (!this.gl) return;

    const gl = this.gl;

    if (this.texture) {
      gl.deleteTexture(this.texture);
      this.texture = null;
    }

    if (this.positionBuffer) {
      gl.deleteBuffer(this.positionBuffer);
      this.positionBuffer = null;
    }

    if (this.texCoordBuffer) {
      gl.deleteBuffer(this.texCoordBuffer);
      this.texCoordBuffer = null;
    }

    if (this.grayscaleProgram) {
      gl.deleteProgram(this.grayscaleProgram);
      this.grayscaleProgram = null;
    }

    if (this.thresholdProgram) {
      gl.deleteProgram(this.thresholdProgram);
      this.thresholdProgram = null;
    }

    this.gl = null;
    this.canvas = null;
    this._isInitialized = false;
  }
}

// Singleton instance
let gpuProcessorInstance: GPUProcessor | null = null;

/**
 * Get the GPU processor singleton instance.
 * Initializes on first call if WebGL is available.
 */
export function getGPUProcessor(): GPUProcessor | null {
  if (gpuProcessorInstance) {
    return gpuProcessorInstance;
  }

  if (!GPUProcessor.isAvailable()) {
    return null;
  }

  gpuProcessorInstance = new GPUProcessor();
  if (!gpuProcessorInstance.initialize()) {
    gpuProcessorInstance = null;
    return null;
  }

  return gpuProcessorInstance;
}
