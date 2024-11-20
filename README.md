# Lyapunov Fractal Visualizer

## Overview
A high-performance web-based Lyapunov fractal visualizer built with C++23 and WebAssembly, showcasing the beauty of chaos theory and playing around with WASM.

## Features
- High-performance fractal computation using C++23
- WebAssembly integration for near-native execution speed
- Interactive visualization with stunning color gradients
- Educational content explaining chaos theory and Lyapunov fractals

## Prerequisites
- Emscripten (for WebAssembly compilation)
- Modern web browser with WebAssembly support
- C++23 compatible compiler

## Build Instructions
1. Install Emscripten
2. Clone the repository
3. Run `make` to compile the WebAssembly module
4. Host the `web/` directory on a web server, such as `python3 -m http.server 8000`
4. Open `http://localhost:8000` in a web browser

## Project Structure
- `src/`: C++ source files
- `web/`: Web interface files
- `wasm/`: WebAssembly compilation output
- `docs/`: Documentation and educational content

## Contributing
Contributions are welcome! 



