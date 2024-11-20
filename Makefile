# Makefile for Lyapunov Fractal WebAssembly Project

# Emscripten Compiler Settings
CXX = em++
CXXFLAGS = -std=c++23 -O3 -s WASM=1 -s ALLOW_MEMORY_GROWTH=1 \
           -s EXPORTED_RUNTIME_METHODS=['ccall','cwrap'] \
           -s EXPORTED_FUNCTIONS=['_malloc','_free'] \
           -s ENVIRONMENT='web' \
           -s MODULARIZE=1 \
           -s EXPORT_NAME='createLyapunovModule' \
           -s NO_EXIT_RUNTIME=1 \
           -s "EXTRA_EXPORTED_RUNTIME_METHODS=['ccall', 'cwrap']" \
           -s ASSERTIONS=1 \
           -s ASYNCIFY=1 \
           --bind

# Directories
SRC_DIR = src
BUILD_DIR = wasm

# Source Files
SOURCES = $(SRC_DIR)/wasm_bindings.cpp

# Output Files
TARGET = $(BUILD_DIR)/lyapunov_fractal.js

all: $(BUILD_DIR) $(TARGET)
	@echo "Build complete. Output files:"
	@ls -l $(BUILD_DIR)

# Compile C++ to WebAssembly
$(TARGET): $(SRC_DIR)/wasm_bindings.cpp $(SRC_DIR)/lyapunov_fractal.hpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(SRC_DIR)/wasm_bindings.cpp -o $(TARGET)

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)
