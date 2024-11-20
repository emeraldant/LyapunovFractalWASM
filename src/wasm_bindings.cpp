#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <emscripten/emscripten.h>
#include "lyapunov_fractal.hpp"
#include <cstdio>
#include <vector>

using namespace emscripten;

// Required main function for WASM module
extern "C" {
    EMSCRIPTEN_KEEPALIVE
    int main() {
        printf("WASM module initialized\n");
        return 0;
    }

    EMSCRIPTEN_KEEPALIVE
    int testFunction() {
        printf("Test function called!\n");
        return 42;
    }
}

// Progress callback using Emscripten's EM_ASM
void reportProgress(double progress) {
    EM_ASM({
        if (typeof window !== 'undefined' && window.updateComputationProgress) {
            window.updateComputationProgress($0);
        }
    }, progress);
}

// Helper function to convert JavaScript parameters to C++ struct
lyapunov::FractalComputer::ComputationParams parseParams(const val& jsParams) {
    printf("Parsing computation parameters...\n");
    
    lyapunov::FractalComputer::ComputationParams params;
    
    try {
        // Log all available properties
        auto properties = val::global("Object").call<val>("keys", jsParams);
        printf("Available parameters: ");
        for (int i = 0; i < properties["length"].as<int>(); i++) {
            std::string prop = properties[i].as<std::string>();
            printf("%s, ", prop.c_str());
        }
        printf("\n");

        if (jsParams.hasOwnProperty("min_x")) {
            params.min_x = jsParams["min_x"].as<double>();
            printf("min_x: %f\n", params.min_x);
        }
        if (jsParams.hasOwnProperty("max_x")) {
            params.max_x = jsParams["max_x"].as<double>();
            printf("max_x: %f\n", params.max_x);
        }
        if (jsParams.hasOwnProperty("min_y")) {
            params.min_y = jsParams["min_y"].as<double>();
            printf("min_y: %f\n", params.min_y);
        }
        if (jsParams.hasOwnProperty("max_y")) {
            params.max_y = jsParams["max_y"].as<double>();
            printf("max_y: %f\n", params.max_y);
        }
        if (jsParams.hasOwnProperty("width")) {
            params.width = jsParams["width"].as<int>();
            printf("width: %d\n", params.width);
        }
        if (jsParams.hasOwnProperty("height")) {
            params.height = jsParams["height"].as<int>();
            printf("height: %d\n", params.height);
        }
        if (jsParams.hasOwnProperty("sequence")) {
            params.sequence = jsParams["sequence"].as<std::string>();
            printf("sequence: %s\n", params.sequence.c_str());
        }
        if (jsParams.hasOwnProperty("iterations")) {
            params.iterations = jsParams["iterations"].as<int>();
            printf("iterations: %d\n", params.iterations);
        }
        if (jsParams.hasOwnProperty("warmup")) {
            params.warmup = jsParams["warmup"].as<int>();
            printf("warmup: %d\n", params.warmup);
        }
        
        printf("Parameters parsed successfully\n");
    } catch (const std::exception& e) {
        printf("Error parsing parameters: %s\n", e.what());
        throw;
    }
    
    return params;
}

// Function to update progress in JavaScript
EM_JS(void, updateProgress, (double progress), {
    if (typeof window !== 'undefined' && window.updateComputationProgress) {
        window.updateComputationProgress(progress);
    }
});

// WASM-exposed function using embind
val computeLyapunovFractal(const val& jsParams) {
    printf("computeLyapunovFractal called\n");
    
    try {
        printf("Parsing parameters...\n");
        auto params = parseParams(jsParams);
        
        printf("Creating fractal computer...\n");
        lyapunov::FractalComputer computer;
        
        printf("Computing fractal data...\n");
        auto fractal_data = computer.computeFractal(params, reportProgress);
        printf("Fractal data computed, size: %zu\n", fractal_data.size());
        
        printf("Normalizing data...\n");
        auto normalized_data = lyapunov::normalizeData(fractal_data);
        printf("Data normalized, size: %zu\n", normalized_data.size());
        
        printf("Creating typed array view...\n");
        auto result = val(typed_memory_view(normalized_data.size(), normalized_data.data()));
        printf("Returning result\n");
        return result;
    } catch (const std::exception& e) {
        printf("Error in computeLyapunovFractal: %s\n", e.what());
        throw;
    }
}

// Bind the functions to JavaScript
EMSCRIPTEN_BINDINGS(lyapunov_module) {
    function("testFunction", &testFunction);
    register_vector<double>("Vector");
    function("computeLyapunovFractal", &computeLyapunovFractal);
}
