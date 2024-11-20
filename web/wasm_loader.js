// WebAssembly Module Loader
let moduleLoadAttempted = false;

async function loadWasmScript() {
    return new Promise((resolve, reject) => {
        const script = document.createElement('script');
        script.src = './wasm/lyapunov_fractal.js';
        script.onload = () => {
            console.log('WASM JavaScript file loaded successfully');
            resolve();
        };
        script.onerror = () => {
            reject(new Error('Failed to load WASM JavaScript file'));
        };
        document.head.appendChild(script);
    });
}

async function createModule() {
    if (moduleLoadAttempted) {
        return Promise.reject(new Error('Module load already attempted'));
    }
    moduleLoadAttempted = true;

    console.log('Starting WASM module load...');
    
    // First load the WASM JavaScript file
    await loadWasmScript();
    
    // Verify WASM binary exists
    const wasmResponse = await fetch('./wasm/lyapunov_fractal.wasm');
    if (!wasmResponse.ok) {
        throw new Error(`WASM binary not found: ${wasmResponse.status}`);
    }
    console.log('WASM binary file found');

    return new Promise((resolve, reject) => {
        // Configure module
        const moduleConfig = {
            locateFile: function(filename) {
                const path = `./wasm/${filename}`;
                console.log('Locating file:', path);
                return path;
            },
            print: function(text) {
                console.log('WASM:', text);
            },
            printErr: function(text) {
                console.error('WASM Error:', text);
            },
            onRuntimeInitialized: function() {
                console.log('WASM runtime initialized');
                resolve(window.Module);
            },
            onAbort: function(what) {
                const error = new Error(`WASM module aborted: ${what}`);
                console.error('WASM aborted:', error);
                reject(error);
            }
        };

        // Create Module with our config
        if (typeof createLyapunovModule !== 'function') {
            reject(new Error('createLyapunovModule not found after loading WASM JavaScript'));
            return;
        }

        createLyapunovModule(moduleConfig).then(module => {
            window.Module = module;
            console.log('Module created successfully');
            
            // Test the module
            if (typeof module._testFunction === 'function') {
                console.log('Test function exists');
                const result = module._testFunction();
                console.log('Test result:', result);
            } else {
                console.warn('Test function not found. Available functions:', 
                    Object.keys(module).filter(k => typeof module[k] === 'function'));
            }
            
            resolve(module);
        }).catch(error => {
            console.error('Error creating module:', error);
            reject(error);
        });
    });
}

// Initialize everything when the DOM is ready
document.addEventListener('DOMContentLoaded', async function() {
    console.log('DOM loaded, initializing module...');
    
    try {
        // First load the WASM module
        const module = await createModule();
        console.log('Module loaded successfully, initializing renderer...');
        
        // Then initialize the fractal renderer
        if (typeof window.initLyapunovFractal === 'function') {
            window.initLyapunovFractal();
        } else {
            throw new Error('initLyapunovFractal function not found');
        }
    } catch (error) {
        console.error('Initialization failed:', error);
        const canvas = document.getElementById('fractal-canvas');
        if (canvas) {
            const ctx = canvas.getContext('2d');
            ctx.fillStyle = '#1e1e1e';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            ctx.fillStyle = '#cf6679';
            ctx.font = '16px Inter';
            ctx.textAlign = 'center';
            ctx.fillText(`Failed to initialize: ${error.message}`, canvas.width / 2, canvas.height / 2);
        }
    }
});
