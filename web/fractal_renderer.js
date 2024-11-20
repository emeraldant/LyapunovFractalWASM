// Color palette configurations
const COLOR_PALETTES = {
    fire: (value) => {
        const r = Math.min(255, value * 5);
        const g = Math.max(0, Math.min(255, (value - 0.2) * 3 * 255));
        const b = Math.max(0, 255 - value * 10);
        return `rgb(${r}, ${g}, ${b})`;
    },
    ocean: (value) => {
        const r = Math.max(0, 100 - value * 200);
        const g = Math.min(255, value * 5 * 255);
        const b = Math.min(255, 255 - value * 3 * 255);
        return `rgb(${r}, ${g}, ${b})`;
    },
    rainbow: (value) => {
        const hue = (1 - value) * 360;
        return `hsl(${hue}, 100%, 50%)`;
    },
    grayscale: (value) => {
        const shade = Math.min(255, value * 255 * 1.5);
        return `rgb(${shade}, ${shade}, ${shade})`;
    }
};

// Helper function to convert HSL to RGB
function hslToRgb(h, s, l) {
    h /= 360;
    let r, g, b;
    if (s === 0) {
        r = g = b = l;
    } else {
        const hue2rgb = (p, q, t) => {
            if (t < 0) t += 1;
            if (t > 1) t -= 1;
            if (t < 1/6) return p + (q - p) * 6 * t;
            if (t < 1/2) return q;
            if (t < 2/3) return p + (q - p) * (2/3 - t) * 6;
            return p;
        };
        const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
        const p = 2 * l - q;
        r = hue2rgb(p, q, h + 1/3);
        g = hue2rgb(p, q, h);
        b = hue2rgb(p, q, h - 1/3);
    }
    return [r, g, b];
}

class LyapunovFractalRenderer {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) {
            throw new Error('Canvas element not found');
        }
        
        this.ctx = this.canvas.getContext('2d');
        
        // Set display size
        const displayWidth = 600;
        const displayHeight = 600;
        this.canvas.style.width = `${displayWidth}px`;
        this.canvas.style.height = `${displayHeight}px`;
        
        // Set actual resolution
        this.canvas.width = 800;
        this.canvas.height = 800;
        
        // Scale context to handle the high DPI resolution
        this.ctx.scale(1, 1);
        
        console.log(`Canvas initialized: ${this.canvas.width}x${this.canvas.height}`);
        
        // Initialize with a dark background
        this.ctx.fillStyle = '#1e1e1e';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    }
    
    showLoadingIndicator() {
        const progressContainer = document.getElementById('progress-container');
        const progressFill = document.getElementById('progress-fill');
        const progressPercentage = document.getElementById('progress-percentage');
        
        if (progressContainer) {
            progressContainer.style.display = 'block';
            progressFill.style.width = '0%';
            progressPercentage.textContent = '0%';
        }
    }
    
    hideLoadingIndicator() {
        const progressContainer = document.getElementById('progress-container');
        const progressFill = document.getElementById('progress-fill');
        const progressPercentage = document.getElementById('progress-percentage');
        
        if (progressContainer) {
            // Animate to 100% before hiding
            progressFill.style.width = '100%';
            progressPercentage.textContent = '100%';
            
            // Hide after a short delay
            setTimeout(() => {
                progressContainer.style.display = 'none';
                // Reset progress elements
                progressFill.style.width = '0%';
                progressPercentage.textContent = '0%';
            }, 500);
        }
    }
    
    updateProgress(progress) {
        const progressFill = document.getElementById('progress-fill');
        const progressPercentage = document.getElementById('progress-percentage');
        
        if (progressFill && progressPercentage) {
            // Smooth progress updates
            const percentage = Math.round(progress * 100);
            progressFill.style.transition = 'width 0.3s ease-in-out';
            progressFill.style.width = `${percentage}%`;
            progressPercentage.textContent = `${percentage}%`;
        }
    }
    
    showError(message) {
        this.ctx.fillStyle = '#1e1e1e';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.fillStyle = '#cf6679';
        this.ctx.font = '16px Inter';
        this.ctx.textAlign = 'center';
        this.ctx.fillText(message, this.canvas.width / 2, this.canvas.height / 2);
    }
    
    async generateFractal() {
        try {
            const sequence = document.getElementById('sequence').value || 'AB';
            const iterations = parseInt(document.getElementById('iterations').value) || 1000;
            const colorScheme = document.getElementById('color-scheme').value;
            
            console.log('Parameters:', { sequence, iterations, colorScheme });
            
            // Show loading indicator
            this.showLoadingIndicator();
            
            // Compute fractal
            const result = await window.Module.computeLyapunovFractal({
                width: this.canvas.width,
                height: this.canvas.height,
                sequence: sequence,
                iterations: iterations,
                min_x: 0.0,
                max_x: 4.0,
                min_y: 0.0,
                max_y: 4.0,
                warmup: 100
            });
            
            // Wait for data to be fully computed
            const data = await result;
            console.log('Fractal data received, length:', data.length);
            
            // Render the result
            if (data && data.length > 0) {
                this.renderFractal(data, colorScheme);
                console.log('Fractal rendered successfully');
                // Smoothly hide loading indicator
                this.hideLoadingIndicator();
            } else {
                throw new Error('No fractal data received');
            }
            
        } catch (error) {
            console.error('Error generating fractal:', error);
            this.showError(error.message);
            this.hideLoadingIndicator();
        }
    }
    
    renderFractal(data, colorScheme) {
        console.log('Rendering fractal with color scheme:', colorScheme);
        
        const imageData = this.ctx.createImageData(this.canvas.width, this.canvas.height);
        const colorFn = this.getColorFunction(colorScheme);
        
        if (!data || data.length === 0) {
            console.error('No data to render');
            return;
        }

        // Apply post-processing effects
        const processed = this.applyPostProcessing(data);
        
        for (let i = 0; i < processed.length; i++) {
            const color = colorFn(processed[i]);
            const idx = i * 4;
            imageData.data[idx] = color[0];     // R
            imageData.data[idx + 1] = color[1]; // G
            imageData.data[idx + 2] = color[2]; // B
            imageData.data[idx + 3] = 255;      // A
        }
        
        this.ctx.putImageData(imageData, 0, 0);
        
        // Add glow effect
        this.applyGlow();
        console.log('Fractal rendered successfully');
    }

    getColorFunction(scheme) {
        const schemes = {
            cosmic: (t) => {
                // Black -> Purple -> White -> Purple -> Black
                const mid = Math.abs(t - 0.5) * 2;  // 0->1->0 as t goes 0->0.5->1
                const phase = mid * Math.PI;
                const intensity = 1 - mid;  // Peak at center (0.5)
                const r = Math.floor(255 * (intensity * 0.8 + 0.2 * Math.sin(phase)));
                const g = Math.floor(255 * (intensity * 0.3));
                const b = Math.floor(255 * (intensity * 0.9 + 0.1 * Math.cos(phase)));
                return [r, g, b];
            },
            nebula: (t) => {
                // Black -> Pink -> White -> Pink -> Black
                const mid = Math.abs(t - 0.5) * 2;
                const intensity = 1 - mid;
                const r = Math.floor(255 * (intensity * 0.9 + 0.1 * Math.pow(t, 1.5)));
                const g = Math.floor(255 * (intensity * 0.5));
                const b = Math.floor(255 * (intensity * 0.8));
                return [r, g, b];
            },
            aurora: (t) => {
                // Black -> Green -> White -> Green -> Black
                const mid = Math.abs(t - 0.5) * 2;
                const intensity = 1 - mid;
                const phase = mid * Math.PI;
                const r = Math.floor(255 * (intensity * 0.6));
                const g = Math.floor(255 * (intensity * 0.9 + 0.1 * Math.sin(phase)));
                const b = Math.floor(255 * (intensity * 0.7));
                return [r, g, b];
            },
            solarflare: (t) => {
                // Black -> Orange -> White -> Orange -> Black
                const mid = Math.abs(t - 0.5) * 2;
                const intensity = 1 - mid;
                const r = Math.floor(255 * (intensity * 0.95));
                const g = Math.floor(255 * (intensity * 0.6));
                const b = Math.floor(255 * (intensity * 0.2));
                return [r, g, b];
            },
            blackhole: (t) => {
                // Black -> Blue -> White -> Blue -> Black
                const mid = Math.abs(t - 0.5) * 2;
                const intensity = 1 - mid;
                const phase = mid * Math.PI;
                const r = Math.floor(255 * (intensity * 0.5));
                const g = Math.floor(255 * (intensity * 0.7));
                const b = Math.floor(255 * (intensity * 0.9 + 0.1 * Math.sin(phase)));
                return [r, g, b];
            },
            psychedelic: (t) => {
                // Black -> Rainbow -> White -> Rainbow -> Black
                const mid = Math.abs(t - 0.5) * 2;
                const intensity = 1 - mid;
                const phase = t * Math.PI * 6;
                const r = Math.floor(255 * (intensity * (0.5 + 0.5 * Math.sin(phase))));
                const g = Math.floor(255 * (intensity * (0.5 + 0.5 * Math.sin(phase + 2 * Math.PI / 3))));
                const b = Math.floor(255 * (intensity * (0.5 + 0.5 * Math.sin(phase + 4 * Math.PI / 3))));
                return [r, g, b];
            },
            rainbow: (t) => {
                // Black -> Rainbow -> White -> Rainbow -> Black
                const mid = Math.abs(t - 0.5) * 2;
                const intensity = 1 - mid;
                const hue = t * 360;
                const [r, g, b] = hslToRgb(hue, 1, 0.5);
                return [
                    Math.floor(255 * (intensity * r)),
                    Math.floor(255 * (intensity * g)),
                    Math.floor(255 * (intensity * b))
                ];
            }
        };
        return schemes[scheme] || schemes.cosmic;
    }

    applyPostProcessing(data) {
        const processed = new Array(data.length);
        const width = this.canvas.width;
        const height = this.canvas.height;
        
        // Apply smoothing and edge enhancement
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const idx = y * width + x;
                let sum = data[idx];
                let count = 1;
                
                // Average with neighbors
                for (let dy = -1; dy <= 1; dy++) {
                    for (let dx = -1; dx <= 1; dx++) {
                        if (dx === 0 && dy === 0) continue;
                        const nx = x + dx;
                        const ny = y + dy;
                        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                            sum += data[ny * width + nx];
                            count++;
                        }
                    }
                }
                processed[idx] = sum / count;
            }
        }
        return processed;
    }

    applyGlow() {
        // Multi-layer glow for space effect
        this.ctx.save();
        
        // Outer nebula glow
        this.ctx.globalCompositeOperation = 'screen';
        this.ctx.filter = 'blur(6px)';
        this.ctx.globalAlpha = 0.3;
        this.ctx.drawImage(this.canvas, 0, 0);
        
        // Inner bright core
        this.ctx.filter = 'blur(2px)';
        this.ctx.globalAlpha = 0.4;
        this.ctx.drawImage(this.canvas, 0, 0);
        
        // Star-like highlights
        this.ctx.filter = 'none';
        this.ctx.globalAlpha = 0.1;
        this.ctx.drawImage(this.canvas, 0, 0);
        
        this.ctx.restore();
    }
}

// Initialize everything when the page loads
window.initLyapunovFractal = function() {
    console.log('Initializing Lyapunov Fractal Renderer');
    const renderer = new LyapunovFractalRenderer('fractal-canvas');
    
    // Add event listener to the generate button
    const generateButton = document.getElementById('generate');
    if (generateButton) {
        generateButton.addEventListener('click', () => {
            renderer.generateFractal();
        });
    }
    
    // Generate initial fractal
    renderer.generateFractal();
};

// Export the initialization function
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { initLyapunovFractal };
}

// Test WebAssembly module loading
async function testWasmModule() {
    try {
        console.log('Testing WASM module...');
        const result = Module.testFunction();
        console.log('Test function result:', result);
        if (result === 42) {
            console.log('WASM module test successful!');
            return true;
        } else {
            console.error('WASM module test failed: unexpected result');
            return false;
        }
    } catch (error) {
        console.error('WASM module test failed:', error);
        return false;
    }
}

// Add global error handler
window.addEventListener('error', function(event) {
    console.error('Global error:', event.error);
    const renderer = new LyapunovFractalRenderer('fractal-canvas');
    renderer.showError('An unexpected error occurred. Check the console for details.');
});

window.updateComputationProgress = (progress) => {
    if (window.fractalRenderer) {
        window.fractalRenderer.updateProgress(progress);
    }
};
