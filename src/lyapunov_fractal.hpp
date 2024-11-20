#pragma once

#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <emscripten.h>
#include <thread>

namespace lyapunov {
    class FractalComputer {
    public:
        struct ComputationParams {
            double min_x = 0.0;  
            double max_x = 4.0;
            double min_y = 0.0;
            double max_y = 4.0;
            int width = 800;
            int height = 600;
            std::string sequence = "ABABABAB";
            int iterations = 10000;
            int warmup = 100;
        };

        std::vector<double> computeFractal(const ComputationParams& params, void (*progressCallback)(double)) {
            std::vector<double> fractal_data(params.width * params.height);
            const double dx = (params.max_x - params.min_x) / (params.width - 1.0);
            const double dy = (params.max_y - params.min_y) / (params.height - 1.0);
            const double initial_x = 0.5;  
            const double log_2 = std::log(2.0);  

            const int seq_len = params.sequence.length();
            std::vector<bool> is_seq_a(params.iterations + params.warmup);
            for (int i = 0; i < is_seq_a.size(); ++i) {
                is_seq_a[i] = params.sequence[i % seq_len] == 'A';
            }

            #pragma omp parallel for schedule(dynamic) if(params.height > 100)
            for (int y = 0; y < params.height; ++y) {
                double r_y = params.min_y + dy * y;
                
                for (int x = 0; x < params.width; ++x) {
                    double r_x = params.min_x + dx * x;
                    
                    if (r_x <= 0.0 || r_y <= 0.0) {
                        fractal_data[y * params.width + x] = -5.0;
                        continue;
                    }

                    double current_x = initial_x;
                    double lyap = 0.0;
                    int valid_terms = 0;

                    for (int i = 0; i < params.warmup; ++i) {
                        const double r = is_seq_a[i] ? r_x : r_y;
                        current_x = r * current_x * (1.0 - current_x);
                        
                        if (current_x <= 0.0 || current_x >= 1.0 || !std::isfinite(current_x)) {
                            break;
                        }
                    }

                    if (std::isfinite(current_x) && current_x > 0.0 && current_x < 1.0) {
                        for (int i = params.warmup; i < params.warmup + params.iterations; ++i) {
                            const double r = is_seq_a[i] ? r_x : r_y;
                            
                            const double deriv = std::abs(r * (1.0 - 2.0 * current_x));
                            
                            if (deriv > 0.0) {
                                lyap += std::log(deriv);
                                valid_terms++;
                            }
                            
                            current_x = r * current_x * (1.0 - current_x);
                            
                            if (current_x <= 0.0 || current_x >= 1.0 || !std::isfinite(current_x)) {
                                break;
                            }
                        }
                    }

                    fractal_data[y * params.width + x] = valid_terms > 0 ? 
                        lyap / (valid_terms * log_2) : -5.0;  
                }

                if (progressCallback && y % 10 == 0) {
                    progressCallback(static_cast<double>(y) / params.height);
                }
                
                if (y % 50 == 0) {
                    emscripten_sleep(1);
                }
            }

            if (progressCallback) {
                progressCallback(1.0);
            }

            return fractal_data;
        }
    };

    std::vector<double> normalizeData(const std::vector<double>& fractal_data) {
        if (fractal_data.empty()) return {};
        
        alignas(32) std::vector<double> normalized(fractal_data.size());
        
        double min_val = std::numeric_limits<double>::max();
        double max_val = std::numeric_limits<double>::lowest();
        
        #pragma omp parallel for reduction(min:min_val) reduction(max:max_val)
        for (size_t i = 0; i < fractal_data.size(); ++i) {
            const double val = fractal_data[i];
            if (std::isfinite(val) && val > -4.0) {
                min_val = std::min(min_val, val);
                max_val = std::max(max_val, val);
            }
        }
        
        if (min_val >= max_val) return normalized;
        
        const double range = max_val - min_val;
        const double range_inv = 1.0 / range;
        
        #pragma omp parallel for
        for (size_t i = 0; i < fractal_data.size(); ++i) {
            const double val = fractal_data[i];
            if (std::isfinite(val) && val > -4.0) {
                const double t = (val - min_val) * range_inv;
                const double s = std::pow(t, 1.2); 
                normalized[i] = 1.0 / (1.0 + std::exp(-7.0 * (s - 0.5))); 
            } else {
                normalized[i] = 0.0;
            }
        }
        
        return normalized;
    }
}
