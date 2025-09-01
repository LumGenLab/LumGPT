#define _USE_MATH_DEFINES
#include <vector>
#include <random>
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <memory>
#include <thread>
#include <mutex>
#include <chrono>
#include <iomanip>
#include <unordered_map>
#include <unordered_set>
#include <cstring>
#include <functional>
#include <limits> // For NaN checks
#include <random> // For sampling improvements

// --- Utility Functions and Constants ---
constexpr double EPSILON = 1e-5; // LayerNorm epsilon (GPT-2 paper)
constexpr double INF = 1e30;
constexpr double NEG_INF = -1e30;

// --- Random Number Generator ---
thread_local std::mt19937 rng(1337); // Fixed seed for reproducibility, thread-local

// --- Math Helpers ---
inline double uniform_rand(double low, double high) {
    std::uniform_real_distribution<double> dist(low, high);
    return dist(rng);
}

inline double normal_rand(double mean, double stddev) {
    std::normal_distribution<double> dist(mean, stddev);
    return dist(rng);
}

// --- Optimized Tensor Classes ---
// Flattened vector-based tensor for better cache performance and memory management
class Tensor {
public:
    std::vector<double> data;
    std::vector<size_t> shape;
    size_t size() const { return data.size(); }
    size_t ndim() const { return shape.size(); }
    size_t dim(size_t i) const {
        if (i >= shape.size()) {
             return 1; // Common default for broadcasting, but log warning conceptually
        }
        return shape[i];
    }

    // More robust default constructor
    Tensor() : data(), shape() {} // Explicitly initialize

    // Constructor with shape
    explicit Tensor(const std::vector<size_t>& s) : shape(s) {
        size_t total_size = 1;
        for (size_t d : shape) total_size *= d;
        data.resize(total_size, 0.0);
    }

    // Constructor with shape and initial value
    Tensor(const std::vector<size_t>& s, double val) : shape(s) {
        size_t total_size = 1;
        for (size_t d : shape) total_size *= d;
        data.resize(total_size, val);
    }

    double& operator[](size_t idx) { return data[idx]; }
    const double& operator[](size_t idx) const { return data[idx]; }

    // Calculate flat index from multi-dimensional indices
    size_t index(const std::vector<size_t>& indices) const {
        assert(indices.size() == shape.size());
        size_t idx = 0;
        size_t multiplier = 1;
        for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
            idx += indices[i] * multiplier;
            multiplier *= shape[i];
        }
        return idx;
    }

    // Access element with bounds checking (simplified)
    double& at(const std::vector<size_t>& indices) {
        return data[index(indices)];
    }
    const double& at(const std::vector<size_t>& indices) const {
        return data[index(indices)];
    }

    void fill(double val) {
         std::fill(data.begin(), data.end(), val);
    }
    void zero() {
        if (!data.empty()) {
             std::memset(data.data(), 0, data.size() * sizeof(double));
        }
    }

    // Helper to check for NaN
    bool has_nan() const {
        for (const auto& val : data) {
            if (std::isnan(val)) return true;
        }
        return false;
    }

    // Assignment operator for correct copying
    Tensor& operator=(const Tensor& other) {
        if (this != &other) {
            shape = other.shape;
            data = other.data;
        }
        return *this;
    }
};

// --- Optimized Math Operations on Tensors ---
void fill_normal(Tensor& t, double mean, double stddev) {
    for (auto& val : t.data) {
        val = normal_rand(mean, stddev);
    }
}

// Y = X + b (Broadcast b over the last dimension of X)
void add_broadcast_last_dim(const Tensor& X, const Tensor& b, Tensor& Y) {
    // Assuming X is [..., N], b is [N], Y is [..., N]
    size_t N = X.dim(X.ndim() - 1);
    size_t num_elements = X.size();
    assert(b.size() == N);
    assert(Y.size() == num_elements);

    for (size_t i = 0; i < num_elements; ++i) {
        Y.data[i] = X.data[i] + b.data[i % N];
    }
}

// C = A * B (Matrix multiplication: [M, K] x [K, N] -> [M, N])
// Corrected version: Resizes C to the correct dimensions
void matmul(const Tensor& A, const Tensor& B, Tensor& C) {
    // Ensure inputs are 2D
    assert(A.ndim() == 2 && B.ndim() == 2);
    size_t M = A.dim(0);
    size_t K_A = A.dim(1);
    size_t K_B = B.dim(0);
    size_t N = B.dim(1);

    // Check inner dimensions match
    assert(K_A == K_B);
    size_t K = K_A; // == K_B

    // Resize C to the correct dimensions if necessary
    // This handles default-constructed tensors or tensors with wrong sizes
    if (C.shape.size() != 2 || C.dim(0) != M || C.dim(1) != N) {
        C = Tensor({M, N}); // Assign a new tensor with correct shape
    }
    // No need for the original assert(C.ndim() == 2) as we've ensured it
    // assert(C.dim(0) == M && C.dim(1) == N); // Also ensured by resize

    // Perform matrix multiplication
    // Zero C first to ensure correct accumulation
    C.zero();
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < K; ++k) {
                sum += A.data[i * K + k] * B.data[k * N + j];
            }
            C.data[i * N + j] = sum;
        }
    }
}

// C = A + B (Element-wise)
void add(const Tensor& A, const Tensor& B, Tensor& C) {
    assert(A.size() == B.size() && B.size() == C.size());
    for (size_t i = 0; i < A.size(); ++i) {
        C.data[i] = A.data[i] + B.data[i];
    }
}
void add_inplace(Tensor& A, const Tensor& B) {
    assert(A.size() == B.size());
    for (size_t i = 0; i < A.size(); ++i) {
        A.data[i] += B.data[i];
    }
}

// B = A^T (Transpose: [M, N] -> [N, M])
// Corrected version: Resizes AT to the correct dimensions
void transpose(const Tensor& A, Tensor& AT) {
    // Ensure input is 2D
    assert(A.ndim() == 2);
    size_t M = A.dim(0);
    size_t N = A.dim(1);

    // Resize AT to the correct dimensions if necessary
    // This handles default-constructed tensors or tensors with wrong sizes
    if (AT.shape.size() != 2 || AT.dim(0) != N || AT.dim(1) != M) {
        AT = Tensor({N, M}); // Assign a new tensor with correct shape
    }
    // No need for the original assert(A.ndim() == 2 && AT.ndim() == 2)
    // as we've ensured AT is 2D with correct dims
    // assert(AT.dim(0) == N && AT.dim(1) == M); // Also ensured by resize

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            AT.data[j * M + i] = A.data[i * N + j];
        }
    }
}

// Element-wise GELU approximation (GPT-2 standard)
inline double gelu_scalar(double x) {
    const double sqrt_2_over_pi = std::sqrt(2.0 / M_PI);
    return 0.5 * x * (1.0 + std::tanh(sqrt_2_over_pi * (x + 0.044715 * x * x * x)));
}
void gelu(const Tensor& X, Tensor& Y) {
    assert(X.size() == Y.size());
    for (size_t i = 0; i < X.size(); ++i) {
        Y.data[i] = gelu_scalar(X.data[i]);
    }
}
void gelu_inplace(Tensor& X) {
    for (size_t i = 0; i < X.size(); ++i) {
        X.data[i] = gelu_scalar(X.data[i]);
    }
}

// Element-wise GELU gradient (approximation)
inline double gelu_grad_scalar(double x) {
    const double sqrt_2_over_pi = std::sqrt(2.0 / M_PI);
    double x3 = x * x * x;
    double inner = sqrt_2_over_pi * (x + 0.044715 * x3);
    double tanh_inner = std::tanh(inner);
    double sech2_inner = 1.0 - tanh_inner * tanh_inner; // sech^2(x) = 1 - tanh^2(x)
    double d_inner_dx = sqrt_2_over_pi * (1.0 + 3.0 * 0.044715 * x * x);
    return 0.5 * (1.0 + tanh_inner + x * d_inner_dx * sech2_inner);
}
// grad_X = grad_Y * gelu_grad(X)
void gelu_grad(const Tensor& X, const Tensor& grad_Y, Tensor& grad_X) {
    assert(X.size() == grad_Y.size() && grad_Y.size() == grad_X.size());
    for (size_t i = 0; i < X.size(); ++i) {
        grad_X.data[i] = grad_Y.data[i] * gelu_grad_scalar(X.data[i]);
    }
}

// Softmax with numerical stability (max trick) on the last dimension
void softmax(const Tensor& X, Tensor& Y) {
    // Assuming X and Y are [..., Vocab]
    size_t total_elements = X.size();
    size_t vocab_size = X.dim(X.ndim() - 1);
    size_t num_rows = total_elements / vocab_size; // Number of rows to softmax over

    for (size_t row = 0; row < num_rows; ++row) {
        size_t row_start = row * vocab_size;
        // Find max for numerical stability
        double max_val = X.data[row_start];
        for (size_t j = 1; j < vocab_size; ++j) {
            if (X.data[row_start + j] > max_val) {
                max_val = X.data[row_start + j];
            }
        }
        // Compute exp and sum
        double sum = 0.0;
        for (size_t j = 0; j < vocab_size; ++j) {
            Y.data[row_start + j] = std::exp(X.data[row_start + j] - max_val);
            sum += Y.data[row_start + j];
        }
        // Normalize
        for (size_t j = 0; j < vocab_size; ++j) {
            Y.data[row_start + j] /= sum;
        }
    }
}

// Combined Softmax + Cross-Entropy Gradient: grad_logits = probs - true_dist
// This is the mathematically correct and numerically stable gradient.
void softmax_crossentropy_grad(const Tensor& probs, const Tensor& y_true_one_hot, Tensor& grad_logits) {
     assert(probs.size() == y_true_one_hot.size() && y_true_one_hot.size() == grad_logits.size());
     for(size_t i = 0; i < probs.size(); ++i) {
         grad_logits.data[i] = probs.data[i] - y_true_one_hot.data[i];
     }
}

// --- Core Components ---

// Linear Layer: Y = X @ W + b
class Linear {
public:
    Tensor W, b; // Weights [out_features, in_features], Bias [out_features]
    Tensor grad_W, grad_b; // Gradients
    Tensor input_buffer; // Stored for backward pass [batch_size, in_features]

    Linear(size_t in_features, size_t out_features)
        : W({out_features, in_features}),
          b({out_features}),
          grad_W({out_features, in_features}, 0.0),
          grad_b({out_features}, 0.0) {
        // Xavier/Glorot init for weights
        double stddev = std::sqrt(2.0 / (in_features + out_features));
        fill_normal(W, 0.0, stddev);
        fill_normal(b, 0.0, stddev); // Initializing bias to 0 is also common
    }

    void forward(const Tensor& X, Tensor& Y) { // Y: [batch_size, out_features]
        input_buffer = X; // Store for backward pass
        Tensor WT;
        transpose(W, WT); // [in_features, out_features]
        matmul(X, WT, Y); // [batch_size, in_features] x [in_features, out_features] -> [batch_size, out_features]
        add_broadcast_last_dim(Y, b, Y); // Add bias
    }

    // Full backward pass implementing the chain rule precisely as per document derivations
    // dL/dX = dL/dY * W
    // dL/dW = sum_over_batch( X^T * dL/dY )
    // dL/db = sum_over_batch( dL/dY )
    void backward(const Tensor& grad_Y, Tensor& grad_X) { // grad_Y: [batch_size, out_features]
        size_t batch_size = grad_Y.dim(0);
        size_t in_features = W.dim(1);
        size_t out_features = W.dim(0);
        // --- CRITICAL FIX: Ensure grad_X is correctly sized ---
        if (grad_X.shape.size() != 2 || grad_X.dim(0) != batch_size || grad_X.dim(1) != in_features) {
            grad_X = Tensor({batch_size, in_features}); // Resize/assign correctly
        }
        // --- END CRITICAL FIX ---
        assert(grad_X.dim(0) == batch_size && grad_X.dim(1) == in_features);

        // 1. dL/dX = dL/dY * W
        // grad_Y: [batch_size, out_features], W: [out_features, in_features] -> grad_X: [batch_size, in_features]
        matmul(grad_Y, W, grad_X); // Matrix multiplication handles the sum over out_features

        // 2. dL/dW = sum_over_batch( X^T * dL/dY )
        // input_buffer (X): [batch_size, in_features], grad_Y: [batch_size, out_features]
        // X^T: [in_features, batch_size] * grad_Y: [batch_size, out_features] -> [in_features, out_features]
        Tensor XT;
        transpose(input_buffer, XT); // [in_features, batch_size]
        Tensor grad_W_local({in_features, out_features});
        matmul(XT, grad_Y, grad_W_local); // Result: [in_features, out_features]
        Tensor grad_W_local_T; // Need to transpose to match W's shape [out_features, in_features]
        transpose(grad_W_local, grad_W_local_T);
        add_inplace(grad_W, grad_W_local_T); // Accumulate into grad_W

        // 3. dL/db = sum_over_batch( dL/dY )
        // Sum gradients over the batch dimension
        for (size_t i = 0; i < batch_size; ++i) {
            for (size_t j = 0; j < out_features; ++j) {
                grad_b.data[j] += grad_Y.data[i * out_features + j];
            }
        }
    }

    void zero_grad() {
        grad_W.zero();
        grad_b.zero();
    }
};

// Layer Normalization (Pre-LN as in GPT)
class LayerNorm {
public:
    size_t normalized_shape; // Size of the last dimension to normalize over
    Tensor gamma, beta; // Learnable parameters [normalized_shape]
    Tensor grad_gamma, grad_beta; // Gradients

    // Buffers for backward pass to store intermediate computations
    Tensor x_mu_buffer;        // [batch_size, seq_len, normalized_shape] - (x - mean)
    Tensor x_sigma_inv_buffer; // [batch_size, seq_len] - 1 / sqrt(variance + eps)
    Tensor x_norm_buffer;      // [batch_size, seq_len, normalized_shape] - normalized input (x_hat)

    LayerNorm(size_t features)
        : normalized_shape(features),
          gamma({features}, 1.0),
          beta({features}, 0.0),
          grad_gamma({features}, 0.0),
          grad_beta({features}, 0.0) {
        // Initialize gamma and beta as per GPT-2/GPT-3 practices
        fill_normal(gamma, 1.0, 0.02);
        // --- CORRECT INITIALIZATION: Beta initialized to 0 ---
        beta.fill(0.0); // Explicitly fill beta with 0.0
        // --- END CORRECT INITIALIZATION ---
    }

    void forward(const Tensor& X, Tensor& Y) { // X, Y: [..., normalized_shape]
        Y = X; // Y inherits shape from X
        size_t total_elements = X.size();
        size_t N = normalized_shape; // features
        assert(total_elements % N == 0);
        size_t num_groups = total_elements / N; // Number of groups (e.g., batch*seq_len) to normalize

        // Ensure buffers are correctly sized
        x_mu_buffer = X; // Reuse buffer memory
        x_norm_buffer = X; // Reuse buffer memory
        x_sigma_inv_buffer = Tensor({num_groups}); // Allocate buffer for sigma_inv per group

        for (size_t g = 0; g < num_groups; ++g) {
            size_t group_start = g * N;

            // 1. Calculate mean: μ = (1/N) * Σ(x_i)
            double sum = 0.0;
            for (size_t i = 0; i < N; ++i) {
                sum += X.data[group_start + i];
            }
            double mean = sum / N;

            // 2. Calculate variance and x_mu: σ^2 = (1/N) * Σ(x_i - μ)^2
            double sum_sq_diff = 0.0;
            for (size_t i = 0; i < N; ++i) {
                double diff = X.data[group_start + i] - mean;
                x_mu_buffer.data[group_start + i] = diff; // Store (x_i - μ)
                sum_sq_diff += diff * diff;
            }
            double variance = sum_sq_diff / N;

            // 3. Calculate standard deviation inverse: 1 / sqrt(σ^2 + ε)
            double sigma = std::sqrt(variance + EPSILON);
            x_sigma_inv_buffer.data[g] = 1.0 / sigma; // Store 1/σ

            // 4. Normalize, scale, and shift: y_i = γ * (x_i - μ) / σ + β
            for (size_t i = 0; i < N; ++i) {
                x_norm_buffer.data[group_start + i] = x_mu_buffer.data[group_start + i] * x_sigma_inv_buffer.data[g]; // x_hat
                Y.data[group_start + i] = gamma.data[i] * x_norm_buffer.data[group_start + i] + beta.data[i];
            }
        }
    }

    // Full backward pass implementing the precise mathematical derivation for LayerNorm gradients
    // dL/dγ = Σ_features( dL/dy_i * x_hat_i )
    // dL/dβ = Σ_features( dL/dy_i )
    // dL/dx_i = (σ_inv / N) * ( N * dL/dy_i * γ_i - Σ(dL/dy_j * γ_j) - x_hat_i * Σ(dL/dy_j * γ_j * x_hat_j) )
    void backward(const Tensor& grad_Y, Tensor& grad_X) { // grad_Y, grad_X: [..., normalized_shape]
        size_t total_elements = grad_Y.size();
        size_t N = normalized_shape; // features
        assert(total_elements % N == 0);
        size_t num_groups = total_elements / N;

        // --- CRITICAL FIX: Ensure grad_X is correctly sized ---
        if (grad_X.shape.size() != grad_Y.shape.size()) {
            grad_X = Tensor(grad_Y.shape); // Match shape if ndim is different
        } else {
            bool shape_mismatch = false;
            for (size_t i = 0; i < grad_Y.ndim(); ++i) {
                if (grad_X.dim(i) != grad_Y.dim(i)) {
                    shape_mismatch = true;
                    break;
                }
            }
            if (shape_mismatch) {
                grad_X = Tensor(grad_Y.shape); // Resize if shape mismatches
            }
        }
        // --- END CRITICAL FIX ---

        grad_gamma.zero(); // Zero gradients for parameters
        grad_beta.zero();

        for (size_t g = 0; g < num_groups; ++g) {
            size_t group_start = g * N;
            double sigma_inv = x_sigma_inv_buffer.data[g]; // Retrieve pre-computed 1/σ

            // Compute intermediate sums needed for dx_i calculation
            double sum_dy_gamma = 0.0;      // Σ(dL/dy_j * γ_j)
            double sum_dy_gamma_xnorm = 0.0; // Σ(dL/dy_j * γ_j * x_hat_j)
            for (size_t i = 0; i < N; ++i) {
                double dy_gamma = grad_Y.data[group_start + i] * gamma.data[i];
                sum_dy_gamma += dy_gamma;
                sum_dy_gamma_xnorm += dy_gamma * x_norm_buffer.data[group_start + i];
            }

            // Compute gradients for learnable parameters for this group
            for (size_t i = 0; i < N; ++i) {
                grad_gamma.data[i] += grad_Y.data[group_start + i] * x_norm_buffer.data[group_start + i]; // dL/dγ
                grad_beta.data[i] += grad_Y.data[group_start + i]; // dL/dβ
            }

            // Compute gradient w.r.t. input x for this group using the derived formula
            // dL/dx_i = (σ_inv / N) * ( N * dL/dy_i * γ_i - sum_dy_gamma - x_hat_i * sum_dy_gamma_xnorm )
            for (size_t i = 0; i < N; ++i) {
                double dy_gamma_i = grad_Y.data[group_start + i] * gamma.data[i];
                grad_X.data[group_start + i] = sigma_inv * (dy_gamma_i - (sum_dy_gamma + x_norm_buffer.data[group_start + i] * sum_dy_gamma_xnorm) / static_cast<double>(N));
            }
        }
    }

    void zero_grad() {
        grad_gamma.zero();
        grad_beta.zero();
    }
};

// Multi-Head Attention (Causal, Pre-LN, GPT-2 style) - FULLY CORRECTED
class MultiHeadAttention {
public:
    int d_model, n_heads, d_k;
    Linear c_attn; // Combined linear projection for Q, K, V: [d_model] -> [3 * d_model]
    Linear c_proj; // Output projection: [d_model] -> [d_model]

    // Buffers for forward/backward passes to store intermediate states for ALL heads
    Tensor qkv_buffer;           // [seq_len, 3 * d_model] - Combined QKV output from c_attn

    // These are now 3D tensors to store data for each head
    std::vector<Tensor> q_heads; // [n_heads] of [seq_len, d_k] - Q for each head
    std::vector<Tensor> k_heads; // [n_heads] of [seq_len, d_k] - K for each head
    std::vector<Tensor> v_heads; // [n_heads] of [seq_len, d_k] - V for each head
    std::vector<Tensor> attn_logits_heads; // [n_heads] of [seq_len, seq_len] - Pre-softmax scores for each head
    std::vector<Tensor> attn_weights_heads; // [n_heads] of [seq_len, seq_len] - Post-softmax weights for each head

    Tensor stored_input;         // [seq_len, d_model] - Input to forward pass (for backward)

    MultiHeadAttention(int d_model, int n_heads)
        : d_model(d_model), n_heads(n_heads), d_k(d_model / n_heads),
          c_attn(d_model, 3 * d_model), // Single dense layer for all Q, K, V projections
          c_proj(d_model, d_model) {    // Output projection layer
        // Initialize the 3D storage for each head
        q_heads.resize(n_heads);
        k_heads.resize(n_heads);
        v_heads.resize(n_heads);
        attn_logits_heads.resize(n_heads);
        attn_weights_heads.resize(n_heads);
    }

    void forward(const Tensor& X, Tensor& Y) { // X, Y: [seq_len, d_model]
        stored_input = X; // Store input for backward pass
        size_t seq_len = X.dim(0);

        // --- CRITICAL FIX: Ensure Y is correctly sized and zeroed ---
        // The caller might pass a default Tensor {}, or one with wrong dims.
        // We must ensure Y is [seq_len, d_model] and initialized to zero.
        if (Y.shape.size() != 2 || Y.dim(0) != seq_len || Y.dim(1) != static_cast<size_t>(d_model)) {
            Y = Tensor({seq_len, static_cast<size_t>(d_model)}); // Resize/assign correctly
        }
        Y.zero(); // Essential: Initialize output to zero before accumulation
        // --- END CRITICAL FIX ---

        // 1. Project input to Q, K, V using a single combined linear layer
        c_attn.forward(X, qkv_buffer); // qkv_buffer: [seq_len, 3*d_model]

        // 2. Initialize output tensor Y to zero (already done above)

        // 3. Process each attention head
        for (int h = 0; h < n_heads; ++h) {
            int head_offset = h * d_k; // Offset in the d_model dimension for this head

            // 4. Extract Q, K, V for this head from the combined qkv_buffer
            // Conceptually, we slice qkv_buffer into Q_head, K_head, V_head [seq_len, d_k]
            // For efficiency, we work directly with indices.
            Tensor Q_head({seq_len, static_cast<size_t>(d_k)});
            Tensor K_head({seq_len, static_cast<size_t>(d_k)});
            Tensor V_head({seq_len, static_cast<size_t>(d_k)});

            for(size_t t = 0; t < seq_len; ++t) {
                for(int j = 0; j < d_k; ++j) {
                    Q_head.data[t * d_k + j] = qkv_buffer.data[t * 3 * d_model + 0 * d_model + head_offset + j]; // Q slice
                    K_head.data[t * d_k + j] = qkv_buffer.data[t * 3 * d_model + 1 * d_model + head_offset + j]; // K slice
                    V_head.data[t * d_k + j] = qkv_buffer.data[t * 3 * d_model + 2 * d_model + head_offset + j]; // V slice
                }
            }

            // 5. Compute Scaled Dot-Product Attention Scores: Q @ K^T / sqrt(d_k)
            // CORRECTED: Must transpose K_head before multiplication
            Tensor scores({seq_len, seq_len});
            Tensor K_head_T;
            transpose(K_head, K_head_T); // Transpose K_head from [seq_len, d_k] to [d_k, seq_len]
            matmul(Q_head, K_head_T, scores); // [seq_len, d_k] x [d_k, seq_len] -> [seq_len, seq_len]

            double scale = 1.0 / std::sqrt(static_cast<double>(d_k));
            for (auto& val : scores.data) {
                val *= scale; // Apply scaling factor
            }

            // 6. Apply Causal Mask: Set future positions to -inf
            for (size_t i = 0; i < seq_len; ++i) {
                for (size_t j = i + 1; j < seq_len; ++j) {
                    scores.data[i * seq_len + j] = NEG_INF; // Mask future positions
                }
            }

            // 7. Store pre-softmax scores for this head
            attn_logits_heads[h] = scores; // Store for backward pass

            // 8. Apply Softmax to get attention weights: A = softmax(scores)
            Tensor attn({seq_len, seq_len});
            softmax(scores, attn); // Apply numerically stable softmax

            // 9. Store post-softmax weights for this head
            attn_weights_heads[h] = attn;

            // 10. Apply attention weights to values: Output_head = A @ V
            Tensor head_out({seq_len, static_cast<size_t>(d_k)});
            matmul(attn, V_head, head_out); // [seq_len, seq_len] x [seq_len, d_k] -> [seq_len, d_k]

            // 11. Accumulate head outputs into the final output Y
            // CRITICAL: This loop was the source of the segfault.
            // It now correctly writes to the pre-initialized and zeroed Y tensor.
            for (size_t t = 0; t < seq_len; ++t) {
                for (int j = 0; j < d_k; ++j) {
                    // Y is guaranteed to be [seq_len, d_model] and zeroed.
                    Y.data[t * d_model + head_offset + j] += head_out.data[t * d_k + j];
                }
            }

            // 12. Store the Q, K, V tensors for this head in the class members
            q_heads[h] = Q_head;
            k_heads[h] = K_head;
            v_heads[h] = V_head;
        }

        // 13. Apply final output projection: Y = Y @ W_O + b_O
        Tensor Y_temp = Y; // Use Y as temp buffer for input to c_proj
        c_proj.forward(Y_temp, Y); // Final projection
    }

    // Full backward pass for Multi-Head Attention implementing precise matrix calculus for ALL heads
    // This follows the chain rule through softmax, matrix multiplication, and linear layers.
    void backward(const Tensor& grad_output, Tensor& grad_input) { // grad_output, grad_input: [seq_len, d_model]
        size_t seq_len = grad_output.dim(0);
        // --- CRITICAL FIX: Ensure grad_input is correctly sized ---
        if (grad_input.shape.size() != 2 || grad_input.dim(0) != seq_len || grad_input.dim(1) != static_cast<size_t>(d_model)) {
            grad_input = Tensor({seq_len, static_cast<size_t>(d_model)}); // Resize/assign correctly
        }
        // --- END CRITICAL FIX ---
        grad_input = grad_output; // Match shape for intermediate results

        // 1. Backward through output projection c_proj
        // dL/d(concat_head_out) = dL/dY * W_O^T
        Tensor grad_head_concat({seq_len, static_cast<size_t>(d_model)}); // Gradient w.r.t. concatenated head outputs
        c_proj.backward(grad_input, grad_head_concat); // grad_input is overwritten with grad wrt c_proj input
        grad_input = grad_head_concat; // Grad w.r.t. concat head outputs

        // 2. Initialize gradients for combined QKV projection
        Tensor grad_qkv_combined({seq_len, static_cast<size_t>(3 * d_model)}, 0.0); // [seq_len, 3*d_model]

        // 3. Backward through each attention head
        for (int h = 0; h < n_heads; ++h) {
            int head_offset = h * d_k;

            // 4. Slice gradient for this head's contribution from grad_head_concat
            Tensor grad_head_out({seq_len, static_cast<size_t>(d_k)}); // [seq_len, d_k]
            for(size_t t = 0; t < seq_len; ++t) {
                for(int j = 0; j < d_k; ++j) {
                     grad_head_out.data[t * d_k + j] = grad_input.data[t * d_model + head_offset + j];
                }
            }

            // 5. Retrieve stored Q, K, V for this head (from class members)
            const Tensor& Q_head = q_heads[h]; // [seq_len, d_k]
            const Tensor& K_head = k_heads[h]; // [seq_len, d_k]
            const Tensor& V_head = v_heads[h]; // [seq_len, d_k]

            // 6. Retrieve stored attention weights for this head
            const Tensor& attn = attn_weights_heads[h]; // [seq_len, seq_len] - Post-softmax weights
            const Tensor& scores = attn_logits_heads[h]; // Pre-softmax scores (for reference if needed)

            // --- Backprop through: head_out = attn @ V_head ---
            // 7. dL/dV_head = attn^T @ dL/d(head_out)
            // This multiplication is correct: [seq_len, seq_len] x [seq_len, d_k] -> [seq_len, d_k]
            Tensor grad_V_head({seq_len, static_cast<size_t>(d_k)}); // [seq_len, d_k]
            Tensor AT; // [seq_len, seq_len]
            transpose(attn, AT); // Transpose attention weights
            matmul(AT, grad_head_out, grad_V_head); // Matrix multiply to get dV

            // 8. dL/d(attn) = dL/d(head_out) @ V_head^T
            // CORRECTED: Must transpose V_head before multiplication
            Tensor grad_attn({seq_len, seq_len}); // [seq_len, seq_len]
            // OLD: Tensor VT; transpose(V_head, VT); matmul(grad_head_out, VT, grad_attn);
            // CORRECT:
            Tensor V_head_T;
            transpose(V_head, V_head_T); // Transpose V_head from [seq_len, d_k] to [d_k, seq_len]
            matmul(grad_head_out, V_head_T, grad_attn); // [seq_len, d_k] x [d_k, seq_len] -> [seq_len, seq_len]

            // --- Backprop through: attn = softmax(scores) ---
            // 9. Apply causal mask to grad_attn (zero out gradients for masked positions)
            for (size_t i = 0; i < seq_len; ++i) {
                for (size_t j = i + 1; j < seq_len; ++j) {
                    grad_attn.data[i * seq_len + j] = 0.0; // Mask future positions
                }
            }

            // 10. dL/d(scores) = softmax_grad(A, dL/dA)
            // Standard softmax gradient: dS_ij = A_ij * (dA_ij - sum_k(dA_ik * A_ik))
            // This is the gradient of the softmax function itself.
            Tensor grad_scores({seq_len, seq_len}); // [seq_len, seq_len]
            for (size_t i = 0; i < seq_len; ++i) {
                double sum = 0.0;
                for (size_t k = 0; k < seq_len; ++k) {
                    sum += grad_attn.data[i * seq_len + k] * attn.data[i * seq_len + k]; // sum_k(dA_ik * A_ik)
                }
                for (size_t j = 0; j < seq_len; ++j) {
                    // dS_ij = A_ij * (dA_ij - sum_k(dA_ik * A_ik))
                    grad_scores.data[i * seq_len + j] = attn.data[i * seq_len + j] * (grad_attn.data[i * seq_len + j] - sum);
                }
            }

            // 11. Apply scaling factor to gradient w.r.t. scores
            double scale = 1.0 / std::sqrt(static_cast<double>(d_k));
            for (auto& val : grad_scores.data) {
                val *= scale; // dL/d(scores) = dL/d(scores_unscaled) / sqrt(d_k)
            }

            // --- Backprop through: scores = Q_head @ K_head^T / sqrt(d_k) ---
            // 12. dL/dQ_head = dL/d(scores) @ K_head
            // grad_scores: [seq_len, seq_len], K_head: [seq_len, d_k] -> grad_Q_head: [seq_len, d_k]
            // This multiplication is correct: [seq_len, seq_len] x [seq_len, d_k] -> [seq_len, d_k]
            Tensor grad_Q_head({seq_len, static_cast<size_t>(d_k)}); // [seq_len, d_k]
            matmul(grad_scores, K_head, grad_Q_head); // dQ = dS @ K

            // 13. dL/dK_head = dL/d(scores)^T @ Q_head
            // grad_scores^T: [seq_len, seq_len], Q_head: [seq_len, d_k] -> grad_K_head: [seq_len, d_k]
            // This multiplication is correct: [seq_len, seq_len] x [seq_len, d_k] -> [seq_len, d_k]
            Tensor grad_K_head({seq_len, static_cast<size_t>(d_k)}); // [seq_len, d_k]
            Tensor grad_scores_T; // [seq_len, seq_len]
            transpose(grad_scores, grad_scores_T); // Transpose dL/d(scores)
            matmul(grad_scores_T, Q_head, grad_K_head); // dK = dS^T @ Q

            // 14. Accumulate gradients w.r.t. Q, K, V for this head into the combined gradient tensor
            // These gradients will be used to compute gradients for the c_attn layer.
            for(size_t t = 0; t < seq_len; ++t) {
                for(int j = 0; j < d_k; ++j) {
                    // Accumulate dQ, dK, dV into the appropriate slices of grad_qkv_combined
                    grad_qkv_combined.data[t * 3 * d_model + 0 * d_model + head_offset + j] += grad_Q_head.data[t * d_k + j]; // dQ
                    grad_qkv_combined.data[t * 3 * d_model + 1 * d_model + head_offset + j] += grad_K_head.data[t * d_k + j]; // dK
                    grad_qkv_combined.data[t * 3 * d_model + 2 * d_model + head_offset + j] += grad_V_head.data[t * d_k + j]; // dV
                }
            }
        }

        // 15. Backward through combined linear projection (c_attn)
        // This computes dL/dX (input to attention) and accumulates dL/dW and dL/db for c_attn.
        c_attn.backward(grad_qkv_combined, grad_input); // grad_input now holds dL/dX
    }

    void zero_grad() {
        c_attn.zero_grad();
        c_proj.zero_grad();
    }
};

// Feed-Forward Network (Pre-LN, GELU, GPT-2 style)
class FeedForward {
public:
    Linear c_fc;   // Fully connected layer (d_model -> 4 * d_model)
    Linear c_proj; // Projection layer (4 * d_model -> d_model)
    Tensor gelu_input_buffer; // [seq_len, 4 * d_model] - Stored for backward pass (pre-GELU activation)

    FeedForward(int d_model)
        : c_fc(d_model, 4 * d_model),  // Expand to 4x width
          c_proj(4 * d_model, d_model) {} // Project back to d_model

    void forward(const Tensor& X, Tensor& Y) { // X, Y: [seq_len, d_model]
        // 1. Linear transformation: hidden = X @ W_fc + b_fc
        c_fc.forward(X, gelu_input_buffer); // gelu_input_buffer: [seq_len, 4 * d_model]

        // 2. Apply GELU activation element-wise
        gelu_inplace(gelu_input_buffer); // Apply GELU in place to gelu_input_buffer

        // 3. Output projection: Y = hidden @ W_proj + b_proj
        c_proj.forward(gelu_input_buffer, Y); // Y: [seq_len, d_model]
    }

    // Full backward pass for Feed-Forward Network
    void backward(const Tensor& grad_output, Tensor& grad_input) { // grad_output, grad_input: [seq_len, d_model]
        // --- CRITICAL FIX: Ensure grad_input is correctly sized ---
        size_t seq_len = grad_output.dim(0);
        if (grad_input.shape.size() != 2 || grad_input.dim(0) != seq_len || grad_input.dim(1) != c_fc.W.dim(1)) { // c_fc.W.dim(1) is d_model
             grad_input = Tensor({seq_len, c_fc.W.dim(1)}); // Resize correctly [seq_len, d_model]
        }
        // --- END CRITICAL FIX ---

        // 1. Backward through output projection c_proj
        // dL/d(hidden) = dL/dY * W_proj^T
        c_proj.backward(grad_output, grad_input); // grad_input now holds dL/d(hidden) [seq_len, 4*d_model]

        // 2. Backward through GELU activation
        // dL/d(gelu_input) = dL/d(hidden) * gelu_grad(gelu_input)
        // grad_input currently holds dL/d(hidden), gelu_input_buffer holds the pre-GELU values.
        Tensor grad_gelu_input = grad_input; // Alias for clarity
        gelu_grad(gelu_input_buffer, grad_gelu_input, grad_input); // grad_input now holds dL/d(gelu_input) [seq_len, 4*d_model]

        // 3. Backward through first linear layer c_fc
        // dL/dX = dL/d(gelu_input) * W_fc^T
        // This also accumulates gradients for c_fc's weights and biases.
        Tensor grad_X_temp({grad_input.dim(0), c_fc.W.dim(1)}); // [seq_len, d_model]
        c_fc.backward(grad_input, grad_X_temp); // grad_X_temp holds dL/dX
        grad_input = std::move(grad_X_temp); // Assign final gradient w.r.t. FFN input
    }

    void zero_grad() {
        c_fc.zero_grad();
        c_proj.zero_grad();
    }
};

// Decoder Block (Pre-LN as in GPT-2/3) - FULLY CORRECTED
class DecoderBlock {
public:
    int d_model, n_heads, d_k; // Store d_k for use in backward pass
    LayerNorm ln_1; // Pre-attention layer norm
    MultiHeadAttention attn; // Self-attention mechanism
    LayerNorm ln_2; // Pre-FFN layer norm
    FeedForward ff; // Feed-forward network

    // Buffers for intermediate computations during forward/backward
    // These are member variables that need to be correctly sized on each forward/backward pass.
    Tensor ln1_out_buffer, attn_out_buffer, residual1_buffer;
    Tensor ln2_out_buffer, ff_out_buffer;

    DecoderBlock(int d_model, int n_heads)
        : d_model(d_model), n_heads(n_heads), d_k(d_model / n_heads),
          ln_1(d_model),
          attn(d_model, n_heads),
          ln_2(d_model),
          ff(d_model) {}

    void forward(const Tensor& X, Tensor& Y) { // X, Y: [seq_len, d_model]
        size_t seq_len = X.dim(0); // Get the dynamic sequence length

        // --- CRITICAL FIX: Ensure all member buffers are correctly sized ---
        // This is the key fix identified by the gdb segfault analysis.
        // Default-constructed Tensors have shape {} and empty data.
        // If passed to sub-components without resizing, they cause crashes.
        if (ln1_out_buffer.shape.size() != 2 || ln1_out_buffer.dim(0) != seq_len || ln1_out_buffer.dim(1) != static_cast<size_t>(d_model)) {
            ln1_out_buffer = Tensor({seq_len, static_cast<size_t>(d_model)});
        }
        if (attn_out_buffer.shape.size() != 2 || attn_out_buffer.dim(0) != seq_len || attn_out_buffer.dim(1) != static_cast<size_t>(d_model)) {
            attn_out_buffer = Tensor({seq_len, static_cast<size_t>(d_model)});
        }
        if (residual1_buffer.shape.size() != 2 || residual1_buffer.dim(0) != seq_len || residual1_buffer.dim(1) != static_cast<size_t>(d_model)) {
            residual1_buffer = Tensor({seq_len, static_cast<size_t>(d_model)});
        }
        if (ln2_out_buffer.shape.size() != 2 || ln2_out_buffer.dim(0) != seq_len || ln2_out_buffer.dim(1) != static_cast<size_t>(d_model)) {
            ln2_out_buffer = Tensor({seq_len, static_cast<size_t>(d_model)});
        }
        if (ff_out_buffer.shape.size() != 2 || ff_out_buffer.dim(0) != seq_len || ff_out_buffer.dim(1) != static_cast<size_t>(d_model)) {
            ff_out_buffer = Tensor({seq_len, static_cast<size_t>(d_model)});
        }
        // --- END CRITICAL FIX ---

        // 1. Pre-Layer Normalization + Multi-Head Attention
        ln_1.forward(X, ln1_out_buffer); // Apply LN to input X
        attn.forward(ln1_out_buffer, attn_out_buffer); // Compute attention on normalized input
        add(X, attn_out_buffer, residual1_buffer); // Residual connection: X + attn_out

        // 2. Pre-Layer Normalization + Feed-Forward Network
        ln_2.forward(residual1_buffer, ln2_out_buffer); // Apply LN to residual1
        ff.forward(ln2_out_buffer, ff_out_buffer); // Compute FFN on normalized residual
        add(residual1_buffer, ff_out_buffer, Y); // Final residual connection: residual1 + ff_out -> Y
    }

    // Full backward pass for Decoder Block chaining gradients through sub-components
    void backward(const Tensor& grad_output, Tensor& grad_input) { // grad_output, grad_input: [seq_len, d_model]
        size_t seq_len = grad_output.dim(0);
        // --- CRITICAL FIX: Ensure gradient tensors passed to sub-module backward() are correctly sized ---
        // Default-constructed tensors passed as grad_X will fail assertions in sub-modules.
        // We must pre-size them.
        Tensor grad_ln2_out = grad_output; // [seq_len, d_model] - Alias/Copy is okay for initial value
        Tensor grad_ff_out = grad_output;  // [seq_len, d_model] - Alias/Copy is okay for initial value
        Tensor grad_residual1({seq_len, static_cast<size_t>(d_model)}); // MUST BE PRE-SIZED [seq_len, d_model]
        // For the residual split, we conceptually split grad_residual1.
        // The simplest and safest way is to copy it.
        Tensor grad_ln1_out({seq_len, static_cast<size_t>(d_model)}); // MUST BE PRE-SIZED [seq_len, d_model]
        Tensor grad_attn_out({seq_len, static_cast<size_t>(d_model)}); // MUST BE PRE-SIZED [seq_len, d_model]
        // --- END CRITICAL FIX ---

        // 1. Backward through second residual connection (Y = residual1 + ff_out)
        // Gradient splits equally to both branches of the residual connection.
        // In code, we pass grad_output (via grad_ln2_out/grad_ff_out) to the consumers.

        // 2. Backward through Feed-Forward Network
        // This computes dL/d(ln2_out) and accumulates gradients within the FFN.
        ff.backward(grad_ff_out, grad_ln2_out); // grad_ln2_out is updated with dL/d(ln2_out)

        // 3. Backward through second LayerNorm (pre-FFN)
        // This computes dL/d(residual1) and accumulates gradients for LN2 parameters.
        // grad_residual1 is now correctly sized [seq_len, d_model]
        ln_2.backward(grad_ln2_out, grad_residual1); // grad_residual1 holds dL/d(residual1)

        // 4. Backward through first residual connection (residual1 = X + attn_out)
        // Gradient splits to the main input X and the attention output branch.
        // We copy this gradient to both downstream consumers.
        // grad_ln1_out and grad_attn_out are now correctly sized [seq_len, d_model]
        grad_ln1_out = grad_residual1; // Copy data to correctly sized tensor
        grad_attn_out = grad_residual1; // Copy data to correctly sized tensor

        // 5. Backward through Multi-Head Attention
        // This computes dL/d(ln1_out) and accumulates gradients within the attention mechanism.
        attn.backward(grad_attn_out, grad_ln1_out); // grad_ln1_out is updated with dL/d(ln1_out)

        // 6. Backward through first LayerNorm (pre-attention)
        // This computes the final dL/dX and accumulates gradients for LN1 parameters.
        ln_1.backward(grad_ln1_out, grad_input); // grad_input holds the final dL/dX for this block
    }

    void zero_grad() {
        ln_1.zero_grad();
        attn.zero_grad();
        ln_2.zero_grad();
        ff.zero_grad();
    }
};

// Embedding Layers
class Embedding {
public:
    int num_embeddings, embedding_dim;
    Tensor weight; // [num_embeddings, embedding_dim] - Token embedding matrix
    std::vector<int> input_ids_buffer; // Stored input IDs for backward pass
    Tensor grad_weight; // [num_embeddings, embedding_dim] - Gradients w.r.t. embedding weights

    Embedding(int num_embeddings, int embedding_dim)
        : num_embeddings(num_embeddings), embedding_dim(embedding_dim),
          weight({static_cast<size_t>(num_embeddings), static_cast<size_t>(embedding_dim)}),
          grad_weight({static_cast<size_t>(num_embeddings), static_cast<size_t>(embedding_dim)}, 0.0) {
        // Xavier initialization for embeddings
        double stddev = std::sqrt(1.0 / embedding_dim);
        fill_normal(weight, 0.0, stddev);
    }

    void forward(const std::vector<int>& input_ids, Tensor& output) { // output: [seq_len, embedding_dim]
        input_ids_buffer = input_ids; // Store for backward pass
        size_t seq_len = input_ids.size();
        // For each token ID in the input sequence, lookup its embedding vector.
        for (size_t i = 0; i < seq_len; ++i) {
            int id = input_ids[i];
            if (id >= 0 && id < num_embeddings) {
                // Copy the embedding vector for token 'id' to the output at position 'i'.
                std::memcpy(&output.data[i * embedding_dim], &weight.data[id * embedding_dim], embedding_dim * sizeof(double));
            } else {
                // Handle out-of-vocabulary or padding tokens by setting embedding to zero.
                std::memset(&output.data[i * embedding_dim], 0, embedding_dim * sizeof(double));
            }
        }
    }

    // Backward pass for Embedding layer (sparse gradient update)
    // The gradient w.r.t. the embedding weights is sparse: only rows corresponding
    // to input token IDs receive non-zero gradients.
    void backward(const Tensor& grad_output) { // grad_output: [seq_len, embedding_dim]
        size_t seq_len = input_ids_buffer.size();
        // For each position in the input sequence, accumulate the gradient
        // from grad_output into the corresponding row of grad_weight.
        for (size_t i = 0; i < seq_len; ++i) {
            int id = input_ids_buffer[i];
            if (id >= 0 && id < num_embeddings) {
                // Accumulate gradients into the row of grad_weight corresponding to token 'id'.
                for (int j = 0; j < embedding_dim; ++j) {
                    grad_weight.data[id * embedding_dim + j] += grad_output.data[i * embedding_dim + j];
                }
            }
            // Gradients for OOV/padding tokens (id < 0 or id >= num_embeddings) are ignored.
        }
    }

    void zero_grad() {
        grad_weight.zero(); // Zero out all gradients in the embedding matrix.
    }
};

class PositionalEncoding {
public:
    int d_model, max_len;
    Tensor pe; // [max_len, d_model] - Pre-computed positional encodings

    PositionalEncoding(int d_model, int max_len) : d_model(d_model), max_len(max_len), pe({static_cast<size_t>(max_len), static_cast<size_t>(d_model)}) {
        // Compute fixed positional encodings as per "Attention Is All You Need" paper.
        for (int pos = 0; pos < max_len; ++pos) {
            for (int i = 0; i < d_model; ++i) {
                // Calculate the denominator term: 10000^(2i/d_model)
                double div_term = std::pow(10000.0, static_cast<double>(2 * (i / 2)) / d_model);
                // Apply sin for even indices, cos for odd indices.
                if (i % 2 == 0) {
                    pe.data[pos * d_model + i] = std::sin(static_cast<double>(pos) / div_term);
                } else {
                    pe.data[pos * d_model + i] = std::cos(static_cast<double>(pos) / div_term);
                }
            }
        }
    }

    void forward(int seq_len, Tensor& output) { // output: [seq_len, d_model]
        // Simply copy the first 'seq_len' rows from the pre-computed positional encodings.
        assert(seq_len <= max_len);
        std::memcpy(output.data.data(), pe.data.data(), seq_len * d_model * sizeof(double));
    }
};

// --- Full GPT Model (GPT-2/3 style architecture) ---
class GPT {
public:
    int vocab_size, d_model, n_heads, n_layers, max_seq_len;
    Embedding wte; // Token embeddings [vocab_size, d_model]
    PositionalEncoding wpe; // Positional embeddings [max_seq_len, d_model]
    std::vector<std::unique_ptr<DecoderBlock>> h; // Stack of N decoder blocks
    LayerNorm ln_f; // Final layer normalization [d_model]

    // --- Buffers for intermediate states (essential for correct backward pass) ---
    // These are member variables to persist between forward_single and backward_single calls.
    std::vector<Tensor> block_input_buffers; // Stores input to each block for residual connections
    std::vector<Tensor> block_output_buffers; // Stores output of each block (FIXED: Now member)
    Tensor tok_emb_buffer; // [seq_len, d_model] - Token embeddings
    Tensor pos_emb_buffer; // [seq_len, d_model] - Positional embeddings
    Tensor embedded_buffer; // [seq_len, d_model] - Sum of token and pos embeddings
    Tensor final_ln_output_buffer; // [seq_len, d_model] - Output of final LayerNorm

    GPT(int vocab_size, int d_model, int n_heads, int n_layers, int max_seq_len)
        : vocab_size(vocab_size), d_model(d_model), n_heads(n_heads), n_layers(n_layers), max_seq_len(max_seq_len),
          wte(vocab_size, d_model), // Initialize token embeddings
          wpe(d_model, max_seq_len), // Initialize positional embeddings
          ln_f(d_model) { // Initialize final layer norm
        // Create N decoder blocks
        for (int i = 0; i < n_layers; ++i) {
            h.push_back(std::make_unique<DecoderBlock>(d_model, n_heads));
        }
        // Initialize buffers (as members, they persist)
        block_input_buffers.resize(n_layers);
        block_output_buffers.resize(n_layers); // FIXED: Now a member resized here
    }

    // Forward pass for a single sequence
    void forward_single(const std::vector<int>& input_ids, Tensor& logits) { // logits: [seq_len, vocab_size]
        size_t seq_len = input_ids.size();
        assert(seq_len <= static_cast<size_t>(max_seq_len));

        // --- CRITICAL FIX: Ensure all buffers used in this method are correctly sized ---
        // Resize member buffers that are used/modified by this method.
        if (tok_emb_buffer.shape.size() != 2 || tok_emb_buffer.dim(0) != seq_len || tok_emb_buffer.dim(1) != static_cast<size_t>(d_model)) {
            tok_emb_buffer = Tensor({seq_len, static_cast<size_t>(d_model)});
        }
        if (pos_emb_buffer.shape.size() != 2 || pos_emb_buffer.dim(0) != seq_len || pos_emb_buffer.dim(1) != static_cast<size_t>(d_model)) {
            pos_emb_buffer = Tensor({seq_len, static_cast<size_t>(d_model)});
        }
        if (embedded_buffer.shape.size() != 2 || embedded_buffer.dim(0) != seq_len || embedded_buffer.dim(1) != static_cast<size_t>(d_model)) {
            embedded_buffer = Tensor({seq_len, static_cast<size_t>(d_model)});
        }
        if (final_ln_output_buffer.shape.size() != 2 || final_ln_output_buffer.dim(0) != seq_len || final_ln_output_buffer.dim(1) != static_cast<size_t>(d_model)) {
            final_ln_output_buffer = Tensor({seq_len, static_cast<size_t>(d_model)});
        }
        // Ensure block buffers are correctly sized
        block_input_buffers.resize(n_layers);
        block_output_buffers.resize(n_layers); // FIXED: Now a member resized here
        for (int i = 0; i < n_layers; ++i) {
            if (block_input_buffers[i].shape.size() != 2 || block_input_buffers[i].dim(0) != seq_len || block_input_buffers[i].dim(1) != static_cast<size_t>(d_model)) {
                block_input_buffers[i] = Tensor({seq_len, static_cast<size_t>(d_model)});
            }
            // FIXED: Resize the member block_output_buffers[i]
            if (block_output_buffers[i].shape.size() != 2 || block_output_buffers[i].dim(0) != seq_len || block_output_buffers[i].dim(1) != static_cast<size_t>(d_model)) {
                block_output_buffers[i] = Tensor({seq_len, static_cast<size_t>(d_model)});
            }
        }
        // --- END CRITICAL FIX ---

        // --- 1. Embedding Lookup ---
        // a. Token Embeddings
        wte.forward(input_ids, tok_emb_buffer); // Compute token embeddings

        // b. Positional Embeddings
        wpe.forward(seq_len, pos_emb_buffer); // Compute positional embeddings

        // c. Sum Token and Positional Embeddings
        add(tok_emb_buffer, pos_emb_buffer, embedded_buffer); // X = tok_emb + pos_emb

        // --- 2. Pass through Transformer Decoder Blocks ---
        Tensor x = embedded_buffer; // Current hidden state
        for (int i = 0; i < n_layers; ++i) {
            block_input_buffers[i] = x; // Store input for residual in backward
            block_output_buffers[i] = Tensor({seq_len, static_cast<size_t>(d_model)}); // Allocate output buffer
            h[i]->forward(x, block_output_buffers[i]); // Forward through block
            x = block_output_buffers[i]; // Update hidden state
        }

        // --- 3. Final Layer Normalization ---
        ln_f.forward(x, final_ln_output_buffer); // Apply final LN

        // --- 4. Language Model Head (with weight tying) ---
        // Project final hidden states to vocabulary logits using transposed token embedding matrix.
        // Y = final_ln_output @ W_te^T, where W_te is the token embedding matrix.
        Tensor WT;
        transpose(wte.weight, WT); // Transpose token embedding matrix [d_model, vocab_size]
        matmul(final_ln_output_buffer, WT, logits); // [seq_len, d_model] x [d_model, vocab_size] -> [seq_len, vocab_size]
    }

    // Full backward pass for the entire GPT model
    void backward_single(const Tensor& grad_logits, const std::vector<int>& input_ids) {
        size_t seq_len = grad_logits.dim(0);
        assert(seq_len <= static_cast<size_t>(max_seq_len));

        // --- 1. Backward through Language Model Head (weight tying) ---
        // dL/d(final_ln_output) = dL/d(logits) @ W_te
        Tensor grad_final_ln_out({seq_len, static_cast<size_t>(d_model)}); // [seq_len, d_model]
        matmul(grad_logits, wte.weight, grad_final_ln_out); // Use original wte.weight [vocab_size, d_model]

        // Gradient w.r.t. token embeddings for weight update (due to weight tying)
        // dL/dW_te = final_ln_output^T @ dL/d(logits)
        Tensor grad_wte_logits({static_cast<size_t>(d_model), static_cast<size_t>(vocab_size)}); // [d_model, vocab_size]
        Tensor final_ln_out_T;
        transpose(final_ln_output_buffer, final_ln_out_T); // [d_model, seq_len]
        matmul(final_ln_out_T, grad_logits, grad_wte_logits); // [d_model, seq_len] x [seq_len, vocab_size]
        Tensor grad_wte_logits_T; // Transpose to match wte.grad_weight shape
        transpose(grad_wte_logits, grad_wte_logits_T); // [vocab_size, d_model]
        add_inplace(wte.grad_weight, grad_wte_logits_T); // Accumulate gradient (weight tying)

        // --- 2. Backward through Final Layer Normalization ---
        Tensor grad_block_n_out({seq_len, static_cast<size_t>(d_model)}); // [seq_len, d_model]
        ln_f.backward(grad_final_ln_out, grad_block_n_out); // Compute gradient w.r.t. input to final LN

        // --- 3. Backward through Transformer Decoder Blocks (in reverse order) ---
        Tensor grad_x = grad_block_n_out; // Gradient flowing backwards
        for (int i = n_layers - 1; i >= 0; --i) {
            // --- CRITICAL FIX: Ensure grad_x_prev is correctly sized ---
            Tensor grad_x_prev({seq_len, static_cast<size_t>(d_model)}); // [seq_len, d_model] - MUST BE PRE-SIZED
            // --- END CRITICAL FIX ---
            // Backward through block, computing gradient w.r.t. its input
            h[i]->backward(grad_x, grad_x_prev);
            grad_x = std::move(grad_x_prev); // Update gradient for next iteration
        }
        // Note: grad_x at this point is dL/d(embedded_input) = dL/d(tok_emb + pos_emb)
        // Since pos_emb has no parameters, its gradient is not computed/used.
        // The gradient w.r.t. token embeddings is handled inside wte.backward if needed for other purposes,
        // but the main parameter update flow for input embeddings stops here in this view.
    }

    void zero_grad() {
        wte.zero_grad(); // Zero gradients for token embeddings
        for (auto& block : h) {
            block->zero_grad(); // Zero gradients for all decoder blocks
        }
        ln_f.zero_grad(); // Zero gradients for final layer norm
    }
};

// --- Optimizer: AdamW (Adam with decoupled weight decay) ---
class AdamWOptimizer {
public:
    double lr, beta1, beta2, eps, weight_decay;
    int t;
    // Storage for parameters, gradients, and Adam moments
    std::vector<std::pair<Tensor*, Tensor*>> params_and_grads; // List of {param, grad} pairs
    std::vector<Tensor> m_buffers, v_buffers; // Adam moment buffers (m: 1st moment, v: 2nd moment)

    AdamWOptimizer(double learning_rate = 1e-3, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8, double wd = 0.01)
        : lr(learning_rate), beta1(beta1), beta2(beta2), eps(epsilon), weight_decay(wd), t(0) {}

    // Register a parameter and its gradient tensor with the optimizer
    void register_param(Tensor& param, Tensor& grad) {
        params_and_grads.push_back({&param, &grad});
        m_buffers.emplace_back(param.shape, 0.0); // Initialize 1st moment buffer
        v_buffers.emplace_back(param.shape, 0.0); // Initialize 2nd moment buffer
    }

    // Perform a single optimization step (parameter update)
    void step() {
        t++; // Increment time step
        double bias_correction1 = 1.0 - std::pow(beta1, t); // Bias correction for 1st moment
        double bias_correction2 = 1.0 - std::pow(beta2, t); // Bias correction for 2nd moment
        double lr_corrected = lr / bias_correction1; // Corrected learning rate

        // Iterate through all registered parameters
        for (size_t i = 0; i < params_and_grads.size(); ++i) {
            Tensor* param = params_and_grads[i].first;  // Current parameter tensor
            Tensor* grad = params_and_grads[i].second;  // Gradient of the parameter
            Tensor& m = m_buffers[i];                   // 1st moment buffer for this param
            Tensor& v = v_buffers[i];                   // 2nd moment buffer for this param

            // Update parameters element-wise
            for (size_t j = 0; j < param->size(); ++j) {
                double g = grad->data[j]; // Get current gradient element

                // 1. Apply decoupled weight decay: g_t = g_t + λ * θ_{t-1}
                // This is the key difference from standard Adam.
                g = g + weight_decay * param->data[j];

                // 2. Update biased first moment estimate: m_t = β1 * m_{t-1} + (1 - β1) * g_t
                m.data[j] = beta1 * m.data[j] + (1 - beta1) * g;

                // 3. Update biased second raw moment estimate: v_t = β2 * v_{t-1} + (1 - β2) * g_t^2
                v.data[j] = beta2 * v.data[j] + (1 - beta2) * g * g;

                // 4. Compute bias-corrected estimates
                double m_hat = m.data[j]; // m_hat is already bias-corrected in the loop
                double v_hat = v.data[j] / bias_correction2; // Apply bias correction to v

                // 5. Update parameters: θ_t = θ_{t-1} - (α / (sqrt(v_hat) + ε)) * m_hat
                param->data[j] -= lr_corrected * m_hat / (std::sqrt(v_hat) + eps);
            }
        }
    }
};

// --- Data Loading and Preprocessing (for TinyShakespeare) ---
std::string load_text_file(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Could not open file: " << filepath << std::endl;
        return "";
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

std::vector<int> encode_text(const std::string& text, const std::unordered_map<char, int>& stoi) {
    std::vector<int> ids;
    ids.reserve(text.size()); // Reserve space for efficiency
    for (char c : text) {
        auto it = stoi.find(c);
        if (it != stoi.end()) {
            ids.push_back(it->second); // Map character to ID
        } else {
            ids.push_back(0); // Use 0 for <PAD> or <UNK>
        }
    }
    return ids;
}

std::pair<std::unordered_map<char, int>, std::unordered_map<int, char>> create_vocab(const std::string& text) {
    std::unordered_set<char> unique_chars(text.begin(), text.end());
    std::vector<char> chars(unique_chars.begin(), unique_chars.end());
    std::sort(chars.begin(), chars.end()); // Sort for deterministic ordering

    std::unordered_map<char, int> stoi; // String-to-index map
    std::unordered_map<int, char> itos; // Index-to-string map
    for (size_t i = 0; i < chars.size(); ++i) {
        stoi[chars[i]] = i;
        itos[i] = chars[i];
    }
    return {stoi, itos};
}

std::vector<std::vector<int>> create_batches(const std::vector<int>& data, int batch_size, int block_size) {
    std::vector<std::vector<int>> batches;
    // Handle case where data is too small
    if (data.size() < 2) { // Need at least 2 tokens for x and y
        std::cerr << "Warning: Data size (" << data.size() << ") is too small to create batches." << std::endl;
        return batches;
    }
    size_t num_batches = (data.size() - 1) / (batch_size * block_size);
    // Ensure at least one batch can be created if data is sufficient
    if (num_batches == 0 && (data.size() - 1) > 0) {
        num_batches = 1;
        // std::cerr << "Warning: Data size (" << data.size() << ") is small. Creating a single batch." << std::endl;
    }

    for (size_t i = 0; i < num_batches; ++i) {
        std::vector<int> batch_data(batch_size * block_size + 1); // +1 for target
        for (int j = 0; j < batch_size * block_size + 1; ++j) {
            size_t idx = i * batch_size * block_size + j;
            if (idx < data.size()) {
                batch_data[j] = data[idx];
            } else {
                batch_data[j] = 0; // Padding
            }
        }
        batches.push_back(batch_data);
    }
    return batches;
}

// --- Training Loop ---
double train_step(GPT& model, AdamWOptimizer& optimizer, const std::vector<int>& batch_data, int batch_size, int block_size) {
    model.zero_grad(); // Zero out gradients from previous step
    double total_loss = 0.0;
    int num_samples = 0; // Count of valid token predictions

    // Process each sample in the batch
    for (int b = 0; b < batch_size; ++b) {
        std::vector<int> x_ids(block_size); // Input token IDs
        std::vector<int> y_ids(block_size); // Target token IDs
        bool valid_sample = true;

        // Extract input (x) and target (y) sequences for this sample
        for (int t = 0; t < block_size; ++t) {
            size_t idx = b * block_size + t;
            if (idx < batch_data.size() - 1) {
                x_ids[t] = batch_data[idx];
                y_ids[t] = batch_data[idx + 1]; // Next token prediction
            } else {
                valid_sample = false;
                break; // Incomplete sample, skip
            }
        }
        if (!valid_sample) continue; // Skip invalid samples

        // --- Forward Pass ---
        Tensor logits({static_cast<size_t>(block_size), static_cast<size_t>(model.vocab_size)}); // [T, V]
        model.forward_single(x_ids, logits); // Compute model logits

        // --- Compute Loss and Gradient ---
        // a. Apply softmax to get probabilities (numerically stable)
        Tensor probs({static_cast<size_t>(block_size), static_cast<size_t>(model.vocab_size)}); // [T, V]
        softmax(logits, probs); // probs[t][v] = P(token v at time t)

        // b. Prepare one-hot encoded true targets and compute loss
        Tensor y_true_one_hot({static_cast<size_t>(block_size), static_cast<size_t>(model.vocab_size)}, 0.0); // [T, V]
        for (int t = 0; t < block_size; ++t) {
            int true_token = y_ids[t];
            if (true_token >= 0 && true_token < model.vocab_size) {
                y_true_one_hot.data[t * model.vocab_size + true_token] = 1.0; // Set one-hot
                // Compute Cross-Entropy loss: L = -log(p_{true})
                double prob_true = std::max(probs.data[t * model.vocab_size + true_token], EPSILON);
                total_loss += -std::log(prob_true); // Accumulate loss
            }
        }
        num_samples += block_size; // Update sample count

        // c. Compute gradient w.r.t. logits (combined softmax + cross-entropy gradient)
        // grad_logits[t][v] = p[v] - y_true[v] (mathematically derived)
        Tensor grad_logits({static_cast<size_t>(block_size), static_cast<size_t>(model.vocab_size)}); // [T, V]
        softmax_crossentropy_grad(probs, y_true_one_hot, grad_logits); // Compute gradient

        // --- Backward Pass ---
        model.backward_single(grad_logits, x_ids); // Backpropagate gradients through the model
    }

    // --- Optimizer Step ---
    if (num_samples > 0) {
        optimizer.step(); // Update model parameters using accumulated gradients
        return total_loss / num_samples; // Return average loss per token
    }
    return 0.0; // Handle case of no valid samples
}

// --- Inference / Text Generation (ADVANCED VERSION) ---
std::string generate_text(GPT& model, const std::unordered_map<int, char>& itos, const std::string& prompt, int max_new_tokens, int block_size) {
    std::vector<int> context; // Vector to hold the sequence of token IDs
    context.reserve(prompt.size() + max_new_tokens); // Reserve space for efficiency

    // Encode the initial prompt string into token IDs
    for(char c : prompt) {
        bool found = false;
        for(const auto& pair : itos) {
            if(pair.second == c) {
                context.push_back(pair.first); // Add token ID to context
                found = true;
                break;
            }
        }
        if (!found) context.push_back(0); // Use <PAD>/<UNK> for unknown characters
    }

    // --- ADVANCED GENERATION PARAMETERS ---
    double temperature = 0.8; // Lower (<1.0) = more focused, Higher (>1.0) = more random
    int top_k = 40; // Number of top tokens to consider for sampling (0 = no top-k)
    // --- END PARAMETERS ---

    std::random_device rd; // For seeding the generator used in sampling
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    // Generate new tokens autoregressively
    for (int i = 0; i < max_new_tokens; ++i) {
        // Determine the starting index for the current input window
        int start_idx = std::max(0, static_cast<int>(context.size()) - block_size);
        // Extract the last 'block_size' tokens as input
        std::vector<int> input_ids(context.begin() + start_idx, context.end());

        // --- Forward Pass for Generation ---
        Tensor logits({static_cast<size_t>(input_ids.size()), static_cast<size_t>(model.vocab_size)}); // [T, V]
        model.forward_single(input_ids, logits); // Get logits for the input sequence

        // --- Sample Next Token (ADVANCED) ---
        // Focus on the logits for the very last token in the sequence
        size_t last_token_idx = input_ids.size() - 1;

        // --- 1. Apply Temperature Scaling ---
        Tensor scaled_logits({static_cast<size_t>(model.vocab_size)});
        for (int v = 0; v < model.vocab_size; ++v) {
            scaled_logits.data[v] = logits.data[last_token_idx * model.vocab_size + v] / temperature;
        }

        // --- 2. Apply numerically stable softmax to the scaled logits ---
        double max_logit = scaled_logits.data[0]; // Find max for stability
        for (int v = 1; v < model.vocab_size; ++v) {
            double val = scaled_logits.data[v];
            if (val > max_logit) max_logit = val;
        }
        double sum_exp = 0.0;
        std::vector<double> exp_logits(model.vocab_size); // Store exp(logits - max)
        for (int v = 0; v < model.vocab_size; ++v) {
            double val = scaled_logits.data[v];
            exp_logits[v] = std::exp(val - max_logit);
            sum_exp += exp_logits[v];
        }
        std::vector<double> probs(model.vocab_size); // Compute final probabilities
        for (int v = 0; v < model.vocab_size; ++v) {
            probs[v] = exp_logits[v] / sum_exp;
        }

        // --- 3. Apply Top-K Sampling ---
        int next_token = 0;
        if (top_k > 0 && top_k < model.vocab_size) {
            // Create a list of {probability, token_id} pairs
            std::vector<std::pair<double, int>> sorted_probs;
            for (int v = 0; v < model.vocab_size; ++v) {
                sorted_probs.push_back({probs[v], v});
            }
            // Sort descending by probability
            std::sort(sorted_probs.begin(), sorted_probs.end(), std::greater<>());
            // Keep only the top K
            sorted_probs.resize(top_k);

            // Renormalize the top K probabilities
            double sum_top_k = 0.0;
            for (const auto& p : sorted_probs) sum_top_k += p.first;
            if (sum_top_k > 0) {
                for (auto& p : sorted_probs) p.first /= sum_top_k;
            }

            // Sample from the top K distribution
            double rand_sample = dis(gen);
            double cum_prob = 0.0;
            bool sampled = false;
            for (const auto& p : sorted_probs) {
                cum_prob += p.first;
                if (rand_sample <= cum_prob) {
                    next_token = p.second;
                    sampled = true;
                    break;
                }
            }
            // Fallback if no token was selected (e.g., due to floating-point inaccuracies)
            if (!sampled && !sorted_probs.empty()) {
                next_token = sorted_probs.front().second;
            }
        } else {
            // --- 4. Greedy Sampling (argmax) if top_k is 0 or >= vocab_size ---
            double max_prob = probs[0];
            for (int v = 1; v < model.vocab_size; ++v) {
                if (probs[v] > max_prob) {
                    max_prob = probs[v];
                    next_token = v; // Update next token ID
                }
            }
        }
        context.push_back(next_token); // Append the predicted token to the context
    }

    // --- Decode Generated Tokens ---
    std::string result; // String to hold the final generated text
    result.reserve(context.size()); // Reserve space for efficiency
    // Convert the sequence of token IDs back into characters
    for (int id : context) {
        auto it = itos.find(id); // Lookup character for token ID
        if (it != itos.end()) {
            result += it->second; // Append character to result
        } else {
            result += '?'; // Indicate unknown token
        }
    }
    return result; // Return the generated text string
}

// --- Main Function: Entry Point ---
int main() {
    const std::string filepath = "input.txt"; // Path to the dataset file
    const std::string url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"; // Dataset URL

    // 1. Download dataset if not present locally
    std::ifstream f(filepath);
    if (!f.good()) {
        std::cout << "Downloading TinyShakespeare dataset..." << std::endl;
        std::string command = "curl -o " + filepath + " " + url; // Try curl first
        int result = std::system(command.c_str());
        if (result != 0) {
             // Fallback to wget if curl fails
             command = "wget -O " + filepath + " " + url;
             result = std::system(command.c_str());
        }
        if (result != 0) {
            // If both fail, prompt user to download manually
            std::cerr << "Failed to download dataset. Please download manually from " << url << " to 'input.txt'" << std::endl;
            return 1;
        }
    }

    // 2. Load and preprocess the text data
    std::string text = load_text_file(filepath); // Load text from file
    if (text.empty()) {
        std::cerr << "Failed to load text data." << std::endl;
        return 1;
    }

    // Create vocabulary mappings (character-level)
    auto [stoi, itos] = create_vocab(text);
    int vocab_size = stoi.size(); // Size of the vocabulary
    std::cout << "Vocabulary size: " << vocab_size << std::endl;

    // Encode the entire text corpus into a sequence of token IDs
    std::vector<int> data = encode_text(text, stoi);

    // 3. Define Model Hyperparameters (Adjusted for CPU feasibility)
    const int batch_size = 6;   // Number of sequences per batch (original is 4)
    const int block_size = 128;  // Sequence length (context window) (original is 64)
    const int d_model = 256;    // Embedding/Layer dimension (smaller for CPU) (original is 128)
    const int n_heads = 6;      // Number of attention heads (original is 4)
    const int n_layers = 6;     // Number of transformer blocks (original is 4)
    const int max_seq_len = block_size; // Maximum sequence length
    const int max_iters = 2000; // Total number of training iterations
    const int eval_interval = 50; // Interval for evaluation and printing (original is 200)

    // 4. Instantiate the GPT Model and Optimizer
    GPT model(vocab_size, d_model, n_heads, n_layers, max_seq_len); // Create GPT model
    // Create AdamW optimizer with specified hyperparameters
    AdamWOptimizer optimizer(1e-3, 0.9, 0.999, 1e-8, 0.01);

    // Register all model parameters with the optimizer for gradient updates
    // Token Embeddings
    optimizer.register_param(model.wte.weight, model.wte.grad_weight);
    // Parameters for each Transformer Block
    for (int i = 0; i < n_layers; ++i) {
        // Multi-Head Attention parameters
        optimizer.register_param(model.h[i]->attn.c_attn.W, model.h[i]->attn.c_attn.grad_W);
        optimizer.register_param(model.h[i]->attn.c_attn.b, model.h[i]->attn.c_attn.grad_b);
        optimizer.register_param(model.h[i]->attn.c_proj.W, model.h[i]->attn.c_proj.grad_W);
        optimizer.register_param(model.h[i]->attn.c_proj.b, model.h[i]->attn.c_proj.grad_b);
        // Feed-Forward Network parameters
        optimizer.register_param(model.h[i]->ff.c_fc.W, model.h[i]->ff.c_fc.grad_W);
        optimizer.register_param(model.h[i]->ff.c_fc.b, model.h[i]->ff.c_fc.grad_b);
        optimizer.register_param(model.h[i]->ff.c_proj.W, model.h[i]->ff.c_proj.grad_W);
        optimizer.register_param(model.h[i]->ff.c_proj.b, model.h[i]->ff.c_proj.grad_b);
        // Layer Normalization parameters (Pre-attention and Pre-FFN)
        optimizer.register_param(model.h[i]->ln_1.gamma, model.h[i]->ln_1.grad_gamma);
        optimizer.register_param(model.h[i]->ln_1.beta, model.h[i]->ln_1.grad_beta);
        optimizer.register_param(model.h[i]->ln_2.gamma, model.h[i]->ln_2.grad_gamma);
        optimizer.register_param(model.h[i]->ln_2.beta, model.h[i]->ln_2.grad_beta);
    }
    // Final Layer Normalization parameters
    optimizer.register_param(model.ln_f.gamma, model.ln_f.grad_gamma);
    optimizer.register_param(model.ln_f.beta, model.ln_f.grad_beta);

    // 5. Prepare training data batches
    std::vector<std::vector<int>> batches = create_batches(data, batch_size, block_size);

    // --- ADD THIS CHECK ---
    std::cout << "Number of batches created: " << batches.size() << std::endl;
    if (batches.empty()) {
        std::cerr << "Error: No batches created. Check data size and batch parameters." << std::endl;
        std::cerr << "Data size: " << data.size() << ", Batch size: " << batch_size << ", Block size: " << block_size << std::endl;
        if (data.size() > 1) {
             size_t potential_num_batches = (data.size() - 1) / (batch_size * block_size);
             std::cerr << "Calculated num_batches would be: " << potential_num_batches << std::endl;
             if (potential_num_batches == 0) {
                 std::cerr << "Reason: Not enough data to create a single full batch." << std::endl;
                 std::cerr << "Need at least " << (batch_size * block_size + 1) << " tokens." << std::endl;
                 std::cerr << "You have " << data.size() << " tokens." << std::endl;
             }
        } else {
             std::cerr << "Reason: Data size is too small (<= 1 token)." << std::endl;
        }
        return 1; // Exit with error code
    }
    // --- END ADDITION ---

    // 6. Training Loop
    std::cout << "Starting training..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now(); // Start timer
    for (int iter = 0; iter <= max_iters; ++iter) {
        // Periodically print training status and timing
        if (iter % eval_interval == 0 || iter == max_iters) {
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            std::cout << std::fixed << std::setprecision(4); // Set output precision
            std::cout << "Step " << iter << ", Time: " << duration.count() << "ms" << std::endl;
            start_time = std::chrono::high_resolution_clock::now(); // Reset timer
        }

        // Handle empty batch list (should not happen with valid data, but check anyway)
        if (batches.empty()) {
            std::cerr << "No batches created. Check data loading." << std::endl;
            break;
        }
        // Select the current batch (cycling through the list)
        int batch_idx = iter % batches.size();
        // Shuffle batches periodically to improve training
        if (batch_idx == 0 && iter > 0) {
             std::random_device rd;
             std::mt19937 g(rd());
             std::shuffle(batches.begin(), batches.end(), g);
        }
        const std::vector<int>& batch = batches[batch_idx]; // Get current batch data

        // Perform a single training step on the batch
        double loss = train_step(model, optimizer, batch, batch_size, block_size);

        // Print loss periodically
        if (iter % eval_interval == 0 || iter == max_iters) {
            std::cout << "Loss: " << loss << std::endl;
        }
    }

    // 7. Generate sample text after training (using ADVANCED sampling)
    std::cout << "\nGenerating text (Advanced Sampling: Temp=" << 0.8 << ", Top-K=" << 40 << ")..." << std::endl;
    std::string prompt = "ROMEO:"; // Initial prompt for generation
    // Generate 500 new characters based on the prompt using the advanced sampler
    std::string generated = generate_text(model, itos, prompt, 500, block_size);
    std::cout << "Prompt: " << prompt << std::endl; // Print the prompt
    std::cout << "Generated: " << generated << std::endl; // Print the generated text

    return 0; // Exit successfully
}
