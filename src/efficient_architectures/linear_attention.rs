// Linear Attention Variants - O(n) instead of O(n²)
//
// Standard attention: O(n²d)
//   Attention(Q, K, V) = softmax(QK^T / √d) V
//   QK^T is n×n matrix
//
// Linear attention: O(nd²)
//   Use kernel trick: φ(Q)(φ(K)^T V)
//   Compute φ(K)^T V first (d×d matrix) → O(nd²)
//
// This file implements:
// 1. Linformer: Low-rank approximation of attention
// 2. Performer: FAVOR+ (Fast Attention Via Orthogonal Random features)
// 3. FNet: Fourier Transform replaces attention
// 4. RWKV: Receptance Weighted Key Value (RNN-like)
//
// References:
// - Wang et al. (2020): "Linformer: Self-Attention with Linear Complexity"
// - Choromanski et al. (2021): "Rethinking Attention with Performers"
// - Lee-Thorp et al. (2021): "FNet: Mixing Tokens with Fourier Transforms"
// - Peng et al. (2023): "RWKV: Reinventing RNNs for the Transformer Era"

use std::f32::consts::PI;

/// Linformer: Low-rank attention approximation
///
/// Key insight: Attention matrix A = softmax(QK^T) is often low-rank
/// Approximate: QK^T ≈ Q(E^T K)^T where E projects K to lower dimension
///
/// Complexity: O(ndk) where k << n (projection dimension)
pub struct Linformer {
    /// Model dimension
    pub d_model: usize,
    /// Number of heads
    pub num_heads: usize,
    /// Projection dimension (k << n for efficiency)
    pub k: usize,
    /// Key projection matrix E (n×k)
    pub E_key: Vec<Vec<f32>>,
    /// Value projection matrix F (n×k)
    pub E_value: Vec<Vec<f32>>,
}

impl Linformer {
    /// Create new Linformer layer
    pub fn new(d_model: usize, num_heads: usize, seq_len: usize, k: usize) -> Self {
        assert!(k <= seq_len, "Projection dimension k must be <= sequence length");

        // Initialize projection matrices
        let E_key = Self::init_projection(seq_len, k);
        let E_value = Self::init_projection(seq_len, k);

        Self {
            d_model,
            num_heads,
            k,
            E_key,
            E_value,
        }
    }

    /// Initialize projection matrix
    fn init_projection(n: usize, k: usize) -> Vec<Vec<f32>> {
        let mut proj = vec![vec![0.0; k]; n];
        let scale = (1.0 / k as f32).sqrt();

        for i in 0..n {
            for j in 0..k {
                let r = ((i * k + j) as f32 * 0.123456).fract();
                proj[i][j] = (r - 0.5) * 2.0 * scale;
            }
        }
        proj
    }

    /// Apply low-rank attention
    ///
    /// Standard: A = softmax(QK^T)V  [O(n²d)]
    /// Linformer: A ≈ softmax(Q(E^T K)^T)(E^T V)  [O(ndk)]
    pub fn forward(&self, Q: &[Vec<f32>], K: &[Vec<f32>], V: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let n = Q.len();
        let d = Q[0].len();

        // Project K: K_proj = E^T K  (k×d)
        let K_proj = Self::matrix_multiply_transpose(&self.E_key, K);

        // Project V: V_proj = E^T V  (k×d)
        let V_proj = Self::matrix_multiply_transpose(&self.E_value, V);

        // Compute attention scores: S = Q K_proj^T  (n×k)
        let mut scores = vec![vec![0.0; self.k]; n];
        for i in 0..n {
            for j in 0..self.k {
                for l in 0..d {
                    scores[i][j] += Q[i][l] * K_proj[j][l];
                }
            }
        }

        // Softmax over k dimension
        let attention = Self::softmax_matrix(&scores);

        // Output: attention × V_proj  (n×d)
        Self::matrix_multiply(&attention, &V_proj)
    }

    /// Matrix multiply: A × B
    fn matrix_multiply(A: &[Vec<f32>], B: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let rows_a = A.len();
        let cols_a = A[0].len();
        let cols_b = B[0].len();

        let mut result = vec![vec![0.0; cols_b]; rows_a];
        for i in 0..rows_a {
            for j in 0..cols_b {
                for k in 0..cols_a {
                    result[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        result
    }

    /// Matrix multiply with transpose: A^T × B
    fn matrix_multiply_transpose(A: &[Vec<f32>], B: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let cols_a = A[0].len();
        let rows_a = A.len();
        let cols_b = B[0].len();

        let mut result = vec![vec![0.0; cols_b]; cols_a];
        for i in 0..cols_a {
            for j in 0..cols_b {
                for k in 0..rows_a {
                    result[i][j] += A[k][i] * B[k][j];
                }
            }
        }
        result
    }

    /// Softmax over rows
    fn softmax_matrix(matrix: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let mut result = matrix.to_vec();
        for row in result.iter_mut() {
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = row.iter().map(|&x| (x - max).exp()).sum();
            for val in row.iter_mut() {
                *val = (*val - max).exp() / sum;
            }
        }
        result
    }
}

/// Performer: FAVOR+ algorithm
///
/// Key insight: Approximate softmax kernel using random features
/// softmax(qk^T) ≈ φ(q)^T φ(k) where φ is random feature map
///
/// Complexity: O(nd²) where d is typically small
pub struct Performer {
    pub d_model: usize,
    pub num_features: usize,
    /// Random projection matrix Ω (d×m)
    pub omega: Vec<Vec<f32>>,
}

impl Performer {
    /// Create new Performer layer
    pub fn new(d_model: usize, num_features: usize) -> Self {
        let omega = Self::init_random_features(d_model, num_features);

        Self {
            d_model,
            num_features,
            omega,
        }
    }

    /// Initialize random features (orthogonal random matrix)
    fn init_random_features(d: usize, m: usize) -> Vec<Vec<f32>> {
        let mut omega = vec![vec![0.0; m]; d];
        let scale = 1.0 / (d as f32).sqrt();

        // Simple random initialization (use proper orthogonal in production)
        for i in 0..d {
            for j in 0..m {
                let r = ((i * m + j) as f32 * 0.31415).fract();
                omega[i][j] = (r - 0.5) * 2.0 * scale;
            }
        }
        omega
    }

    /// Feature map φ(x) = exp(Ωx) (positive random features)
    fn feature_map(&self, x: &[f32]) -> Vec<f32> {
        let mut features = vec![0.0; self.num_features];

        // Ωx
        for i in 0..self.num_features {
            for j in 0..x.len() {
                features[i] += self.omega[j][i] * x[j];
            }
        }

        // exp(Ωx) / √m (normalization)
        let norm = (self.num_features as f32).sqrt();
        for val in features.iter_mut() {
            *val = val.exp() / norm;
        }

        features
    }

    /// Linear attention: φ(Q)(φ(K)^T V)
    pub fn forward(&self, Q: &[Vec<f32>], K: &[Vec<f32>], V: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let n = Q.len();
        let d = V[0].len();

        // Compute φ(K)^T V  (m×d)
        let phi_K: Vec<Vec<f32>> = K.iter().map(|k| self.feature_map(k)).collect();
        let KV = Self::transpose_multiply(&phi_K, V);

        // Compute φ(Q) (φ(K)^T V)  (n×d)
        let mut output = vec![vec![0.0; d]; n];
        for i in 0..n {
            let phi_q = self.feature_map(&Q[i]);
            for j in 0..d {
                for k in 0..self.num_features {
                    output[i][j] += phi_q[k] * KV[k][j];
                }
            }
        }

        output
    }

    /// Compute A^T B
    fn transpose_multiply(A: &[Vec<f32>], B: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let m = A[0].len();
        let n = B.len();
        let d = B[0].len();

        let mut result = vec![vec![0.0; d]; m];
        for i in 0..m {
            for j in 0..d {
                for k in 0..n {
                    result[i][j] += A[k][i] * B[k][j];
                }
            }
        }
        result
    }
}

/// FNet: Replace attention with Fourier Transform
///
/// Key insight: FFT provides global mixing at O(n log n) cost
/// No learned parameters for mixing!
///
/// Complexity: O(n log n) vs O(n²) attention
pub struct FNet {
    pub d_model: usize,
}

impl FNet {
    pub fn new(d_model: usize) -> Self {
        Self { d_model }
    }

    /// Apply 2D Fourier Transform (over sequence and feature dimensions)
    pub fn forward(&self, input: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let seq_len = input.len();
        let d_model = input[0].len();

        // 1D FFT over sequence dimension
        let mut after_seq_fft = Vec::with_capacity(d_model);
        for dim in 0..d_model {
            let sequence: Vec<f32> = input.iter().map(|token| token[dim]).collect();
            let fft_result = Self::fft_1d(&sequence);
            after_seq_fft.push(fft_result);
        }

        // 1D FFT over feature dimension
        let mut output = vec![vec![0.0; d_model]; seq_len];
        for pos in 0..seq_len {
            let features: Vec<f32> = after_seq_fft.iter().map(|dim_fft| dim_fft[pos]).collect();
            let fft_result = Self::fft_1d(&features);
            output[pos] = fft_result;
        }

        output
    }

    /// 1D Fast Fourier Transform (simplified - real part only)
    ///
    /// In production, use proper FFT library (e.g., rustfft)
    fn fft_1d(input: &[f32]) -> Vec<f32> {
        let n = input.len();
        let mut output = vec![0.0; n];

        // Discrete Fourier Transform (DFT) - O(n²) for simplicity
        // In production, use FFT O(n log n)
        for k in 0..n {
            let mut real = 0.0;
            for t in 0..n {
                let angle = -2.0 * PI * (k as f32) * (t as f32) / (n as f32);
                real += input[t] * angle.cos();
            }
            output[k] = real / (n as f32).sqrt(); // Normalization
        }

        output
    }
}

/// RWKV: Receptance Weighted Key Value
///
/// Key insight: Combine RNN and Transformer advantages
/// Uses time-mixing and channel-mixing without quadratic attention
///
/// Complexity: O(n) - truly linear!
pub struct RWKV {
    pub d_model: usize,
    /// Time-mixing weights
    pub mu_r: Vec<f32>,
    pub mu_k: Vec<f32>,
    pub mu_v: Vec<f32>,
    /// Projection matrices
    pub W_r: Vec<Vec<f32>>,
    pub W_k: Vec<Vec<f32>>,
    pub W_v: Vec<Vec<f32>>,
    pub W_o: Vec<Vec<f32>>,
}

impl RWKV {
    pub fn new(d_model: usize) -> Self {
        let mu_r = vec![0.5; d_model];
        let mu_k = vec![0.5; d_model];
        let mu_v = vec![0.5; d_model];

        let W_r = Self::init_matrix(d_model, d_model);
        let W_k = Self::init_matrix(d_model, d_model);
        let W_v = Self::init_matrix(d_model, d_model);
        let W_o = Self::init_matrix(d_model, d_model);

        Self {
            d_model,
            mu_r,
            mu_k,
            mu_v,
            W_r,
            W_k,
            W_v,
            W_o,
        }
    }

    fn init_matrix(rows: usize, cols: usize) -> Vec<Vec<f32>> {
        let mut matrix = vec![vec![0.0; cols]; rows];
        let scale = (1.0 / cols as f32).sqrt();
        for i in 0..rows {
            for j in 0..cols {
                let r = ((i * cols + j) as f32 * 0.271828).fract();
                matrix[i][j] = (r - 0.5) * 2.0 * scale;
            }
        }
        matrix
    }

    /// Time-mixing: interpolate between current and previous token
    fn time_mix(&self, x_current: &[f32], x_prev: &[f32], mu: &[f32]) -> Vec<f32> {
        x_current
            .iter()
            .zip(x_prev.iter())
            .zip(mu.iter())
            .map(|((&curr, &prev), &mu_val)| mu_val * curr + (1.0 - mu_val) * prev)
            .collect()
    }

    fn matmul(W: &[Vec<f32>], x: &[f32]) -> Vec<f32> {
        let mut result = vec![0.0; W.len()];
        for i in 0..W.len() {
            for j in 0..x.len().min(W[0].len()) {
                result[i] += W[i][j] * x[j];
            }
        }
        result
    }

    /// Forward pass (single step with state)
    pub fn forward_step(
        &self,
        x_current: &[f32],
        x_prev: &[f32],
        state: &mut Vec<f32>,
    ) -> Vec<f32> {
        // Time-mixing
        let x_r = self.time_mix(x_current, x_prev, &self.mu_r);
        let x_k = self.time_mix(x_current, x_prev, &self.mu_k);
        let x_v = self.time_mix(x_current, x_prev, &self.mu_v);

        // Receptance, Key, Value
        let r = Self::matmul(&self.W_r, &x_r).iter().map(|&x| sigmoid(x)).collect::<Vec<_>>();
        let k = Self::matmul(&self.W_k, &x_k).iter().map(|&x| x.exp()).collect::<Vec<_>>();
        let v = Self::matmul(&self.W_v, &x_v);

        // WKV (Weighted Key-Value)
        let mut wkv = vec![0.0; self.d_model];
        for i in 0..self.d_model.min(r.len()).min(k.len()).min(v.len()).min(state.len()) {
            wkv[i] = (r[i] * state[i]) / (k[i] + 1e-8);
            // Update state for next step
            state[i] = state[i] + k[i] * v[i];
        }

        // Output
        Self::matmul(&self.W_o, &wkv)
    }

    /// Forward pass on sequence
    pub fn forward_sequence(&self, inputs: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let mut outputs = Vec::with_capacity(inputs.len());
        let mut state = vec![0.0; self.d_model];
        let mut x_prev = vec![0.0; self.d_model];

        for x_current in inputs {
            let output = self.forward_step(x_current, &x_prev, &mut state);
            outputs.push(output.clone());
            x_prev = x_current.clone();
        }

        outputs
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linformer_creation() {
        let linformer = Linformer::new(64, 8, 512, 128);
        assert_eq!(linformer.d_model, 64);
        assert_eq!(linformer.num_heads, 8);
        assert_eq!(linformer.k, 128);
        assert_eq!(linformer.E_key.len(), 512);
        assert_eq!(linformer.E_key[0].len(), 128);
    }

    #[test]
    fn test_linformer_forward() {
        let linformer = Linformer::new(32, 4, 64, 16);
        let Q = vec![vec![0.5; 32]; 64];
        let K = vec![vec![0.3; 32]; 64];
        let V = vec![vec![0.7; 32]; 64];

        let output = linformer.forward(&Q, &K, &V);
        assert_eq!(output.len(), 64);
        assert_eq!(output[0].len(), 32);
    }

    #[test]
    fn test_performer_creation() {
        let performer = Performer::new(64, 256);
        assert_eq!(performer.d_model, 64);
        assert_eq!(performer.num_features, 256);
        assert_eq!(performer.omega.len(), 64);
        assert_eq!(performer.omega[0].len(), 256);
    }

    #[test]
    fn test_performer_feature_map() {
        let performer = Performer::new(32, 128);
        let x = vec![0.5; 32];
        let features = performer.feature_map(&x);

        assert_eq!(features.len(), 128);
        // Features should be positive (exp)
        assert!(features.iter().all(|&f| f > 0.0));
    }

    #[test]
    fn test_performer_forward() {
        let performer = Performer::new(16, 64);
        let Q = vec![vec![0.5; 16]; 32];
        let K = vec![vec![0.3; 16]; 32];
        let V = vec![vec![0.7; 16]; 32];

        let output = performer.forward(&Q, &K, &V);
        assert_eq!(output.len(), 32);
        assert_eq!(output[0].len(), 16);
    }

    #[test]
    fn test_fnet_creation() {
        let fnet = FNet::new(128);
        assert_eq!(fnet.d_model, 128);
    }

    #[test]
    fn test_fnet_forward() {
        let fnet = FNet::new(16);
        let input = vec![vec![1.0; 16]; 32];

        let output = fnet.forward(&input);
        assert_eq!(output.len(), 32);
        assert_eq!(output[0].len(), 16);
    }

    #[test]
    fn test_fft_1d() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let output = FNet::fft_1d(&input);

        assert_eq!(output.len(), 4);
        // DC component should be average (simplified check)
        assert!(output[0].abs() > 0.0);
    }

    #[test]
    fn test_rwkv_creation() {
        let rwkv = RWKV::new(64);
        assert_eq!(rwkv.d_model, 64);
        assert_eq!(rwkv.mu_r.len(), 64);
        assert_eq!(rwkv.W_r.len(), 64);
        assert_eq!(rwkv.W_r[0].len(), 64);
    }

    #[test]
    fn test_rwkv_time_mix() {
        let rwkv = RWKV::new(4);
        let x_current = vec![1.0, 2.0, 3.0, 4.0];
        let x_prev = vec![0.0, 0.0, 0.0, 0.0];
        let mu = vec![0.5, 0.5, 0.5, 0.5];

        let mixed = rwkv.time_mix(&x_current, &x_prev, &mu);
        assert_eq!(mixed.len(), 4);
        // Should be average
        assert!((mixed[0] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_rwkv_forward_step() {
        let rwkv = RWKV::new(16);
        let x_current = vec![0.5; 16];
        let x_prev = vec![0.3; 16];
        let mut state = vec![0.0; 16];

        let output = rwkv.forward_step(&x_current, &x_prev, &mut state);
        assert_eq!(output.len(), 16);
        // State should be updated
        assert!(state.iter().any(|&x| x != 0.0));
    }

    #[test]
    fn test_rwkv_sequence() {
        let rwkv = RWKV::new(8);
        let inputs = vec![
            vec![1.0; 8],
            vec![0.5; 8],
            vec![0.25; 8],
        ];

        let outputs = rwkv.forward_sequence(&inputs);
        assert_eq!(outputs.len(), 3);
        assert_eq!(outputs[0].len(), 8);
    }
}
