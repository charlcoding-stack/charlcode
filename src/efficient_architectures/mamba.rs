// Mamba: Selective State Space Models
//
// Mamba improves upon S4 by making the SSM parameters data-dependent (selective).
// This allows the model to focus on important parts of the sequence and filter noise.
//
// Key innovation: Instead of fixed A, B, C, D parameters,
// they are computed dynamically based on the input.
//
// Allow SSM notation (A, B, C matrices) to match academic literature
#![allow(non_snake_case)]
//
// Architecture:
//   input x → Linear projection →
//   ├─ Selective SSM path (selective parameters)
//   └─ Gated path (multiplicative gating)
//   → Combine → Output
//
// Selective SSM:
//   B, C, Δ = f(x)  // Data-dependent parameters
//   A is fixed (HiPPO)
//   x_{k+1} = Ā(Δ) x_k + B̄(Δ) u_k
//   y_k = C x_k
//
// Complexity: O(n) vs O(n²) for transformers
//
// References:
// - Gu & Dao (2023): "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"

/// Mamba layer configuration
#[derive(Debug, Clone)]
pub struct MambaConfig {
    /// Model dimension
    pub d_model: usize,
    /// State dimension
    pub d_state: usize,
    /// Expansion factor (typically 2)
    pub expand_factor: usize,
    /// Convolution width for local context
    pub d_conv: usize,
    /// Discretization timestep range
    pub dt_min: f32,
    pub dt_max: f32,
}

impl MambaConfig {
    pub fn new(d_model: usize) -> Self {
        Self {
            d_model,
            d_state: 16, // Default state size
            expand_factor: 2,
            d_conv: 4,
            dt_min: 0.001,
            dt_max: 0.1,
        }
    }

    pub fn with_state_size(mut self, d_state: usize) -> Self {
        self.d_state = d_state;
        self
    }

    pub fn with_expand_factor(mut self, factor: usize) -> Self {
        self.expand_factor = factor;
        self
    }
}

/// Selective SSM parameters (data-dependent)
#[derive(Debug, Clone)]
pub struct SelectiveParams {
    /// Input-dependent B (N×H)
    pub B: Vec<Vec<f32>>,
    /// Input-dependent C (H×N)
    pub C: Vec<Vec<f32>>,
    /// Input-dependent timestep Δ
    pub delta: Vec<f32>,
}

/// Mamba block: Selective SSM + Gating
#[derive(Debug, Clone)]
pub struct MambaBlock {
    pub config: MambaConfig,
    /// Fixed state matrix A (HiPPO initialized)
    pub A: Vec<Vec<f32>>,
    /// Linear projection for input
    pub in_proj: Vec<Vec<f32>>,
    /// Linear projection for output
    pub out_proj: Vec<Vec<f32>>,
    /// Convolution weights for local context
    pub conv_weights: Vec<f32>,
    /// Parameter projection weights (for B, C, Δ)
    pub param_proj: Vec<Vec<f32>>,
}

impl MambaBlock {
    /// Create new Mamba block
    pub fn new(config: MambaConfig) -> Self {
        let d_model = config.d_model;
        let d_state = config.d_state;
        let d_inner = d_model * config.expand_factor;

        // Initialize A with HiPPO
        let A = Self::initialize_hippo(d_state);

        // Input projection: d_model → 2*d_inner (for both SSM and gate)
        let in_proj = Self::init_matrix(2 * d_inner, d_model, 0.1);

        // Output projection: d_inner → d_model
        let out_proj = Self::init_matrix(d_model, d_inner, 0.1);

        // Convolution for local context
        let conv_weights = vec![0.1; config.d_conv * d_inner];

        // Parameter projection: d_inner → (B_size + C_size + 1)
        // B_size = d_state, C_size = d_state, 1 for delta
        let param_size = d_state + d_state + 1;
        let param_proj = Self::init_matrix(param_size, d_inner, 0.1);

        Self {
            config,
            A,
            in_proj,
            out_proj,
            conv_weights,
            param_proj,
        }
    }

    /// Initialize HiPPO matrix (same as S4)
    fn initialize_hippo(n: usize) -> Vec<Vec<f32>> {
        let mut A = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                if i > j {
                    let val = -((2 * i + 1) as f32).sqrt() * ((2 * j + 1) as f32).sqrt();
                    A[i][j] = val;
                } else if i == j {
                    A[i][j] = -((2 * i + 1) as f32);
                }
            }
        }
        A
    }

    /// Initialize matrix with small random values
    fn init_matrix(rows: usize, cols: usize, scale: f32) -> Vec<Vec<f32>> {
        let mut matrix = vec![vec![0.0; cols]; rows];
        for i in 0..rows {
            for j in 0..cols {
                let r = ((i * cols + j) as f32 * 0.123456).fract();
                matrix[i][j] = (r - 0.5) * 2.0 * scale;
            }
        }
        matrix
    }

    /// Compute selective parameters from input
    ///
    /// B, C, Δ = f(x)  // Data-dependent
    fn compute_selective_params(&self, x_projected: &[f32]) -> SelectiveParams {
        let d_state = self.config.d_state;

        // Project to parameter space
        let mut params = vec![0.0; self.param_proj.len()];
        for i in 0..self.param_proj.len() {
            for j in 0..x_projected.len().min(self.param_proj[0].len()) {
                params[i] += self.param_proj[i][j] * x_projected[j];
            }
        }

        // Split into B, C, delta
        let mut B = vec![vec![0.0; 1]; d_state];
        let mut C = vec![vec![0.0; d_state]; 1];
        let mut delta = vec![self.config.dt_min; 1];

        // Extract B (first d_state params)
        for i in 0..d_state {
            if i < params.len() {
                B[i][0] = params[i];
            }
        }

        // Extract C (next d_state params)
        for i in 0..d_state {
            if d_state + i < params.len() {
                C[0][i] = params[d_state + i];
            }
        }

        // Extract delta (last param) and clamp
        if params.len() > 2 * d_state {
            let delta_raw = params[2 * d_state];
            // Softplus to ensure positive: log(1 + exp(x))
            let delta_val = (1.0 + delta_raw.exp()).ln();
            delta[0] = delta_val.clamp(self.config.dt_min, self.config.dt_max);
        }

        SelectiveParams { B, C, delta }
    }

    /// Discretize with selective parameters
    ///
    /// Ā = exp(Δ·A), B̄ = (Ā - I)·A⁻¹·B ≈ Δ·B
    fn selective_discretize(&self, delta: f32) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let n = self.config.d_state;

        // Simplified: Ā ≈ I + Δ·A
        let mut A_bar = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    A_bar[i][j] = 1.0 + delta * self.A[i][j];
                } else {
                    A_bar[i][j] = delta * self.A[i][j];
                }
            }
        }

        // B̄ ≈ Δ·B (computed per-token with selective B)
        let B_bar = A_bar.clone(); // Placeholder

        (A_bar, B_bar)
    }

    /// Forward pass with selective SSM
    pub fn forward(&self, input: &[f32], state: &mut Vec<f32>) -> Vec<f32> {
        let d_model = self.config.d_model;
        let d_inner = d_model * self.config.expand_factor;

        // Input projection: x → [x_ssm, x_gate]
        let mut projected = vec![0.0; 2 * d_inner];
        for i in 0..projected.len().min(self.in_proj.len()) {
            for j in 0..input.len().min(self.in_proj[0].len()) {
                projected[i] += self.in_proj[i][j] * input[j];
            }
        }

        // Split into SSM path and gate path
        let x_ssm = &projected[0..d_inner];
        let x_gate = &projected[d_inner..];

        // Compute selective parameters
        let sel_params = self.compute_selective_params(x_ssm);

        // Selective SSM step
        let (A_bar, _B_bar) = self.selective_discretize(sel_params.delta[0]);

        // Update state: x_{k+1} = Ā x_k + B̄ u_k
        let mut new_state = vec![0.0; self.config.d_state];
        for i in 0..self.config.d_state {
            // Ā x_k
            for j in 0..self.config.d_state {
                new_state[i] += A_bar[i][j] * state[j];
            }
            // + B̄ u_k (using selective B)
            for j in 0..x_ssm.len().min(1) {
                new_state[i] += sel_params.delta[0] * sel_params.B[i][0] * x_ssm[j];
            }
        }
        *state = new_state;

        // Output: y = C x_k (using selective C)
        let mut y_ssm = vec![0.0; d_inner];
        for i in 0..d_inner.min(1) {
            for j in 0..self.config.d_state {
                y_ssm[i] += sel_params.C[0][j] * state[j];
            }
        }

        // Gating: y_ssm * σ(x_gate)
        let mut gated = vec![0.0; d_inner];
        for i in 0..d_inner.min(y_ssm.len()).min(x_gate.len()) {
            let gate_val = sigmoid(x_gate[i]);
            gated[i] = y_ssm[i] * gate_val;
        }

        // Output projection
        let mut output = vec![0.0; d_model];
        for i in 0..d_model.min(self.out_proj.len()) {
            for j in 0..gated.len().min(self.out_proj[0].len()) {
                output[i] += self.out_proj[i][j] * gated[j];
            }
        }

        output
    }

    /// Forward pass on sequence
    pub fn forward_sequence(&self, inputs: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let mut state = vec![0.0; self.config.d_state];
        let mut outputs = Vec::with_capacity(inputs.len());

        for input in inputs {
            let output = self.forward(input, &mut state);
            outputs.push(output);
        }

        outputs
    }
}

/// Sigmoid activation
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Gated SSM unit (used in Mamba)
pub struct GatedSSM {
    pub ssm: MambaBlock,
    pub gate_proj: Vec<Vec<f32>>,
}

impl GatedSSM {
    pub fn new(config: MambaConfig) -> Self {
        let d_model = config.d_model;
        let d_inner = d_model * config.expand_factor;

        let ssm = MambaBlock::new(config);
        let gate_proj = MambaBlock::init_matrix(d_inner, d_model, 0.1);

        Self { ssm, gate_proj }
    }

    pub fn forward(&self, input: &[f32], state: &mut Vec<f32>) -> Vec<f32> {
        self.ssm.forward(input, state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mamba_config() {
        let config = MambaConfig::new(128)
            .with_state_size(16)
            .with_expand_factor(2);

        assert_eq!(config.d_model, 128);
        assert_eq!(config.d_state, 16);
        assert_eq!(config.expand_factor, 2);
    }

    #[test]
    fn test_mamba_block_creation() {
        let config = MambaConfig::new(64);
        let block = MambaBlock::new(config);

        assert_eq!(block.A.len(), 16); // Default d_state
        assert_eq!(block.A[0].len(), 16);
    }

    #[test]
    fn test_selective_params_computation() {
        let config = MambaConfig::new(32);
        let block = MambaBlock::new(config.clone());

        let x_projected = vec![0.5; 32 * 2];
        let params = block.compute_selective_params(&x_projected);

        assert_eq!(params.B.len(), config.d_state);
        assert_eq!(params.C.len(), 1);
        assert_eq!(params.C[0].len(), config.d_state);
        assert_eq!(params.delta.len(), 1);

        // Delta should be within bounds
        assert!(params.delta[0] >= config.dt_min);
        assert!(params.delta[0] <= config.dt_max);
    }

    #[test]
    fn test_selective_discretize() {
        let config = MambaConfig::new(32);
        let block = MambaBlock::new(config);

        let delta = 0.01;
        let (A_bar, _B_bar) = block.selective_discretize(delta);

        assert_eq!(A_bar.len(), block.config.d_state);
        assert_eq!(A_bar[0].len(), block.config.d_state);
    }

    #[test]
    fn test_mamba_forward() {
        let config = MambaConfig::new(32);
        let block = MambaBlock::new(config.clone());

        let input = vec![0.5; 32];
        let mut state = vec![0.0; config.d_state];

        let output = block.forward(&input, &mut state);

        assert_eq!(output.len(), 32);
        // State should be updated
        assert!(state.iter().any(|&x| x != 0.0));
    }

    #[test]
    fn test_mamba_sequence() {
        let config = MambaConfig::new(16);
        let block = MambaBlock::new(config);

        let inputs = vec![vec![1.0; 16], vec![0.5; 16], vec![0.25; 16]];

        let outputs = block.forward_sequence(&inputs);

        assert_eq!(outputs.len(), 3);
        assert_eq!(outputs[0].len(), 16);
    }

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-5);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);
    }

    #[test]
    fn test_gated_ssm() {
        let config = MambaConfig::new(32);
        let gated = GatedSSM::new(config.clone());

        let input = vec![0.5; 32];
        let mut state = vec![0.0; config.d_state];

        let output = gated.forward(&input, &mut state);

        assert_eq!(output.len(), 32);
    }

    #[test]
    fn test_state_evolution() {
        let config = MambaConfig::new(16);
        let block = MambaBlock::new(config.clone());

        let mut state = vec![0.0; config.d_state];
        let input = vec![1.0; 16];

        // Step 1
        let _ = block.forward(&input, &mut state);
        let state_1 = state.clone();

        // Step 2
        let _ = block.forward(&input, &mut state);
        let state_2 = state.clone();

        // States should be different
        assert_ne!(state_1, state_2);
    }

    #[test]
    fn test_hippo_initialization() {
        let A = MambaBlock::initialize_hippo(4);

        // HiPPO is lower triangular
        assert_eq!(A[0][1], 0.0);
        assert_eq!(A[0][2], 0.0);

        // Diagonal is negative
        assert!(A[0][0] < 0.0);
        assert!(A[1][1] < 0.0);
    }
}
