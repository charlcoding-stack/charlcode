// S4: Structured State Spaces
//
// S4 replaces attention with state space models, achieving O(n) complexity
// instead of O(n²) for transformers on sequence length n.
//
// Continuous-time state space model:
//   dx/dt = Ax + Bu
//   y = Cx + Du
//
// Where:
//   - A (N×N): State matrix (HiPPO initialized)
//   - B (N×1): Input matrix
//   - C (1×N): Output matrix
//   - D (scalar): Feedthrough
//   - x (N): Hidden state
//   - u: Input
//   - y: Output
//
// Discretization (Zero-Order Hold):
//   x_k+1 = Ā x_k + B̄ u_k
//   y_k = C x_k + D u_k
//
// Where Ā = exp(Δ·A), B̄ = (Ā - I)·A⁻¹·B
//
// References:
// - Gu et al. (2021): "Efficiently Modeling Long Sequences with Structured State Spaces"
// - Gu et al. (2022): "How to Train Your HiPPO"

use std::f32::consts::PI;

/// State space model configuration
#[derive(Debug, Clone)]
pub struct SSMConfig {
    /// State dimension (N)
    pub state_size: usize,
    /// Input/output dimension (H)
    pub hidden_size: usize,
    /// Discretization step size (Δ)
    pub dt: f32,
    /// Initialization strategy
    pub init_strategy: InitStrategy,
}

impl SSMConfig {
    pub fn new(state_size: usize, hidden_size: usize) -> Self {
        Self {
            state_size,
            hidden_size,
            dt: 0.001, // Default timestep
            init_strategy: InitStrategy::HiPPO,
        }
    }

    pub fn with_dt(mut self, dt: f32) -> Self {
        self.dt = dt;
        self
    }

    pub fn with_init_strategy(mut self, strategy: InitStrategy) -> Self {
        self.init_strategy = strategy;
        self
    }
}

/// Initialization strategy for state matrix A
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InitStrategy {
    /// HiPPO (High-order Polynomial Projection Operator)
    HiPPO,
    /// Random initialization
    Random,
    /// Identity matrix
    Identity,
}

/// Discretization method
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DiscretizationMethod {
    /// Zero-Order Hold (ZOH)
    ZeroOrderHold,
    /// Bilinear transform (Tustin)
    Bilinear,
    /// Forward Euler
    Euler,
}

/// S4 Layer: Structured State Space Model
pub struct S4Layer {
    /// Configuration
    pub config: SSMConfig,
    /// Continuous state matrix A (N×N)
    pub A: Vec<Vec<f32>>,
    /// Input matrix B (N×H)
    pub B: Vec<Vec<f32>>,
    /// Output matrix C (H×N)
    pub C: Vec<Vec<f32>>,
    /// Feedthrough D (H×H)
    pub D: Vec<Vec<f32>>,
    /// Discretized state matrix Ā (cached)
    A_discrete: Option<Vec<Vec<f32>>>,
    /// Discretized input matrix B̄ (cached)
    B_discrete: Option<Vec<Vec<f32>>>,
}

impl S4Layer {
    /// Create new S4 layer
    pub fn new(config: SSMConfig) -> Self {
        let state_size = config.state_size;
        let hidden_size = config.hidden_size;

        // Initialize matrices
        let A = Self::initialize_A(state_size, config.init_strategy);
        let B = Self::initialize_matrix(state_size, hidden_size, 0.1);
        let C = Self::initialize_matrix(hidden_size, state_size, 0.1);
        let D = Self::initialize_matrix(hidden_size, hidden_size, 0.01);

        Self {
            config,
            A,
            B,
            C,
            D,
            A_discrete: None,
            B_discrete: None,
        }
    }

    /// Initialize state matrix A using HiPPO
    ///
    /// HiPPO matrices are optimal for memorizing sequences
    fn initialize_A(n: usize, strategy: InitStrategy) -> Vec<Vec<f32>> {
        match strategy {
            InitStrategy::HiPPO => {
                // HiPPO-LegS matrix (simplified)
                let mut A = vec![vec![0.0; n]; n];
                for i in 0..n {
                    for j in 0..n {
                        if i > j {
                            // Lower triangular: A_ij = -(2i+1)^{1/2} * (2j+1)^{1/2}
                            let val = -((2 * i + 1) as f32).sqrt() * ((2 * j + 1) as f32).sqrt();
                            A[i][j] = val;
                        } else if i == j {
                            // Diagonal: A_ii = -(2i+1)
                            A[i][j] = -((2 * i + 1) as f32);
                        }
                        // Upper triangular is zero
                    }
                }
                A
            }
            InitStrategy::Random => Self::initialize_matrix(n, n, 0.1),
            InitStrategy::Identity => {
                let mut A = vec![vec![0.0; n]; n];
                for i in 0..n {
                    A[i][i] = 1.0;
                }
                A
            }
        }
    }

    /// Initialize matrix with small random values
    fn initialize_matrix(rows: usize, cols: usize, scale: f32) -> Vec<Vec<f32>> {
        let mut matrix = vec![vec![0.0; cols]; rows];
        for i in 0..rows {
            for j in 0..cols {
                // Simple pseudo-random (use proper RNG in production)
                let r = ((i * cols + j) as f32 * 0.123456).fract();
                matrix[i][j] = (r - 0.5) * 2.0 * scale;
            }
        }
        matrix
    }

    /// Discretize the continuous SSM using Zero-Order Hold
    ///
    /// Ā = exp(Δ·A)
    /// B̄ = (Ā - I)·A⁻¹·B ≈ Δ·B for small Δ
    pub fn discretize(&mut self, method: DiscretizationMethod) {
        let n = self.config.state_size;
        let dt = self.config.dt;

        match method {
            DiscretizationMethod::ZeroOrderHold => {
                // Simplified: Ā ≈ I + Δ·A (first-order approximation)
                let mut A_discrete = vec![vec![0.0; n]; n];
                for i in 0..n {
                    for j in 0..n {
                        if i == j {
                            A_discrete[i][j] = 1.0 + dt * self.A[i][j];
                        } else {
                            A_discrete[i][j] = dt * self.A[i][j];
                        }
                    }
                }
                self.A_discrete = Some(A_discrete);

                // B̄ ≈ Δ·B
                let mut B_discrete = vec![vec![0.0; self.config.hidden_size]; n];
                for i in 0..n {
                    for j in 0..self.config.hidden_size {
                        B_discrete[i][j] = dt * self.B[i][j];
                    }
                }
                self.B_discrete = Some(B_discrete);
            }
            DiscretizationMethod::Bilinear => {
                // Bilinear transform: Ā = (I + Δ/2·A)(I - Δ/2·A)⁻¹
                // Simplified first-order approximation
                let mut A_discrete = vec![vec![0.0; n]; n];
                for i in 0..n {
                    for j in 0..n {
                        if i == j {
                            A_discrete[i][j] = 1.0 + dt * 0.5 * self.A[i][j];
                        } else {
                            A_discrete[i][j] = dt * 0.5 * self.A[i][j];
                        }
                    }
                }
                self.A_discrete = Some(A_discrete);

                let mut B_discrete = vec![vec![0.0; self.config.hidden_size]; n];
                for i in 0..n {
                    for j in 0..self.config.hidden_size {
                        B_discrete[i][j] = dt * self.B[i][j];
                    }
                }
                self.B_discrete = Some(B_discrete);
            }
            DiscretizationMethod::Euler => {
                // Forward Euler: same as ZOH first-order
                self.discretize(DiscretizationMethod::ZeroOrderHold);
            }
        }
    }

    /// Forward pass (recurrent mode - O(n) per step)
    ///
    /// x_{k+1} = Ā x_k + B̄ u_k
    /// y_k = C x_k + D u_k
    pub fn forward_recurrent(&self, input: &[f32], state: &mut Vec<f32>) -> Vec<f32> {
        assert_eq!(input.len(), self.config.hidden_size, "Input dimension mismatch");
        assert_eq!(state.len(), self.config.state_size, "State dimension mismatch");

        let A_bar = self.A_discrete.as_ref().expect("Must discretize first");
        let B_bar = self.B_discrete.as_ref().expect("Must discretize first");

        // Update state: x_{k+1} = Ā x_k + B̄ u_k
        let mut new_state = vec![0.0; self.config.state_size];

        // Ā x_k
        for i in 0..self.config.state_size {
            for j in 0..self.config.state_size {
                new_state[i] += A_bar[i][j] * state[j];
            }
        }

        // + B̄ u_k
        for i in 0..self.config.state_size {
            for j in 0..self.config.hidden_size {
                new_state[i] += B_bar[i][j] * input[j];
            }
        }

        *state = new_state;

        // Compute output: y_k = C x_k + D u_k
        let mut output = vec![0.0; self.config.hidden_size];

        // C x_k
        for i in 0..self.config.hidden_size {
            for j in 0..self.config.state_size {
                output[i] += self.C[i][j] * state[j];
            }
        }

        // + D u_k
        for i in 0..self.config.hidden_size {
            for j in 0..self.config.hidden_size {
                output[i] += self.D[i][j] * input[j];
            }
        }

        output
    }

    /// Forward pass on sequence (O(n·L) for sequence length L)
    pub fn forward_sequence(&self, inputs: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let mut state = vec![0.0; self.config.state_size];
        let mut outputs = Vec::with_capacity(inputs.len());

        for input in inputs {
            let output = self.forward_recurrent(input, &mut state);
            outputs.push(output);
        }

        outputs
    }

    /// Get current configuration
    pub fn get_config(&self) -> &SSMConfig {
        &self.config
    }

    /// Check if discretized
    pub fn is_discretized(&self) -> bool {
        self.A_discrete.is_some() && self.B_discrete.is_some()
    }
}

/// Parallel scan for efficient training (O(n log n))
///
/// Computes all hidden states in parallel using associative scan
pub struct ParallelScan;

impl ParallelScan {
    /// Parallel scan operation for SSM
    ///
    /// Given sequence of inputs u_0, u_1, ..., u_{L-1}
    /// Compute all states x_0, x_1, ..., x_{L-1} in parallel
    pub fn scan(
        A_bar: &[Vec<f32>],
        B_bar: &[Vec<f32>],
        inputs: &[Vec<f32>],
        initial_state: &[f32],
    ) -> Vec<Vec<f32>> {
        let seq_len = inputs.len();
        let state_size = initial_state.len();

        // For simplicity, use sequential scan
        // In production, use parallel prefix sum with GPU
        let mut states = Vec::with_capacity(seq_len);
        let mut current_state = initial_state.to_vec();

        for input in inputs {
            // x_{k+1} = Ā x_k + B̄ u_k
            let mut next_state = vec![0.0; state_size];

            for i in 0..state_size {
                for j in 0..state_size {
                    next_state[i] += A_bar[i][j] * current_state[j];
                }
                for j in 0..input.len() {
                    next_state[i] += B_bar[i][j] * input[j];
                }
            }

            states.push(next_state.clone());
            current_state = next_state;
        }

        states
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ssm_config_creation() {
        let config = SSMConfig::new(64, 128)
            .with_dt(0.01)
            .with_init_strategy(InitStrategy::HiPPO);

        assert_eq!(config.state_size, 64);
        assert_eq!(config.hidden_size, 128);
        assert_eq!(config.dt, 0.01);
        assert_eq!(config.init_strategy, InitStrategy::HiPPO);
    }

    #[test]
    fn test_s4_layer_creation() {
        let config = SSMConfig::new(32, 64);
        let layer = S4Layer::new(config);

        assert_eq!(layer.A.len(), 32);
        assert_eq!(layer.A[0].len(), 32);
        assert_eq!(layer.B.len(), 32);
        assert_eq!(layer.B[0].len(), 64);
        assert_eq!(layer.C.len(), 64);
        assert_eq!(layer.C[0].len(), 32);
    }

    #[test]
    fn test_hippo_initialization() {
        let A = S4Layer::initialize_A(4, InitStrategy::HiPPO);

        // Check dimensions
        assert_eq!(A.len(), 4);
        assert_eq!(A[0].len(), 4);

        // HiPPO is lower triangular
        assert_eq!(A[0][1], 0.0);
        assert_eq!(A[0][2], 0.0);
        assert_eq!(A[1][2], 0.0);

        // Diagonal is negative
        assert!(A[0][0] < 0.0);
        assert!(A[1][1] < 0.0);
    }

    #[test]
    fn test_identity_initialization() {
        let A = S4Layer::initialize_A(3, InitStrategy::Identity);

        assert_eq!(A[0][0], 1.0);
        assert_eq!(A[1][1], 1.0);
        assert_eq!(A[2][2], 1.0);
        assert_eq!(A[0][1], 0.0);
        assert_eq!(A[1][0], 0.0);
    }

    #[test]
    fn test_discretization() {
        let config = SSMConfig::new(4, 8).with_dt(0.01);
        let mut layer = S4Layer::new(config);

        assert!(!layer.is_discretized());

        layer.discretize(DiscretizationMethod::ZeroOrderHold);

        assert!(layer.is_discretized());
        assert!(layer.A_discrete.is_some());
        assert!(layer.B_discrete.is_some());
    }

    #[test]
    fn test_forward_recurrent() {
        let config = SSMConfig::new(4, 2).with_dt(0.1);
        let mut layer = S4Layer::new(config);
        layer.discretize(DiscretizationMethod::ZeroOrderHold);

        let input = vec![1.0, 0.5];
        let mut state = vec![0.0; 4];

        let output = layer.forward_recurrent(&input, &mut state);

        assert_eq!(output.len(), 2);
        // State should have been updated
        assert!(state.iter().any(|&x| x != 0.0));
    }

    #[test]
    fn test_forward_sequence() {
        let config = SSMConfig::new(8, 4).with_dt(0.01);
        let mut layer = S4Layer::new(config);
        layer.discretize(DiscretizationMethod::ZeroOrderHold);

        let inputs = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
        ];

        let outputs = layer.forward_sequence(&inputs);

        assert_eq!(outputs.len(), 3);
        assert_eq!(outputs[0].len(), 4);
    }

    #[test]
    fn test_parallel_scan() {
        let n = 4;
        let h = 2;

        // Simple identity-like matrices for testing
        let mut A_bar = vec![vec![0.0; n]; n];
        for i in 0..n {
            A_bar[i][i] = 0.9; // Decay factor
        }

        let B_bar = vec![vec![0.1; h]; n];

        let inputs = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
        ];

        let initial_state = vec![0.0; n];

        let states = ParallelScan::scan(&A_bar, &B_bar, &inputs, &initial_state);

        assert_eq!(states.len(), 2);
        assert_eq!(states[0].len(), n);
    }

    #[test]
    fn test_state_accumulation() {
        let config = SSMConfig::new(4, 2).with_dt(0.1);
        let mut layer = S4Layer::new(config);
        layer.discretize(DiscretizationMethod::ZeroOrderHold);

        let mut state = vec![0.0; 4];
        let input = vec![1.0, 1.0];

        // First step
        let _ = layer.forward_recurrent(&input, &mut state);
        let state_after_1 = state.clone();

        // Second step - state should change
        let _ = layer.forward_recurrent(&input, &mut state);
        let state_after_2 = state.clone();

        // States should be different
        assert_ne!(state_after_1, state_after_2);
    }

    #[test]
    fn test_different_discretization_methods() {
        let config = SSMConfig::new(4, 2).with_dt(0.01);

        let mut layer_zoh = S4Layer::new(config.clone());
        layer_zoh.discretize(DiscretizationMethod::ZeroOrderHold);

        let mut layer_bilinear = S4Layer::new(config.clone());
        layer_bilinear.discretize(DiscretizationMethod::Bilinear);

        let mut layer_euler = S4Layer::new(config);
        layer_euler.discretize(DiscretizationMethod::Euler);

        // All should be discretized
        assert!(layer_zoh.is_discretized());
        assert!(layer_bilinear.is_discretized());
        assert!(layer_euler.is_discretized());
    }
}
