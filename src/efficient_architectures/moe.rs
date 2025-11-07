// Mixture of Experts (MoE) - Sparse Conditional Computation
//
// Key insight: Instead of using full network for every input,
// route each input to a subset of "expert" networks.
//
// Allow weight matrix notation (W1, W2, W_router) for clarity
#![allow(non_snake_case)]
//
// Benefits:
// - 10x model capacity with 2x compute cost
// - Sparse activation (only top-K experts per token)
// - Scalable to thousands of experts
//
// Architecture:
//   input → Router → Top-K experts → Combine → output
//
// Router learns which experts to use for each input.
// Only top-K experts are computed (sparse gating).
//
// Example: 64 experts, top-2 routing
//   → 2/64 = 3% of experts active
//   → 32x less computation than dense layer
//
// References:
// - Shazeer et al. (2017): "Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer"
// - Fedus et al. (2021): "Switch Transformers: Scaling to Trillion Parameter Models"
// - Lepikhin et al. (2021): "GShard: Scaling Giant Models with Conditional Computation"

use std::collections::HashMap;

/// Expert network (simple feedforward for this implementation)
#[derive(Debug, Clone)]
pub struct Expert {
    /// Input dimension
    pub d_model: usize,
    /// Hidden dimension
    pub d_ff: usize,
    /// First layer weights (d_ff × d_model)
    pub W1: Vec<Vec<f32>>,
    /// Second layer weights (d_model × d_ff)
    pub W2: Vec<Vec<f32>>,
}

impl Expert {
    /// Create new expert
    pub fn new(d_model: usize, d_ff: usize) -> Self {
        let W1 = Self::init_matrix(d_ff, d_model);
        let W2 = Self::init_matrix(d_model, d_ff);

        Self {
            d_model,
            d_ff,
            W1,
            W2,
        }
    }

    fn init_matrix(rows: usize, cols: usize) -> Vec<Vec<f32>> {
        let mut matrix = vec![vec![0.0; cols]; rows];
        let scale = (2.0 / cols as f32).sqrt();

        for i in 0..rows {
            for j in 0..cols {
                let r = ((i * cols + j) as f32 * 0.123456).fract();
                matrix[i][j] = (r - 0.5) * 2.0 * scale;
            }
        }
        matrix
    }

    /// Forward pass: x → ReLU(W1·x) → W2
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        // W1·x + ReLU
        let mut hidden = vec![0.0; self.d_ff];
        for i in 0..self.d_ff {
            for j in 0..x.len().min(self.W1[0].len()) {
                hidden[i] += self.W1[i][j] * x[j];
            }
            hidden[i] = hidden[i].max(0.0); // ReLU
        }

        // W2·hidden
        let mut output = vec![0.0; self.d_model];
        for i in 0..self.d_model {
            for j in 0..hidden.len().min(self.W2[0].len()) {
                output[i] += self.W2[i][j] * hidden[j];
            }
        }

        output
    }
}

/// Routing strategy
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RoutingStrategy {
    /// Top-K routing (select K experts with highest scores)
    TopK,
    /// Switch routing (top-1, used in Switch Transformers)
    Switch,
    /// Expert Choice (experts choose top-K tokens)
    ExpertChoice,
}

/// Load balancing loss types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LoadBalancingLoss {
    /// Auxiliary loss to encourage balanced expert usage
    Auxiliary,
    /// Expert capacity (hard limit on tokens per expert)
    Capacity,
    /// None (no load balancing)
    None,
}

/// Router: Decides which experts to use for each input
#[derive(Debug, Clone)]
pub struct Router {
    pub d_model: usize,
    pub num_experts: usize,
    pub top_k: usize,
    /// Router weights (num_experts × d_model)
    pub W_router: Vec<Vec<f32>>,
    /// Routing strategy
    pub strategy: RoutingStrategy,
    /// Noise for training (exploration)
    pub noise_std: f32,
}

impl Router {
    /// Create new router
    pub fn new(d_model: usize, num_experts: usize, top_k: usize) -> Self {
        assert!(top_k <= num_experts, "top_k must be <= num_experts");

        let W_router = Self::init_matrix(num_experts, d_model);

        Self {
            d_model,
            num_experts,
            top_k,
            W_router,
            strategy: RoutingStrategy::TopK,
            noise_std: 0.0,
        }
    }

    fn init_matrix(rows: usize, cols: usize) -> Vec<Vec<f32>> {
        let mut matrix = vec![vec![0.0; cols]; rows];
        let scale = (1.0 / cols as f32).sqrt();

        for i in 0..rows {
            for j in 0..cols {
                let r = ((i * cols + j) as f32 * 0.314159).fract();
                matrix[i][j] = (r - 0.5) * 2.0 * scale;
            }
        }
        matrix
    }

    /// With routing strategy
    pub fn with_strategy(mut self, strategy: RoutingStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// With noise for exploration during training
    pub fn with_noise(mut self, noise_std: f32) -> Self {
        self.noise_std = noise_std;
        self
    }

    /// Compute routing scores for input
    ///
    /// scores = softmax(W_router · x)
    pub fn route(&self, x: &[f32]) -> Vec<(usize, f32)> {
        // Compute logits: W_router · x
        let mut logits = vec![0.0; self.num_experts];
        for i in 0..self.num_experts {
            for j in 0..x.len().min(self.W_router[0].len()) {
                logits[i] += self.W_router[i][j] * x[j];
            }
        }

        // Add noise during training (optional)
        if self.noise_std > 0.0 {
            for logit in logits.iter_mut() {
                // Simple pseudo-random noise
                let noise = (*logit * 0.123).fract() * self.noise_std;
                *logit += noise;
            }
        }

        // Softmax
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_logits: Vec<f32> = logits.iter().map(|&l| (l - max_logit).exp()).collect();
        let sum_exp: f32 = exp_logits.iter().sum();
        let probs: Vec<f32> = exp_logits.iter().map(|&e| e / sum_exp).collect();

        // Select top-K experts
        let mut expert_probs: Vec<(usize, f32)> =
            probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();

        // Sort by probability (descending)
        expert_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Take top-K
        let k = match self.strategy {
            RoutingStrategy::Switch => 1,
            _ => self.top_k,
        };

        expert_probs.truncate(k);

        // Renormalize (so selected expert weights sum to 1)
        let sum_selected: f32 = expert_probs.iter().map(|(_, p)| p).sum();
        for (_, p) in expert_probs.iter_mut() {
            *p /= sum_selected;
        }

        expert_probs
    }

    /// Compute load balancing auxiliary loss
    ///
    /// Encourages balanced expert usage
    pub fn compute_load_balancing_loss(&self, all_routes: &[Vec<(usize, f32)>]) -> f32 {
        let n = all_routes.len() as f32;

        // Count how many times each expert is selected
        let mut expert_counts = vec![0.0; self.num_experts];
        for routes in all_routes {
            for &(expert_id, _) in routes {
                expert_counts[expert_id] += 1.0;
            }
        }

        // Ideal: each expert selected n/num_experts times
        let ideal_count = n / self.num_experts as f32;

        // MSE from ideal distribution
        let mut loss = 0.0;
        for count in expert_counts {
            let diff = count - ideal_count;
            loss += diff * diff;
        }

        loss / self.num_experts as f32
    }
}

/// Mixture of Experts layer
#[derive(Debug, Clone)]
pub struct MoELayer {
    pub d_model: usize,
    pub num_experts: usize,
    pub top_k: usize,
    /// Router
    pub router: Router,
    /// Expert networks
    pub experts: Vec<Expert>,
    /// Load balancing
    pub load_balancing: LoadBalancingLoss,
    pub load_balancing_weight: f32,
}

impl MoELayer {
    /// Create new MoE layer
    pub fn new(d_model: usize, d_ff: usize, num_experts: usize, top_k: usize) -> Self {
        let router = Router::new(d_model, num_experts, top_k);

        let mut experts = Vec::with_capacity(num_experts);
        for _ in 0..num_experts {
            experts.push(Expert::new(d_model, d_ff));
        }

        Self {
            d_model,
            num_experts,
            top_k,
            router,
            experts,
            load_balancing: LoadBalancingLoss::Auxiliary,
            load_balancing_weight: 0.01,
        }
    }

    /// With routing strategy
    pub fn with_strategy(mut self, strategy: RoutingStrategy) -> Self {
        self.router = self.router.with_strategy(strategy);
        self
    }

    /// With load balancing
    pub fn with_load_balancing(mut self, lb_type: LoadBalancingLoss, weight: f32) -> Self {
        self.load_balancing = lb_type;
        self.load_balancing_weight = weight;
        self
    }

    /// Forward pass through MoE layer
    ///
    /// output = Σ_i g_i · Expert_i(x)
    /// where g_i is the routing weight for expert i
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        // Route to top-K experts
        let routes = self.router.route(x);

        // Compute weighted sum of expert outputs
        let mut output = vec![0.0; self.d_model];

        for (expert_id, weight) in routes {
            let expert_output = self.experts[expert_id].forward(x);

            for i in 0..output.len() {
                output[i] += weight * expert_output[i];
            }
        }

        output
    }

    /// Forward pass on batch
    pub fn forward_batch(&self, inputs: &[Vec<f32>]) -> Vec<Vec<f32>> {
        inputs.iter().map(|x| self.forward(x)).collect()
    }

    /// Get routing statistics (for analysis)
    pub fn get_routing_stats(&self, inputs: &[Vec<f32>]) -> HashMap<usize, usize> {
        let mut expert_usage = HashMap::new();

        for input in inputs {
            let routes = self.router.route(input);
            for (expert_id, _) in routes {
                *expert_usage.entry(expert_id).or_insert(0) += 1;
            }
        }

        expert_usage
    }

    /// Compute total loss (includes load balancing)
    pub fn compute_loss_with_balancing(&self, inputs: &[Vec<f32>], base_loss: f32) -> f32 {
        match self.load_balancing {
            LoadBalancingLoss::Auxiliary => {
                // Collect all routes
                let all_routes: Vec<_> = inputs.iter().map(|x| self.router.route(x)).collect();

                let lb_loss = self.router.compute_load_balancing_loss(&all_routes);
                base_loss + self.load_balancing_weight * lb_loss
            }
            _ => base_loss,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expert_creation() {
        let expert = Expert::new(64, 256);
        assert_eq!(expert.d_model, 64);
        assert_eq!(expert.d_ff, 256);
        assert_eq!(expert.W1.len(), 256);
        assert_eq!(expert.W1[0].len(), 64);
    }

    #[test]
    fn test_expert_forward() {
        let expert = Expert::new(32, 128);
        let input = vec![0.5; 32];
        let output = expert.forward(&input);

        assert_eq!(output.len(), 32);
    }

    #[test]
    fn test_router_creation() {
        let router = Router::new(64, 8, 2);
        assert_eq!(router.d_model, 64);
        assert_eq!(router.num_experts, 8);
        assert_eq!(router.top_k, 2);
        assert_eq!(router.W_router.len(), 8);
    }

    #[test]
    fn test_router_route() {
        let router = Router::new(32, 4, 2);
        let input = vec![0.5; 32];

        let routes = router.route(&input);

        // Should select top-2 experts
        assert_eq!(routes.len(), 2);

        // Weights should sum to ~1
        let sum_weights: f32 = routes.iter().map(|(_, w)| w).sum();
        assert!((sum_weights - 1.0).abs() < 1e-5);

        // Expert IDs should be valid
        for (expert_id, _) in routes {
            assert!(expert_id < 4);
        }
    }

    #[test]
    fn test_router_switch_strategy() {
        let router = Router::new(32, 4, 2).with_strategy(RoutingStrategy::Switch);
        let input = vec![0.5; 32];

        let routes = router.route(&input);

        // Switch routing = top-1
        assert_eq!(routes.len(), 1);
    }

    #[test]
    fn test_router_with_noise() {
        let router = Router::new(32, 4, 2).with_noise(0.1);
        assert_eq!(router.noise_std, 0.1);
    }

    #[test]
    fn test_load_balancing_loss() {
        let router = Router::new(32, 4, 1);

        // Unbalanced: all routes go to expert 0
        let all_routes = vec![
            vec![(0, 1.0)],
            vec![(0, 1.0)],
            vec![(0, 1.0)],
            vec![(0, 1.0)],
        ];

        let loss = router.compute_load_balancing_loss(&all_routes);

        // Should have high loss (unbalanced)
        assert!(loss > 0.0);

        // Balanced: routes distributed evenly
        let balanced_routes = vec![
            vec![(0, 1.0)],
            vec![(1, 1.0)],
            vec![(2, 1.0)],
            vec![(3, 1.0)],
        ];

        let balanced_loss = router.compute_load_balancing_loss(&balanced_routes);

        // Should have lower loss
        assert!(balanced_loss < loss);
    }

    #[test]
    fn test_moe_layer_creation() {
        let moe = MoELayer::new(64, 256, 8, 2);

        assert_eq!(moe.d_model, 64);
        assert_eq!(moe.num_experts, 8);
        assert_eq!(moe.top_k, 2);
        assert_eq!(moe.experts.len(), 8);
    }

    #[test]
    fn test_moe_forward() {
        let moe = MoELayer::new(32, 128, 4, 2);
        let input = vec![0.5; 32];

        let output = moe.forward(&input);

        assert_eq!(output.len(), 32);
    }

    #[test]
    fn test_moe_forward_batch() {
        let moe = MoELayer::new(16, 64, 4, 2);
        let inputs = vec![vec![0.5; 16], vec![0.3; 16], vec![0.7; 16]];

        let outputs = moe.forward_batch(&inputs);

        assert_eq!(outputs.len(), 3);
        assert_eq!(outputs[0].len(), 16);
    }

    #[test]
    fn test_moe_routing_stats() {
        let moe = MoELayer::new(16, 64, 4, 1);
        let inputs = vec![vec![0.5; 16]; 10];

        let stats = moe.get_routing_stats(&inputs);

        // Should have routed to experts
        assert!(stats.len() > 0);

        // Total routes should equal number of inputs
        let total_routes: usize = stats.values().sum();
        assert_eq!(total_routes, 10);
    }

    #[test]
    fn test_moe_with_load_balancing() {
        let moe =
            MoELayer::new(16, 64, 4, 1).with_load_balancing(LoadBalancingLoss::Auxiliary, 0.01);

        assert_eq!(moe.load_balancing, LoadBalancingLoss::Auxiliary);
        assert_eq!(moe.load_balancing_weight, 0.01);
    }

    #[test]
    fn test_moe_loss_with_balancing() {
        let moe =
            MoELayer::new(16, 64, 4, 1).with_load_balancing(LoadBalancingLoss::Auxiliary, 0.01);

        let inputs = vec![vec![0.5; 16]; 8];
        let base_loss = 1.0;

        let total_loss = moe.compute_loss_with_balancing(&inputs, base_loss);

        // Should include load balancing loss
        assert!(total_loss >= base_loss);
    }

    #[test]
    fn test_sparse_activation() {
        let moe = MoELayer::new(32, 128, 64, 2);
        let input = vec![0.5; 32];

        // Only 2 out of 64 experts are activated
        let routes = moe.router.route(&input);
        assert_eq!(routes.len(), 2);

        // 2/64 = 3.125% of experts active
        let activation_ratio = routes.len() as f32 / moe.num_experts as f32;
        assert!(activation_ratio < 0.05);
    }
}
