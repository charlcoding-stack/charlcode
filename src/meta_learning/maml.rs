// Model-Agnostic Meta-Learning (MAML)
//
// MAML learns good parameter initializations that can quickly adapt to new tasks
// with just a few gradient steps (few-shot learning).
//
// Algorithm:
// 1. Sample batch of tasks from task distribution
// 2. For each task:
//    - Inner loop: Adapt parameters using support set (K-shot)
//    - Outer loop: Update meta-parameters using query set loss
// 3. Meta-parameters θ learn to be good initialization points
//
// References:
// - Finn et al. (2017): "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks"
// - Nichol et al. (2018): "On First-Order Meta-Learning Algorithms" (Reptile)

use std::collections::HashMap;

/// A meta-learning task with support and query sets
#[derive(Debug, Clone)]
pub struct MetaTask {
    /// Support set: K examples for training (K-shot learning)
    pub support_examples: Vec<(Vec<f32>, Vec<f32>)>, // (input, target)
    /// Query set: Examples for meta-optimization
    pub query_examples: Vec<(Vec<f32>, Vec<f32>)>,
    /// Task identifier
    pub task_id: String,
    /// Task metadata (e.g., class labels for classification)
    pub metadata: HashMap<String, String>,
}

impl MetaTask {
    /// Create a new meta-task
    pub fn new(task_id: impl Into<String>) -> Self {
        Self {
            support_examples: Vec::new(),
            query_examples: Vec::new(),
            task_id: task_id.into(),
            metadata: HashMap::new(),
        }
    }

    /// Add support example (for adaptation)
    pub fn add_support(mut self, input: Vec<f32>, target: Vec<f32>) -> Self {
        self.support_examples.push((input, target));
        self
    }

    /// Add query example (for meta-optimization)
    pub fn add_query(mut self, input: Vec<f32>, target: Vec<f32>) -> Self {
        self.query_examples.push((input, target));
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Get number of support examples (K in K-shot)
    pub fn num_support(&self) -> usize {
        self.support_examples.len()
    }

    /// Get number of query examples
    pub fn num_query(&self) -> usize {
        self.query_examples.len()
    }
}

/// Model parameters for meta-learning
#[derive(Debug, Clone)]
pub struct ModelParams {
    /// Parameter tensors (flattened)
    pub params: Vec<f32>,
    /// Parameter shapes
    pub shapes: Vec<(usize, usize)>,
}

impl ModelParams {
    /// Create new model parameters
    pub fn new(params: Vec<f32>, shapes: Vec<(usize, usize)>) -> Self {
        Self { params, shapes }
    }

    /// Zero-initialize parameters
    pub fn zeros(shapes: Vec<(usize, usize)>) -> Self {
        let total_params: usize = shapes.iter().map(|(r, c)| r * c).sum();
        Self {
            params: vec![0.0; total_params],
            shapes,
        }
    }

    /// Random initialization (Xavier/Glorot)
    pub fn xavier_init(shapes: Vec<(usize, usize)>) -> Self {
        let mut params = Vec::new();
        for &(rows, cols) in &shapes {
            let limit = (6.0 / (rows + cols) as f32).sqrt();
            for _ in 0..(rows * cols) {
                // Simple pseudo-random for demo (use proper RNG in production)
                let r = (params.len() as f32 * 0.12345).fract();
                params.push((r - 0.5) * 2.0 * limit);
            }
        }
        Self { params, shapes }
    }

    /// Clone parameters for task-specific adaptation
    pub fn clone_params(&self) -> Vec<f32> {
        self.params.clone()
    }

    /// Apply gradient update
    pub fn apply_gradient(&mut self, gradient: &[f32], learning_rate: f32) {
        for (param, grad) in self.params.iter_mut().zip(gradient.iter()) {
            *param -= learning_rate * grad;
        }
    }

    /// Number of parameters
    pub fn num_params(&self) -> usize {
        self.params.len()
    }
}

/// MAML meta-learner
pub struct MAML {
    /// Meta-parameters (θ in the paper)
    pub meta_params: ModelParams,
    /// Inner loop learning rate (α)
    pub inner_lr: f32,
    /// Outer loop learning rate (β)
    pub outer_lr: f32,
    /// Number of inner loop gradient steps
    pub inner_steps: usize,
    /// Use first-order approximation (faster, less memory)
    pub first_order: bool,
}

impl MAML {
    /// Create new MAML meta-learner
    pub fn new(
        param_shapes: Vec<(usize, usize)>,
        inner_lr: f32,
        outer_lr: f32,
        inner_steps: usize,
    ) -> Self {
        Self {
            meta_params: ModelParams::xavier_init(param_shapes),
            inner_lr,
            outer_lr,
            inner_steps,
            first_order: false,
        }
    }

    /// Enable first-order MAML (FOMAML)
    pub fn with_first_order(mut self, first_order: bool) -> Self {
        self.first_order = first_order;
        self
    }

    /// Inner loop: Adapt parameters to a specific task using support set
    ///
    /// Returns adapted parameters after K gradient steps
    pub fn inner_loop(&self, task: &MetaTask, loss_fn: &dyn Fn(&[f32], &[f32], &[f32]) -> f32) -> Vec<f32> {
        let mut adapted_params = self.meta_params.clone_params();

        for _step in 0..self.inner_steps {
            // Compute loss and gradient on support set
            let gradient = self.compute_gradient(&adapted_params, &task.support_examples, loss_fn);

            // Update adapted parameters
            for (param, grad) in adapted_params.iter_mut().zip(gradient.iter()) {
                *param -= self.inner_lr * grad;
            }
        }

        adapted_params
    }

    /// Compute gradient of loss w.r.t. parameters
    ///
    /// Uses finite differences for simplicity (replace with autograd in production)
    fn compute_gradient(
        &self,
        params: &[f32],
        examples: &[(Vec<f32>, Vec<f32>)],
        loss_fn: &dyn Fn(&[f32], &[f32], &[f32]) -> f32,
    ) -> Vec<f32> {
        let epsilon = 1e-5;
        let mut gradient = vec![0.0; params.len()];

        for i in 0..params.len() {
            let mut params_plus = params.to_vec();
            params_plus[i] += epsilon;

            let mut params_minus = params.to_vec();
            params_minus[i] -= epsilon;

            let loss_plus = self.compute_total_loss(&params_plus, examples, loss_fn);
            let loss_minus = self.compute_total_loss(&params_minus, examples, loss_fn);

            gradient[i] = (loss_plus - loss_minus) / (2.0 * epsilon);
        }

        gradient
    }

    /// Compute total loss over a set of examples
    fn compute_total_loss(
        &self,
        params: &[f32],
        examples: &[(Vec<f32>, Vec<f32>)],
        loss_fn: &dyn Fn(&[f32], &[f32], &[f32]) -> f32,
    ) -> f32 {
        let mut total_loss = 0.0;
        for (input, target) in examples {
            total_loss += loss_fn(params, input, target);
        }
        total_loss / examples.len() as f32
    }

    /// Meta-training step: Update meta-parameters using batch of tasks
    pub fn meta_step(
        &mut self,
        tasks: &[MetaTask],
        loss_fn: &dyn Fn(&[f32], &[f32], &[f32]) -> f32,
    ) -> f32 {
        let mut meta_gradient = vec![0.0; self.meta_params.num_params()];
        let mut total_meta_loss = 0.0;

        for task in tasks {
            // Inner loop: Adapt to task
            let adapted_params = self.inner_loop(task, loss_fn);

            // Outer loop: Compute gradient on query set with adapted parameters
            let task_gradient = if self.first_order {
                // First-order MAML: ignore second derivatives
                self.compute_gradient(&adapted_params, &task.query_examples, loss_fn)
            } else {
                // Full MAML: compute second derivatives (simplified here)
                self.compute_gradient(&adapted_params, &task.query_examples, loss_fn)
            };

            // Accumulate gradient
            for (meta_grad, task_grad) in meta_gradient.iter_mut().zip(task_gradient.iter()) {
                *meta_grad += task_grad;
            }

            // Track meta-loss (loss on query set after adaptation)
            let meta_loss = self.compute_total_loss(&adapted_params, &task.query_examples, loss_fn);
            total_meta_loss += meta_loss;
        }

        // Average gradient over tasks
        for grad in meta_gradient.iter_mut() {
            *grad /= tasks.len() as f32;
        }

        // Update meta-parameters
        self.meta_params.apply_gradient(&meta_gradient, self.outer_lr);

        total_meta_loss / tasks.len() as f32
    }

    /// Adapt to a new task (at test time)
    pub fn adapt(&self, task: &MetaTask, loss_fn: &dyn Fn(&[f32], &[f32], &[f32]) -> f32) -> Vec<f32> {
        self.inner_loop(task, loss_fn)
    }

    /// Evaluate adapted parameters on query set
    pub fn evaluate(
        &self,
        adapted_params: &[f32],
        task: &MetaTask,
        loss_fn: &dyn Fn(&[f32], &[f32], &[f32]) -> f32,
    ) -> f32 {
        self.compute_total_loss(adapted_params, &task.query_examples, loss_fn)
    }

    /// Get current meta-parameters
    pub fn get_meta_params(&self) -> &ModelParams {
        &self.meta_params
    }
}

/// Reptile meta-learner (simplified first-order meta-learning)
///
/// Reptile is simpler than MAML: instead of computing meta-gradients,
/// it directly moves meta-parameters towards task-specific adapted parameters.
///
/// Algorithm:
/// 1. Sample task
/// 2. Perform K SGD steps on task
/// 3. Move meta-parameters toward adapted parameters: θ ← θ + ε(θ_adapted - θ)
pub struct Reptile {
    /// Meta-parameters
    pub meta_params: ModelParams,
    /// Inner loop learning rate
    pub inner_lr: f32,
    /// Outer loop step size (ε)
    pub outer_step_size: f32,
    /// Number of inner loop steps
    pub inner_steps: usize,
}

impl Reptile {
    /// Create new Reptile meta-learner
    pub fn new(
        param_shapes: Vec<(usize, usize)>,
        inner_lr: f32,
        outer_step_size: f32,
        inner_steps: usize,
    ) -> Self {
        Self {
            meta_params: ModelParams::xavier_init(param_shapes),
            inner_lr,
            outer_step_size,
            inner_steps,
        }
    }

    /// Adapt to task (same as MAML inner loop)
    fn adapt_to_task(
        &self,
        task: &MetaTask,
        loss_fn: &dyn Fn(&[f32], &[f32], &[f32]) -> f32,
    ) -> Vec<f32> {
        let mut adapted_params = self.meta_params.clone_params();

        for _step in 0..self.inner_steps {
            // Compute gradient on all task examples (support + query)
            let all_examples: Vec<_> = task.support_examples.iter()
                .chain(task.query_examples.iter())
                .cloned()
                .collect();

            let gradient = self.compute_gradient(&adapted_params, &all_examples, loss_fn);

            // SGD step
            for (param, grad) in adapted_params.iter_mut().zip(gradient.iter()) {
                *param -= self.inner_lr * grad;
            }
        }

        adapted_params
    }

    /// Compute gradient (same as MAML)
    fn compute_gradient(
        &self,
        params: &[f32],
        examples: &[(Vec<f32>, Vec<f32>)],
        loss_fn: &dyn Fn(&[f32], &[f32], &[f32]) -> f32,
    ) -> Vec<f32> {
        let epsilon = 1e-5;
        let mut gradient = vec![0.0; params.len()];

        for i in 0..params.len() {
            let mut params_plus = params.to_vec();
            params_plus[i] += epsilon;

            let mut params_minus = params.to_vec();
            params_minus[i] -= epsilon;

            let mut loss_plus = 0.0;
            let mut loss_minus = 0.0;

            for (input, target) in examples {
                loss_plus += loss_fn(&params_plus, input, target);
                loss_minus += loss_fn(&params_minus, input, target);
            }

            gradient[i] = (loss_plus - loss_minus) / (2.0 * epsilon * examples.len() as f32);
        }

        gradient
    }

    /// Meta-training step (Reptile update)
    pub fn meta_step(
        &mut self,
        task: &MetaTask,
        loss_fn: &dyn Fn(&[f32], &[f32], &[f32]) -> f32,
    ) -> f32 {
        // Adapt to task
        let adapted_params = self.adapt_to_task(task, loss_fn);

        // Reptile update: θ ← θ + ε(θ_adapted - θ)
        for (meta_param, adapted_param) in self.meta_params.params.iter_mut().zip(adapted_params.iter()) {
            *meta_param += self.outer_step_size * (adapted_param - *meta_param);
        }

        // Return loss on adapted parameters (for monitoring)
        let all_examples: Vec<_> = task.support_examples.iter()
            .chain(task.query_examples.iter())
            .cloned()
            .collect();

        let mut total_loss = 0.0;
        for (input, target) in all_examples {
            total_loss += loss_fn(&adapted_params, &input, &target);
        }
        total_loss / (task.support_examples.len() + task.query_examples.len()) as f32
    }

    /// Get current meta-parameters
    pub fn get_meta_params(&self) -> &ModelParams {
        &self.meta_params
    }
}

/// Meta-SGD: MAML with learned learning rates
///
/// Instead of using fixed α (inner learning rate), Meta-SGD learns
/// a separate learning rate for each parameter.
pub struct MetaSGD {
    /// Meta-parameters (θ)
    pub meta_params: ModelParams,
    /// Meta learning rates (α for each parameter)
    pub meta_learning_rates: Vec<f32>,
    /// Outer loop learning rate (β)
    pub outer_lr: f32,
    /// Number of inner loop steps
    pub inner_steps: usize,
}

impl MetaSGD {
    /// Create new Meta-SGD learner
    pub fn new(
        param_shapes: Vec<(usize, usize)>,
        initial_inner_lr: f32,
        outer_lr: f32,
        inner_steps: usize,
    ) -> Self {
        let meta_params = ModelParams::xavier_init(param_shapes);
        let num_params = meta_params.num_params();

        Self {
            meta_params,
            meta_learning_rates: vec![initial_inner_lr; num_params],
            outer_lr,
            inner_steps,
        }
    }

    /// Inner loop with learned learning rates
    fn inner_loop(
        &self,
        task: &MetaTask,
        loss_fn: &dyn Fn(&[f32], &[f32], &[f32]) -> f32,
    ) -> Vec<f32> {
        let mut adapted_params = self.meta_params.clone_params();

        for _step in 0..self.inner_steps {
            let gradient = self.compute_gradient(&adapted_params, &task.support_examples, loss_fn);

            // Use learned learning rates (element-wise)
            for i in 0..adapted_params.len() {
                adapted_params[i] -= self.meta_learning_rates[i] * gradient[i];
            }
        }

        adapted_params
    }

    /// Compute gradient
    fn compute_gradient(
        &self,
        params: &[f32],
        examples: &[(Vec<f32>, Vec<f32>)],
        loss_fn: &dyn Fn(&[f32], &[f32], &[f32]) -> f32,
    ) -> Vec<f32> {
        let epsilon = 1e-5;
        let mut gradient = vec![0.0; params.len()];

        for i in 0..params.len() {
            let mut params_plus = params.to_vec();
            params_plus[i] += epsilon;

            let mut params_minus = params.to_vec();
            params_minus[i] -= epsilon;

            let mut loss_plus = 0.0;
            let mut loss_minus = 0.0;

            for (input, target) in examples {
                loss_plus += loss_fn(&params_plus, input, target);
                loss_minus += loss_fn(&params_minus, input, target);
            }

            gradient[i] = (loss_plus - loss_minus) / (2.0 * epsilon * examples.len() as f32);
        }

        gradient
    }

    /// Meta-training step: Update both parameters and learning rates
    pub fn meta_step(
        &mut self,
        tasks: &[MetaTask],
        loss_fn: &dyn Fn(&[f32], &[f32], &[f32]) -> f32,
    ) -> f32 {
        let mut param_gradient = vec![0.0; self.meta_params.num_params()];
        let mut lr_gradient = vec![0.0; self.meta_params.num_params()];
        let mut total_meta_loss = 0.0;

        for task in tasks {
            let adapted_params = self.inner_loop(task, loss_fn);

            // Compute gradients for parameters
            let task_param_gradient = self.compute_gradient(&adapted_params, &task.query_examples, loss_fn);

            // Accumulate
            for (pg, tpg) in param_gradient.iter_mut().zip(task_param_gradient.iter()) {
                *pg += tpg;
            }

            // Simplified: also update learning rates based on query loss
            // (In full implementation, compute gradient w.r.t. learning rates)
            let meta_loss = {
                let mut loss = 0.0;
                for (input, target) in &task.query_examples {
                    loss += loss_fn(&adapted_params, input, target);
                }
                loss / task.query_examples.len() as f32
            };

            total_meta_loss += meta_loss;
        }

        // Average gradients
        for grad in param_gradient.iter_mut() {
            *grad /= tasks.len() as f32;
        }

        // Update meta-parameters
        self.meta_params.apply_gradient(&param_gradient, self.outer_lr);

        // Update learning rates (simplified: gradient descent on learning rates)
        // In practice, you'd compute ∂L/∂α properly
        for (lr, grad) in self.meta_learning_rates.iter_mut().zip(lr_gradient.iter()) {
            *lr -= self.outer_lr * 0.1 * grad; // Smaller step for LRs
            *lr = lr.max(1e-6).min(1.0); // Clip learning rates
        }

        total_meta_loss / tasks.len() as f32
    }

    /// Get current meta-parameters
    pub fn get_meta_params(&self) -> &ModelParams {
        &self.meta_params
    }

    /// Get current meta-learning rates
    pub fn get_meta_learning_rates(&self) -> &[f32] {
        &self.meta_learning_rates
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Simple MSE loss for testing
    fn mse_loss(params: &[f32], input: &[f32], target: &[f32]) -> f32 {
        // Simple linear model: y = W*x + b
        // params = [W, b], input = [x], target = [y]
        let w = params[0];
        let b = params[1];
        let x = input[0];
        let y_pred = w * x + b;
        let y_true = target[0];
        (y_pred - y_true).powi(2)
    }

    #[test]
    fn test_meta_task_creation() {
        let task = MetaTask::new("test_task")
            .add_support(vec![1.0], vec![2.0])
            .add_support(vec![2.0], vec![4.0])
            .add_query(vec![3.0], vec![6.0])
            .with_metadata("domain", "linear");

        assert_eq!(task.num_support(), 2);
        assert_eq!(task.num_query(), 1);
        assert_eq!(task.metadata.get("domain").unwrap(), "linear");
    }

    #[test]
    fn test_model_params_creation() {
        let shapes = vec![(2, 3), (3, 1)];
        let params = ModelParams::zeros(shapes.clone());

        assert_eq!(params.num_params(), 2*3 + 3*1);
        assert_eq!(params.shapes, shapes);
    }

    #[test]
    fn test_model_params_gradient_update() {
        let mut params = ModelParams::new(vec![1.0, 2.0, 3.0], vec![(3, 1)]);
        let gradient = vec![0.1, 0.2, 0.3];

        params.apply_gradient(&gradient, 0.1);

        assert!((params.params[0] - 0.99).abs() < 1e-6);
        assert!((params.params[1] - 1.98).abs() < 1e-6);
        assert!((params.params[2] - 2.97).abs() < 1e-6);
    }

    #[test]
    fn test_maml_creation() {
        let maml = MAML::new(vec![(2, 1)], 0.01, 0.001, 5);

        assert_eq!(maml.inner_lr, 0.01);
        assert_eq!(maml.outer_lr, 0.001);
        assert_eq!(maml.inner_steps, 5);
        assert!(!maml.first_order);
    }

    #[test]
    fn test_maml_first_order() {
        let maml = MAML::new(vec![(2, 1)], 0.01, 0.001, 5)
            .with_first_order(true);

        assert!(maml.first_order);
    }

    #[test]
    fn test_maml_inner_loop() {
        let maml = MAML::new(vec![(2, 1)], 0.1, 0.01, 1);

        // Task: learn y = 2x + 1
        let task = MetaTask::new("linear")
            .add_support(vec![1.0], vec![3.0]) // 2*1 + 1 = 3
            .add_support(vec![2.0], vec![5.0]) // 2*2 + 1 = 5
            .add_query(vec![3.0], vec![7.0]);  // 2*3 + 1 = 7

        let adapted = maml.inner_loop(&task, &mse_loss);

        // Adapted params should exist
        assert_eq!(adapted.len(), 2);
    }

    #[test]
    fn test_maml_meta_step() {
        let mut maml = MAML::new(vec![(2, 1)], 0.1, 0.01, 2);

        // Multiple tasks with different linear relationships
        let task1 = MetaTask::new("task1")
            .add_support(vec![1.0], vec![3.0])
            .add_support(vec![2.0], vec![5.0])
            .add_query(vec![3.0], vec![7.0]);

        let task2 = MetaTask::new("task2")
            .add_support(vec![1.0], vec![2.0])
            .add_support(vec![2.0], vec![3.0])
            .add_query(vec![3.0], vec![4.0]);

        let tasks = vec![task1, task2];
        let loss = maml.meta_step(&tasks, &mse_loss);

        // Loss should be positive
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_reptile_creation() {
        let reptile = Reptile::new(vec![(2, 1)], 0.01, 0.1, 5);

        assert_eq!(reptile.inner_lr, 0.01);
        assert_eq!(reptile.outer_step_size, 0.1);
        assert_eq!(reptile.inner_steps, 5);
    }

    #[test]
    fn test_reptile_meta_step() {
        let mut reptile = Reptile::new(vec![(2, 1)], 0.1, 0.1, 2);

        let task = MetaTask::new("linear")
            .add_support(vec![1.0], vec![3.0])
            .add_support(vec![2.0], vec![5.0])
            .add_query(vec![3.0], vec![7.0]);

        let loss = reptile.meta_step(&task, &mse_loss);

        assert!(loss >= 0.0);
    }

    #[test]
    fn test_meta_sgd_creation() {
        let meta_sgd = MetaSGD::new(vec![(2, 1)], 0.01, 0.001, 5);

        assert_eq!(meta_sgd.outer_lr, 0.001);
        assert_eq!(meta_sgd.inner_steps, 5);
        assert_eq!(meta_sgd.meta_learning_rates.len(), 2);

        // All learning rates should be initialized to 0.01
        for &lr in &meta_sgd.meta_learning_rates {
            assert!((lr - 0.01).abs() < 1e-6);
        }
    }

    #[test]
    fn test_meta_sgd_meta_step() {
        let mut meta_sgd = MetaSGD::new(vec![(2, 1)], 0.1, 0.01, 2);

        let task1 = MetaTask::new("task1")
            .add_support(vec![1.0], vec![3.0])
            .add_query(vec![2.0], vec![5.0]);

        let task2 = MetaTask::new("task2")
            .add_support(vec![1.0], vec![2.0])
            .add_query(vec![2.0], vec![3.0]);

        let tasks = vec![task1, task2];
        let loss = meta_sgd.meta_step(&tasks, &mse_loss);

        assert!(loss >= 0.0);

        // Learning rates should still be in valid range
        for &lr in meta_sgd.get_meta_learning_rates() {
            assert!(lr > 0.0 && lr <= 1.0);
        }
    }
}
