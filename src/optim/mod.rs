// Optimization Module
// Phase 6: Optimizers, training loops, and metrics

use crate::autograd::Tensor;
use std::collections::HashMap;

// Optimizer trait - all optimizers must implement this
pub trait Optimizer {
    fn step(&mut self, parameters: &mut [&mut Tensor]);
    fn zero_grad(&mut self, parameters: &mut [&mut Tensor]);
    fn get_lr(&self) -> f64;
    fn set_lr(&mut self, lr: f64);
}

// SGD (Stochastic Gradient Descent) with optional momentum
pub struct SGD {
    pub lr: f64,
    pub momentum: f64,
    pub weight_decay: f64,
    velocity: HashMap<usize, Vec<f64>>,
}

impl SGD {
    pub fn new(lr: f64) -> Self {
        SGD {
            lr,
            momentum: 0.0,
            weight_decay: 0.0,
            velocity: HashMap::new(),
        }
    }

    pub fn with_momentum(mut self, momentum: f64) -> Self {
        self.momentum = momentum;
        self
    }

    pub fn with_weight_decay(mut self, weight_decay: f64) -> Self {
        self.weight_decay = weight_decay;
        self
    }
}

impl Optimizer for SGD {
    fn step(&mut self, parameters: &mut [&mut Tensor]) {
        for param in parameters.iter_mut() {
            if !param.requires_grad {
                continue;
            }

            if let Some(grad) = &param.grad {
                let param_id = param.id;

                // Initialize velocity if not exists
                if !self.velocity.contains_key(&param_id) {
                    self.velocity.insert(param_id, vec![0.0; param.data.len()]);
                }

                let velocity = self.velocity.get_mut(&param_id).unwrap();

                for i in 0..param.data.len() {
                    // Add weight decay (L2 regularization)
                    let mut g = grad[i];
                    if self.weight_decay > 0.0 {
                        g += self.weight_decay * param.data[i];
                    }

                    // Momentum update
                    if self.momentum > 0.0 {
                        velocity[i] = self.momentum * velocity[i] + g;
                        param.data[i] -= self.lr * velocity[i];
                    } else {
                        param.data[i] -= self.lr * g;
                    }
                }
            }
        }
    }

    fn zero_grad(&mut self, parameters: &mut [&mut Tensor]) {
        for param in parameters.iter_mut() {
            if let Some(ref mut grad) = param.grad {
                for g in grad.iter_mut() {
                    *g = 0.0;
                }
            }
        }
    }

    fn get_lr(&self) -> f64 {
        self.lr
    }

    fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
    }
}

// Adam (Adaptive Moment Estimation)
pub struct Adam {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    pub weight_decay: f64,
    m: HashMap<usize, Vec<f64>>, // First moment
    v: HashMap<usize, Vec<f64>>, // Second moment
    t: usize, // Time step
}

impl Adam {
    pub fn new(lr: f64) -> Self {
        Adam {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0,
            m: HashMap::new(),
            v: HashMap::new(),
            t: 0,
        }
    }

    pub fn with_betas(mut self, beta1: f64, beta2: f64) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }

    pub fn with_weight_decay(mut self, weight_decay: f64) -> Self {
        self.weight_decay = weight_decay;
        self
    }
}

impl Optimizer for Adam {
    fn step(&mut self, parameters: &mut [&mut Tensor]) {
        self.t += 1;

        for param in parameters.iter_mut() {
            if !param.requires_grad {
                continue;
            }

            if let Some(grad) = &param.grad {
                let param_id = param.id;

                // Initialize moments if not exists
                if !self.m.contains_key(&param_id) {
                    self.m.insert(param_id, vec![0.0; param.data.len()]);
                    self.v.insert(param_id, vec![0.0; param.data.len()]);
                }

                let m = self.m.get_mut(&param_id).unwrap();
                let v = self.v.get_mut(&param_id).unwrap();

                for i in 0..param.data.len() {
                    // Add weight decay
                    let mut g = grad[i];
                    if self.weight_decay > 0.0 {
                        g += self.weight_decay * param.data[i];
                    }

                    // Update biased first moment estimate
                    m[i] = self.beta1 * m[i] + (1.0 - self.beta1) * g;

                    // Update biased second moment estimate
                    v[i] = self.beta2 * v[i] + (1.0 - self.beta2) * g * g;

                    // Bias correction
                    let m_hat = m[i] / (1.0 - self.beta1.powi(self.t as i32));
                    let v_hat = v[i] / (1.0 - self.beta2.powi(self.t as i32));

                    // Update parameters
                    param.data[i] -= self.lr * m_hat / (v_hat.sqrt() + self.epsilon);
                }
            }
        }
    }

    fn zero_grad(&mut self, parameters: &mut [&mut Tensor]) {
        for param in parameters.iter_mut() {
            if let Some(ref mut grad) = param.grad {
                for g in grad.iter_mut() {
                    *g = 0.0;
                }
            }
        }
    }

    fn get_lr(&self) -> f64 {
        self.lr
    }

    fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
    }
}

// RMSprop (Root Mean Square Propagation)
pub struct RMSprop {
    pub lr: f64,
    pub alpha: f64,
    pub epsilon: f64,
    pub weight_decay: f64,
    v: HashMap<usize, Vec<f64>>, // Moving average of squared gradients
}

impl RMSprop {
    pub fn new(lr: f64) -> Self {
        RMSprop {
            lr,
            alpha: 0.99,
            epsilon: 1e-8,
            weight_decay: 0.0,
            v: HashMap::new(),
        }
    }

    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    pub fn with_weight_decay(mut self, weight_decay: f64) -> Self {
        self.weight_decay = weight_decay;
        self
    }
}

impl Optimizer for RMSprop {
    fn step(&mut self, parameters: &mut [&mut Tensor]) {
        for param in parameters.iter_mut() {
            if !param.requires_grad {
                continue;
            }

            if let Some(grad) = &param.grad {
                let param_id = param.id;

                // Initialize moving average if not exists
                if !self.v.contains_key(&param_id) {
                    self.v.insert(param_id, vec![0.0; param.data.len()]);
                }

                let v = self.v.get_mut(&param_id).unwrap();

                for i in 0..param.data.len() {
                    // Add weight decay
                    let mut g = grad[i];
                    if self.weight_decay > 0.0 {
                        g += self.weight_decay * param.data[i];
                    }

                    // Update moving average of squared gradients
                    v[i] = self.alpha * v[i] + (1.0 - self.alpha) * g * g;

                    // Update parameters
                    param.data[i] -= self.lr * g / (v[i].sqrt() + self.epsilon);
                }
            }
        }
    }

    fn zero_grad(&mut self, parameters: &mut [&mut Tensor]) {
        for param in parameters.iter_mut() {
            if let Some(ref mut grad) = param.grad {
                for g in grad.iter_mut() {
                    *g = 0.0;
                }
            }
        }
    }

    fn get_lr(&self) -> f64 {
        self.lr
    }

    fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
    }
}

// AdaGrad (Adaptive Gradient)
pub struct AdaGrad {
    pub lr: f64,
    pub epsilon: f64,
    pub weight_decay: f64,
    sum_squared_grad: HashMap<usize, Vec<f64>>,
}

impl AdaGrad {
    pub fn new(lr: f64) -> Self {
        AdaGrad {
            lr,
            epsilon: 1e-10,
            weight_decay: 0.0,
            sum_squared_grad: HashMap::new(),
        }
    }

    pub fn with_weight_decay(mut self, weight_decay: f64) -> Self {
        self.weight_decay = weight_decay;
        self
    }
}

impl Optimizer for AdaGrad {
    fn step(&mut self, parameters: &mut [&mut Tensor]) {
        for param in parameters.iter_mut() {
            if !param.requires_grad {
                continue;
            }

            if let Some(grad) = &param.grad {
                let param_id = param.id;

                // Initialize sum if not exists
                if !self.sum_squared_grad.contains_key(&param_id) {
                    self.sum_squared_grad.insert(param_id, vec![0.0; param.data.len()]);
                }

                let sum_sq = self.sum_squared_grad.get_mut(&param_id).unwrap();

                for i in 0..param.data.len() {
                    // Add weight decay
                    let mut g = grad[i];
                    if self.weight_decay > 0.0 {
                        g += self.weight_decay * param.data[i];
                    }

                    // Accumulate squared gradients
                    sum_sq[i] += g * g;

                    // Update parameters
                    param.data[i] -= self.lr * g / (sum_sq[i].sqrt() + self.epsilon);
                }
            }
        }
    }

    fn zero_grad(&mut self, parameters: &mut [&mut Tensor]) {
        for param in parameters.iter_mut() {
            if let Some(ref mut grad) = param.grad {
                for g in grad.iter_mut() {
                    *g = 0.0;
                }
            }
        }
    }

    fn get_lr(&self) -> f64 {
        self.lr
    }

    fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
    }
}

// Learning Rate Scheduler trait
pub trait LRScheduler {
    fn step(&mut self, optimizer: &mut dyn Optimizer);
    fn get_lr(&self) -> f64;
}

// Step LR - decay by gamma every step_size epochs
pub struct StepLR {
    pub step_size: usize,
    pub gamma: f64,
    current_epoch: usize,
    base_lr: f64,
}

impl StepLR {
    pub fn new(base_lr: f64, step_size: usize, gamma: f64) -> Self {
        StepLR {
            step_size,
            gamma,
            current_epoch: 0,
            base_lr,
        }
    }
}

impl LRScheduler for StepLR {
    fn step(&mut self, optimizer: &mut dyn Optimizer) {
        self.current_epoch += 1;
        let num_decays = self.current_epoch / self.step_size;
        let new_lr = self.base_lr * self.gamma.powi(num_decays as i32);
        optimizer.set_lr(new_lr);
    }

    fn get_lr(&self) -> f64 {
        let num_decays = self.current_epoch / self.step_size;
        self.base_lr * self.gamma.powi(num_decays as i32)
    }
}

// Exponential LR - decay by gamma every epoch
pub struct ExponentialLR {
    pub gamma: f64,
    base_lr: f64,
    current_epoch: usize,
}

impl ExponentialLR {
    pub fn new(base_lr: f64, gamma: f64) -> Self {
        ExponentialLR {
            gamma,
            base_lr,
            current_epoch: 0,
        }
    }
}

impl LRScheduler for ExponentialLR {
    fn step(&mut self, optimizer: &mut dyn Optimizer) {
        self.current_epoch += 1;
        let new_lr = self.base_lr * self.gamma.powi(self.current_epoch as i32);
        optimizer.set_lr(new_lr);
    }

    fn get_lr(&self) -> f64 {
        self.base_lr * self.gamma.powi(self.current_epoch as i32)
    }
}

// Gradient clipping utilities
pub fn clip_grad_norm(parameters: &mut [&mut Tensor], max_norm: f64) -> f64 {
    let mut total_norm = 0.0;

    // Calculate total norm
    for param in parameters.iter() {
        if let Some(grad) = &param.grad {
            for &g in grad.iter() {
                total_norm += g * g;
            }
        }
    }

    total_norm = total_norm.sqrt();

    // Clip if needed
    if total_norm > max_norm {
        let clip_coef = max_norm / (total_norm + 1e-6);
        for param in parameters.iter_mut() {
            if let Some(grad) = &mut param.grad {
                for g in grad.iter_mut() {
                    *g *= clip_coef;
                }
            }
        }
    }

    total_norm
}

pub fn clip_grad_value(parameters: &mut [&mut Tensor], clip_value: f64) {
    for param in parameters.iter_mut() {
        if let Some(grad) = &mut param.grad {
            for g in grad.iter_mut() {
                *g = g.max(-clip_value).min(clip_value);
            }
        }
    }
}

// Metrics for evaluation
#[derive(Debug, Clone)]
pub struct Metrics {
    pub accuracy: f64,
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    pub loss: f64,
}

impl Metrics {
    pub fn new() -> Self {
        Metrics {
            accuracy: 0.0,
            precision: 0.0,
            recall: 0.0,
            f1_score: 0.0,
            loss: 0.0,
        }
    }

    pub fn compute_binary(predictions: &[f64], targets: &[f64], threshold: f64) -> Self {
        let mut tp = 0.0;
        let mut fp = 0.0;
        let mut tn = 0.0;
        let mut fn_count = 0.0;

        for (pred, target) in predictions.iter().zip(targets.iter()) {
            let pred_class = if *pred >= threshold { 1.0 } else { 0.0 };

            if pred_class == 1.0 && *target == 1.0 {
                tp += 1.0;
            } else if pred_class == 1.0 && *target == 0.0 {
                fp += 1.0;
            } else if pred_class == 0.0 && *target == 0.0 {
                tn += 1.0;
            } else {
                fn_count += 1.0;
            }
        }

        let accuracy = (tp + tn) / (tp + tn + fp + fn_count);
        let precision = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
        let recall = if tp + fn_count > 0.0 { tp / (tp + fn_count) } else { 0.0 };
        let f1_score = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };

        Metrics {
            accuracy,
            precision,
            recall,
            f1_score,
            loss: 0.0,
        }
    }

    pub fn compute_multiclass(predictions: &[Vec<f64>], targets: &[usize]) -> Self {
        let mut correct = 0;
        let total = predictions.len();

        for (pred, target) in predictions.iter().zip(targets.iter()) {
            let pred_class = pred
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap();

            if pred_class == *target {
                correct += 1;
            }
        }

        let accuracy = correct as f64 / total as f64;

        Metrics {
            accuracy,
            precision: 0.0, // TODO: Implement for multiclass
            recall: 0.0,
            f1_score: 0.0,
            loss: 0.0,
        }
    }
}

// Training History
#[derive(Debug, Clone)]
pub struct History {
    pub train_losses: Vec<f64>,
    pub val_losses: Vec<f64>,
    pub train_accuracies: Vec<f64>,
    pub val_accuracies: Vec<f64>,
}

impl History {
    pub fn new() -> Self {
        History {
            train_losses: Vec::new(),
            val_losses: Vec::new(),
            train_accuracies: Vec::new(),
            val_accuracies: Vec::new(),
        }
    }

    pub fn add_epoch(&mut self, train_loss: f64, val_loss: f64, train_acc: f64, val_acc: f64) {
        self.train_losses.push(train_loss);
        self.val_losses.push(val_loss);
        self.train_accuracies.push(train_acc);
        self.val_accuracies.push(val_acc);
    }

    pub fn best_val_loss(&self) -> Option<f64> {
        self.val_losses
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .copied()
    }

    pub fn best_val_accuracy(&self) -> Option<f64> {
        self.val_accuracies
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sgd_creation() {
        let sgd = SGD::new(0.01);
        assert_eq!(sgd.lr, 0.01);
        assert_eq!(sgd.momentum, 0.0);
    }

    #[test]
    fn test_sgd_with_momentum() {
        let sgd = SGD::new(0.01).with_momentum(0.9);
        assert_eq!(sgd.momentum, 0.9);
    }

    #[test]
    fn test_adam_creation() {
        let adam = Adam::new(0.001);
        assert_eq!(adam.lr, 0.001);
        assert_eq!(adam.beta1, 0.9);
        assert_eq!(adam.beta2, 0.999);
    }

    #[test]
    fn test_rmsprop_creation() {
        let rmsprop = RMSprop::new(0.01);
        assert_eq!(rmsprop.lr, 0.01);
        assert_eq!(rmsprop.alpha, 0.99);
    }

    #[test]
    fn test_adagrad_creation() {
        let adagrad = AdaGrad::new(0.01);
        assert_eq!(adagrad.lr, 0.01);
    }

    #[test]
    fn test_sgd_step() {
        let mut sgd = SGD::new(0.1);
        let mut tensor = Tensor::with_grad(vec![1.0, 2.0], vec![2]);
        tensor.grad = Some(vec![0.5, 0.5]);

        let mut params = vec![&mut tensor];
        sgd.step(&mut params);

        assert_eq!(tensor.data[0], 0.95); // 1.0 - 0.1 * 0.5
        assert_eq!(tensor.data[1], 1.95); // 2.0 - 0.1 * 0.5
    }

    #[test]
    fn test_sgd_zero_grad() {
        let mut sgd = SGD::new(0.1);
        let mut tensor = Tensor::with_grad(vec![1.0], vec![1]);
        tensor.grad = Some(vec![5.0]);

        let mut params = vec![&mut tensor];
        sgd.zero_grad(&mut params);

        assert_eq!(tensor.grad, Some(vec![0.0]));
    }

    #[test]
    fn test_step_lr_scheduler() {
        let base_lr = 0.1;
        let mut scheduler = StepLR::new(base_lr, 2, 0.5);
        let mut sgd = SGD::new(base_lr);

        assert_eq!(scheduler.get_lr(), 0.1);

        scheduler.step(&mut sgd);
        assert_eq!(scheduler.get_lr(), 0.1); // epoch 1, no decay yet

        scheduler.step(&mut sgd);
        assert_eq!(scheduler.get_lr(), 0.05); // epoch 2, first decay

        scheduler.step(&mut sgd);
        assert_eq!(scheduler.get_lr(), 0.05); // epoch 3, still 0.05

        scheduler.step(&mut sgd);
        assert_eq!(scheduler.get_lr(), 0.025); // epoch 4, second decay
    }

    #[test]
    fn test_exponential_lr_scheduler() {
        let base_lr = 0.1;
        let mut scheduler = ExponentialLR::new(base_lr, 0.9);
        let mut sgd = SGD::new(base_lr);

        assert_eq!(scheduler.get_lr(), 0.1);

        scheduler.step(&mut sgd);
        assert!((scheduler.get_lr() - 0.09).abs() < 1e-6);

        scheduler.step(&mut sgd);
        assert!((scheduler.get_lr() - 0.081).abs() < 1e-6);
    }

    #[test]
    fn test_clip_grad_value() {
        let mut tensor = Tensor::with_grad(vec![1.0, 2.0], vec![2]);
        tensor.grad = Some(vec![5.0, -10.0]);

        let mut params = vec![&mut tensor];
        clip_grad_value(&mut params, 3.0);

        assert_eq!(tensor.grad.as_ref().unwrap()[0], 3.0);
        assert_eq!(tensor.grad.as_ref().unwrap()[1], -3.0);
    }

    #[test]
    fn test_metrics_binary() {
        let predictions = vec![0.9, 0.2, 0.8, 0.3];
        let targets = vec![1.0, 0.0, 1.0, 0.0];

        let metrics = Metrics::compute_binary(&predictions, &targets, 0.5);

        assert_eq!(metrics.accuracy, 1.0);
        assert_eq!(metrics.precision, 1.0);
        assert_eq!(metrics.recall, 1.0);
        assert_eq!(metrics.f1_score, 1.0);
    }

    #[test]
    fn test_metrics_binary_imperfect() {
        let predictions = vec![0.9, 0.8, 0.2, 0.3];
        let targets = vec![1.0, 0.0, 1.0, 0.0];

        let metrics = Metrics::compute_binary(&predictions, &targets, 0.5);

        assert_eq!(metrics.accuracy, 0.5); // 2 correct out of 4
    }

    #[test]
    fn test_metrics_multiclass() {
        let predictions = vec![
            vec![0.8, 0.1, 0.1],
            vec![0.1, 0.8, 0.1],
            vec![0.1, 0.1, 0.8],
        ];
        let targets = vec![0, 1, 2];

        let metrics = Metrics::compute_multiclass(&predictions, &targets);
        assert_eq!(metrics.accuracy, 1.0);
    }

    #[test]
    fn test_history() {
        let mut history = History::new();
        history.add_epoch(0.5, 0.6, 0.8, 0.75);
        history.add_epoch(0.3, 0.4, 0.9, 0.85);

        assert_eq!(history.train_losses.len(), 2);
        assert_eq!(history.best_val_loss(), Some(0.4));
        assert_eq!(history.best_val_accuracy(), Some(0.85));
    }

    #[test]
    fn test_optimizer_lr_update() {
        let mut sgd = SGD::new(0.1);
        assert_eq!(sgd.get_lr(), 0.1);

        sgd.set_lr(0.01);
        assert_eq!(sgd.get_lr(), 0.01);
    }
}
