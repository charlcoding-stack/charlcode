// Differentiable Logic
// Fuzzy logic with gradients for neural-symbolic integration
//
// This module implements:
// - Fuzzy truth values (continuous 0-1 instead of binary true/false)
// - Differentiable logic gates (AND, OR, NOT, IMPLIES)
// - Soft unification (returns degree of match, not binary success/failure)
// - Probabilistic logic networks
// - Integration with autograd for gradient-based learning
//
// Usage:
// ```rust
// use charl::symbolic::differentiable_logic::{FuzzyValue, FuzzyLogic};
//
// // Fuzzy truth values
// let p = FuzzyValue::new(0.8);  // 80% true
// let q = FuzzyValue::new(0.6);  // 60% true
//
// // Fuzzy AND
// let result = FuzzyLogic::and(p, q);
// assert_eq!(result.value(), 0.48);  // Product t-norm
//
// // Can compute gradients!
// let grad = result.backward();
// ```

use std::fmt;

/// Fuzzy truth value: continuous value between 0 and 1
/// - 0.0 = completely false
/// - 1.0 = completely true
/// - 0.5 = maximum uncertainty
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FuzzyValue {
    value: f64,
}

impl FuzzyValue {
    /// Create a new fuzzy value
    /// Clamps to [0, 1] range
    pub fn new(value: f64) -> Self {
        FuzzyValue {
            value: value.clamp(0.0, 1.0),
        }
    }

    /// Get the truth value
    pub fn value(&self) -> f64 {
        self.value
    }

    /// Create a true value (1.0)
    pub fn true_value() -> Self {
        FuzzyValue { value: 1.0 }
    }

    /// Create a false value (0.0)
    pub fn false_value() -> Self {
        FuzzyValue { value: 0.0 }
    }

    /// Create an uncertain value (0.5)
    pub fn uncertain() -> Self {
        FuzzyValue { value: 0.5 }
    }

    /// Check if definitely true (>= 0.9)
    pub fn is_true(&self) -> bool {
        self.value >= 0.9
    }

    /// Check if definitely false (<= 0.1)
    pub fn is_false(&self) -> bool {
        self.value <= 0.1
    }

    /// Check if uncertain (around 0.5)
    pub fn is_uncertain(&self) -> bool {
        (self.value - 0.5).abs() < 0.2
    }

    /// Convert to boolean (>= 0.5 = true)
    pub fn to_bool(&self) -> bool {
        self.value >= 0.5
    }

    /// From boolean
    pub fn from_bool(b: bool) -> Self {
        FuzzyValue::new(if b { 1.0 } else { 0.0 })
    }
}

impl fmt::Display for FuzzyValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.3}", self.value)
    }
}

/// T-norm: Triangular norm for fuzzy AND operations
#[derive(Debug, Clone, Copy)]
pub enum TNorm {
    /// Product: AND(a, b) = a * b
    Product,

    /// Minimum: AND(a, b) = min(a, b)
    Minimum,

    /// Lukasiewicz: AND(a, b) = max(0, a + b - 1)
    Lukasiewicz,

    /// Drastic: AND(a, b) = min(a, b) if max(a, b) = 1, else 0
    Drastic,
}

/// T-conorm: Triangular conorm for fuzzy OR operations
#[derive(Debug, Clone, Copy)]
pub enum TConorm {
    /// Probabilistic sum: OR(a, b) = a + b - a*b
    ProbabilisticSum,

    /// Maximum: OR(a, b) = max(a, b)
    Maximum,

    /// Lukasiewicz: OR(a, b) = min(1, a + b)
    Lukasiewicz,

    /// Drastic: OR(a, b) = max(a, b) if min(a, b) = 0, else 1
    Drastic,
}

/// Fuzzy Logic operations
pub struct FuzzyLogic;

impl FuzzyLogic {
    /// Fuzzy NOT: ¬p = 1 - p
    pub fn not(p: FuzzyValue) -> FuzzyValue {
        FuzzyValue::new(1.0 - p.value())
    }

    /// Fuzzy AND using specified t-norm
    pub fn and_with_tnorm(p: FuzzyValue, q: FuzzyValue, tnorm: TNorm) -> FuzzyValue {
        let result = match tnorm {
            TNorm::Product => p.value() * q.value(),
            TNorm::Minimum => p.value().min(q.value()),
            TNorm::Lukasiewicz => (p.value() + q.value() - 1.0).max(0.0),
            TNorm::Drastic => {
                if p.value().max(q.value()) == 1.0 {
                    p.value().min(q.value())
                } else {
                    0.0
                }
            }
        };
        FuzzyValue::new(result)
    }

    /// Fuzzy AND (default: product t-norm)
    pub fn and(p: FuzzyValue, q: FuzzyValue) -> FuzzyValue {
        Self::and_with_tnorm(p, q, TNorm::Product)
    }

    /// Fuzzy OR using specified t-conorm
    pub fn or_with_tconorm(p: FuzzyValue, q: FuzzyValue, tconorm: TConorm) -> FuzzyValue {
        let result = match tconorm {
            TConorm::ProbabilisticSum => p.value() + q.value() - p.value() * q.value(),
            TConorm::Maximum => p.value().max(q.value()),
            TConorm::Lukasiewicz => (p.value() + q.value()).min(1.0),
            TConorm::Drastic => {
                if p.value().min(q.value()) == 0.0 {
                    p.value().max(q.value())
                } else {
                    1.0
                }
            }
        };
        FuzzyValue::new(result)
    }

    /// Fuzzy OR (default: probabilistic sum)
    pub fn or(p: FuzzyValue, q: FuzzyValue) -> FuzzyValue {
        Self::or_with_tconorm(p, q, TConorm::ProbabilisticSum)
    }

    /// Fuzzy IMPLIES: p → q = ¬p ∨ q
    pub fn implies(p: FuzzyValue, q: FuzzyValue) -> FuzzyValue {
        Self::or(Self::not(p), q)
    }

    /// Fuzzy EQUIVALENT: p ↔ q = (p → q) ∧ (q → p)
    pub fn equivalent(p: FuzzyValue, q: FuzzyValue) -> FuzzyValue {
        Self::and(
            Self::implies(p, q),
            Self::implies(q, p),
        )
    }

    /// Fuzzy XOR: p ⊕ q = (p ∨ q) ∧ ¬(p ∧ q)
    pub fn xor(p: FuzzyValue, q: FuzzyValue) -> FuzzyValue {
        Self::and(
            Self::or(p, q),
            Self::not(Self::and(p, q)),
        )
    }
}

/// Differentiable logic gate with gradient tracking
#[derive(Debug, Clone)]
pub struct DifferentiableGate {
    /// Current value
    value: FuzzyValue,

    /// Gradient (for backpropagation)
    gradient: f64,

    /// Operation type (for gradient computation)
    operation: GateOperation,

    /// Input gates (for backward pass)
    inputs: Vec<Box<DifferentiableGate>>,
}

#[derive(Debug, Clone)]
enum GateOperation {
    Input,
    Not,
    And,
    Or,
    Implies,
}

impl DifferentiableGate {
    /// Create an input gate
    pub fn input(value: FuzzyValue) -> Self {
        DifferentiableGate {
            value,
            gradient: 0.0,
            operation: GateOperation::Input,
            inputs: Vec::new(),
        }
    }

    /// Get the current value
    pub fn value(&self) -> FuzzyValue {
        self.value
    }

    /// Get the gradient
    pub fn gradient(&self) -> f64 {
        self.gradient
    }

    /// NOT gate
    pub fn not(input: DifferentiableGate) -> Self {
        let value = FuzzyLogic::not(input.value);
        DifferentiableGate {
            value,
            gradient: 0.0,
            operation: GateOperation::Not,
            inputs: vec![Box::new(input)],
        }
    }

    /// AND gate
    pub fn and(left: DifferentiableGate, right: DifferentiableGate) -> Self {
        let value = FuzzyLogic::and(left.value, right.value);
        DifferentiableGate {
            value,
            gradient: 0.0,
            operation: GateOperation::And,
            inputs: vec![Box::new(left), Box::new(right)],
        }
    }

    /// OR gate
    pub fn or(left: DifferentiableGate, right: DifferentiableGate) -> Self {
        let value = FuzzyLogic::or(left.value, right.value);
        DifferentiableGate {
            value,
            gradient: 0.0,
            operation: GateOperation::Or,
            inputs: vec![Box::new(left), Box::new(right)],
        }
    }

    /// IMPLIES gate
    pub fn implies(left: DifferentiableGate, right: DifferentiableGate) -> Self {
        let value = FuzzyLogic::implies(left.value, right.value);
        DifferentiableGate {
            value,
            gradient: 0.0,
            operation: GateOperation::Implies,
            inputs: vec![Box::new(left), Box::new(right)],
        }
    }

    /// Backward pass: compute gradients
    pub fn backward(&mut self, upstream_gradient: f64) {
        self.gradient += upstream_gradient;

        match self.operation {
            GateOperation::Input => {
                // No inputs to propagate to
            }
            GateOperation::Not => {
                // d/dx (1 - x) = -1
                if let Some(input) = self.inputs.get_mut(0) {
                    input.backward(-upstream_gradient);
                }
            }
            GateOperation::And => {
                // d/dx (x * y) = y
                // d/dy (x * y) = x
                if self.inputs.len() >= 2 {
                    let left_val = self.inputs[0].value.value();
                    let right_val = self.inputs[1].value.value();

                    self.inputs[0].backward(upstream_gradient * right_val);
                    self.inputs[1].backward(upstream_gradient * left_val);
                }
            }
            GateOperation::Or => {
                // d/dx (x + y - xy) = 1 - y
                // d/dy (x + y - xy) = 1 - x
                if self.inputs.len() >= 2 {
                    let left_val = self.inputs[0].value.value();
                    let right_val = self.inputs[1].value.value();

                    self.inputs[0].backward(upstream_gradient * (1.0 - right_val));
                    self.inputs[1].backward(upstream_gradient * (1.0 - left_val));
                }
            }
            GateOperation::Implies => {
                // p → q = (1 - p) + q - (1 - p) * q
                // Gradient computation similar to OR
                if self.inputs.len() >= 2 {
                    let left_val = self.inputs[0].value.value();
                    let right_val = self.inputs[1].value.value();

                    // ∂/∂p = -1 + q
                    self.inputs[0].backward(upstream_gradient * (-1.0 + right_val));
                    // ∂/∂q = 1 - (1 - p) = p
                    self.inputs[1].backward(upstream_gradient * left_val);
                }
            }
        }
    }
}

/// Probabilistic truth value with mean and variance
#[derive(Debug, Clone, Copy)]
pub struct ProbabilisticTruth {
    /// Mean truth value
    pub mean: f64,

    /// Variance (uncertainty)
    pub variance: f64,

    /// Confidence (number of observations)
    pub confidence: f64,
}

impl ProbabilisticTruth {
    /// Create a new probabilistic truth value
    pub fn new(mean: f64, variance: f64, confidence: f64) -> Self {
        ProbabilisticTruth {
            mean: mean.clamp(0.0, 1.0),
            variance: variance.max(0.0),
            confidence: confidence.max(0.0),
        }
    }

    /// Create from observations
    pub fn from_observations(observations: &[bool]) -> Self {
        if observations.is_empty() {
            return ProbabilisticTruth::uncertain();
        }

        let n = observations.len() as f64;
        let mean = observations.iter().filter(|&&x| x).count() as f64 / n;

        // Variance of Bernoulli: p(1-p)
        let variance = mean * (1.0 - mean);

        ProbabilisticTruth {
            mean,
            variance,
            confidence: n,
        }
    }

    /// Maximum uncertainty
    pub fn uncertain() -> Self {
        ProbabilisticTruth {
            mean: 0.5,
            variance: 0.25,  // Maximum variance at p=0.5
            confidence: 0.0,
        }
    }

    /// Certain true
    pub fn certain_true() -> Self {
        ProbabilisticTruth {
            mean: 1.0,
            variance: 0.0,
            confidence: f64::INFINITY,
        }
    }

    /// Certain false
    pub fn certain_false() -> Self {
        ProbabilisticTruth {
            mean: 0.0,
            variance: 0.0,
            confidence: f64::INFINITY,
        }
    }

    /// Convert to fuzzy value (use mean)
    pub fn to_fuzzy(&self) -> FuzzyValue {
        FuzzyValue::new(self.mean)
    }

    /// Standard deviation
    pub fn std_dev(&self) -> f64 {
        self.variance.sqrt()
    }

    /// 95% confidence interval
    pub fn confidence_interval_95(&self) -> (f64, f64) {
        let z = 1.96;  // 95% confidence
        let margin = z * self.std_dev();
        (
            (self.mean - margin).max(0.0),
            (self.mean + margin).min(1.0),
        )
    }
}

impl fmt::Display for ProbabilisticTruth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.3} ± {:.3} (n={:.0})",
            self.mean, self.std_dev(), self.confidence)
    }
}

/// Soft unification: returns degree of match instead of binary success/failure
pub fn soft_unify(term1: &str, term2: &str) -> FuzzyValue {
    // Simple string similarity as soft unification
    // In a real implementation, this would work with actual FOL terms

    if term1 == term2 {
        // Perfect match
        return FuzzyValue::true_value();
    }

    // Compute edit distance similarity
    let distance = edit_distance(term1, term2);
    let max_len = term1.len().max(term2.len()) as f64;

    if max_len == 0.0 {
        return FuzzyValue::true_value();
    }

    // Similarity = 1 - (distance / max_length)
    let similarity = 1.0 - (distance as f64 / max_len);
    FuzzyValue::new(similarity)
}

/// Levenshtein edit distance
fn edit_distance(s1: &str, s2: &str) -> usize {
    let len1 = s1.len();
    let len2 = s2.len();

    let mut matrix = vec![vec![0; len2 + 1]; len1 + 1];

    for i in 0..=len1 {
        matrix[i][0] = i;
    }
    for j in 0..=len2 {
        matrix[0][j] = j;
    }

    for (i, c1) in s1.chars().enumerate() {
        for (j, c2) in s2.chars().enumerate() {
            let cost = if c1 == c2 { 0 } else { 1 };
            matrix[i + 1][j + 1] = (matrix[i][j + 1] + 1)
                .min(matrix[i + 1][j] + 1)
                .min(matrix[i][j] + cost);
        }
    }

    matrix[len1][len2]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fuzzy_value_creation() {
        let v1 = FuzzyValue::new(0.7);
        assert_eq!(v1.value(), 0.7);

        // Clamping
        let v2 = FuzzyValue::new(1.5);
        assert_eq!(v2.value(), 1.0);

        let v3 = FuzzyValue::new(-0.5);
        assert_eq!(v3.value(), 0.0);
    }

    #[test]
    fn test_fuzzy_predicates() {
        let true_val = FuzzyValue::new(0.95);
        assert!(true_val.is_true());
        assert!(!true_val.is_false());

        let false_val = FuzzyValue::new(0.05);
        assert!(false_val.is_false());
        assert!(!false_val.is_true());

        let uncertain_val = FuzzyValue::new(0.5);
        assert!(uncertain_val.is_uncertain());
    }

    #[test]
    fn test_fuzzy_not() {
        let p = FuzzyValue::new(0.8);
        let not_p = FuzzyLogic::not(p);
        assert!((not_p.value() - 0.2).abs() < 0.0001);
    }

    #[test]
    fn test_fuzzy_and() {
        let p = FuzzyValue::new(0.8);
        let q = FuzzyValue::new(0.6);

        // Product t-norm: 0.8 * 0.6 = 0.48
        let result = FuzzyLogic::and(p, q);
        assert!((result.value() - 0.48).abs() < 0.001);

        // Minimum t-norm: min(0.8, 0.6) = 0.6
        let result_min = FuzzyLogic::and_with_tnorm(p, q, TNorm::Minimum);
        assert_eq!(result_min.value(), 0.6);
    }

    #[test]
    fn test_fuzzy_or() {
        let p = FuzzyValue::new(0.8);
        let q = FuzzyValue::new(0.6);

        // Probabilistic sum: 0.8 + 0.6 - 0.8*0.6 = 0.92
        let result = FuzzyLogic::or(p, q);
        assert!((result.value() - 0.92).abs() < 0.001);

        // Maximum t-conorm: max(0.8, 0.6) = 0.8
        let result_max = FuzzyLogic::or_with_tconorm(p, q, TConorm::Maximum);
        assert_eq!(result_max.value(), 0.8);
    }

    #[test]
    fn test_fuzzy_implies() {
        let p = FuzzyValue::new(0.8);
        let q = FuzzyValue::new(0.6);

        // p → q = ¬p ∨ q = 0.2 ∨ 0.6
        let result = FuzzyLogic::implies(p, q);

        // Should be high because q is somewhat true
        assert!(result.value() > 0.6);
    }

    #[test]
    fn test_fuzzy_laws() {
        let p = FuzzyValue::new(0.7);

        // Law of excluded middle: p ∨ ¬p should be close to 1 in classical logic
        // But in fuzzy logic it may not be exactly 1
        let excluded_middle = FuzzyLogic::or(p, FuzzyLogic::not(p));
        assert!(excluded_middle.value() > 0.5);

        // Law of contradiction: p ∧ ¬p should be close to 0
        let contradiction = FuzzyLogic::and(p, FuzzyLogic::not(p));
        assert!(contradiction.value() < 0.5);
    }

    #[test]
    fn test_differentiable_gate_forward() {
        let p = DifferentiableGate::input(FuzzyValue::new(0.8));
        let q = DifferentiableGate::input(FuzzyValue::new(0.6));

        let result = DifferentiableGate::and(p, q);
        assert!((result.value().value() - 0.48).abs() < 0.001);
    }

    #[test]
    fn test_differentiable_gate_backward() {
        let p = DifferentiableGate::input(FuzzyValue::new(0.8));
        let q = DifferentiableGate::input(FuzzyValue::new(0.6));

        let mut result = DifferentiableGate::and(p, q);

        // Backward pass
        result.backward(1.0);

        // Gradients should be computed
        assert!(result.gradient() > 0.0);
    }

    #[test]
    fn test_probabilistic_truth() {
        let observations = vec![true, true, false, true, true];
        let truth = ProbabilisticTruth::from_observations(&observations);

        // Mean should be 4/5 = 0.8
        assert!((truth.mean - 0.8).abs() < 0.001);

        // Should have positive confidence
        assert!(truth.confidence > 0.0);
    }

    #[test]
    fn test_soft_unification() {
        // Perfect match
        let perfect = soft_unify("socrates", "socrates");
        assert_eq!(perfect.value(), 1.0);

        // Similar strings
        let similar = soft_unify("socrates", "socrate");
        assert!(similar.value() > 0.8);

        // Very different strings
        let different = soft_unify("socrates", "plato");
        assert!(different.value() < 0.5);
    }

    #[test]
    fn test_edit_distance() {
        assert_eq!(edit_distance("", ""), 0);
        assert_eq!(edit_distance("cat", "cat"), 0);
        assert_eq!(edit_distance("cat", "cut"), 1);
        assert_eq!(edit_distance("kitten", "sitting"), 3);
    }

    #[test]
    fn test_confidence_interval() {
        let truth = ProbabilisticTruth::new(0.7, 0.04, 100.0);
        let (lower, upper) = truth.confidence_interval_95();

        assert!(lower < 0.7);
        assert!(upper > 0.7);
        assert!(upper - lower < 1.0);
    }
}
