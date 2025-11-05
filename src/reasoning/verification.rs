// Self-Verification & Critique
//
// Verifiable reasoning through explicit validation and uncertainty quantification.
//
// Key capabilities:
// 1. Verification: Check logical consistency, facts, calculations
// 2. Critique: Self-critique and iterative refinement
// 3. Uncertainty: Epistemic and aleatoric uncertainty quantification
//
// Example:
//   Reasoning: "2 + 2 = 5"
//   Verification: FAILED - calculation error detected
//   Critique: "The sum is incorrect. 2 + 2 = 4, not 5."
//   Confidence: LOW (0.1)
//
// References:
// - Varshney et al. (2022): "Self-Consistency Improves Chain of Thought Reasoning"
// - Gal & Ghahramani (2016): "Dropout as a Bayesian Approximation"
// - Guo et al. (2017): "On Calibration of Modern Neural Networks"

use std::collections::HashMap;
use crate::reasoning::chain_of_thought::{ChainOfThought, ReasoningStep};

/// Verification result for a reasoning step or chain
#[derive(Debug, Clone, PartialEq)]
pub enum VerificationStatus {
    /// Verification passed
    Passed,
    /// Verification failed
    Failed(String), // Error message
    /// Unable to verify (insufficient information)
    Unknown,
}

impl VerificationStatus {
    pub fn is_passed(&self) -> bool {
        matches!(self, VerificationStatus::Passed)
    }

    pub fn is_failed(&self) -> bool {
        matches!(self, VerificationStatus::Failed(_))
    }
}

/// Type of verification check
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VerificationType {
    /// Logical consistency (no contradictions)
    LogicalConsistency,
    /// Fact checking against knowledge
    FactChecking,
    /// Mathematical calculation verification
    CalculationVerification,
    /// Contradiction detection
    ContradictionDetection,
}

/// Result of a verification check
#[derive(Debug, Clone)]
pub struct VerificationResult {
    pub verification_type: VerificationType,
    pub status: VerificationStatus,
    pub confidence: f32, // [0, 1] - confidence in the verification
    pub details: String,
}

impl VerificationResult {
    pub fn new(
        verification_type: VerificationType,
        status: VerificationStatus,
        confidence: f32,
        details: impl Into<String>,
    ) -> Self {
        Self {
            verification_type,
            status,
            confidence: confidence.clamp(0.0, 1.0),
            details: details.into(),
        }
    }

    pub fn passed(verification_type: VerificationType, details: impl Into<String>) -> Self {
        Self::new(verification_type, VerificationStatus::Passed, 1.0, details)
    }

    pub fn failed(verification_type: VerificationType, error: impl Into<String>) -> Self {
        let error_str = error.into();
        Self::new(
            verification_type,
            VerificationStatus::Failed(error_str.clone()),
            1.0,
            error_str,
        )
    }

    pub fn unknown(verification_type: VerificationType, details: impl Into<String>) -> Self {
        Self::new(verification_type, VerificationStatus::Unknown, 0.5, details)
    }
}

/// Verifier for reasoning chains
pub struct ReasoningVerifier {
    /// Enable different verification types
    pub enable_logical_consistency: bool,
    pub enable_fact_checking: bool,
    pub enable_calculation_verification: bool,
    pub enable_contradiction_detection: bool,
    /// Known facts for fact checking (key-value pairs)
    pub knowledge_base: HashMap<String, String>,
}

impl ReasoningVerifier {
    pub fn new() -> Self {
        Self {
            enable_logical_consistency: true,
            enable_fact_checking: true,
            enable_calculation_verification: true,
            enable_contradiction_detection: true,
            knowledge_base: HashMap::new(),
        }
    }

    /// Add a fact to knowledge base
    pub fn add_fact(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.knowledge_base.insert(key.into(), value.into());
    }

    /// Verify a single reasoning step
    pub fn verify_step(&self, step: &ReasoningStep) -> Vec<VerificationResult> {
        let mut results = Vec::new();

        // Logical consistency
        if self.enable_logical_consistency {
            results.push(self.check_logical_consistency(step));
        }

        // Fact checking
        if self.enable_fact_checking {
            results.push(self.check_facts(step));
        }

        // Calculation verification
        if self.enable_calculation_verification && step.computation.is_some() {
            results.push(self.check_calculation(step));
        }

        results
    }

    /// Verify entire chain of thought
    pub fn verify_chain(&self, chain: &ChainOfThought) -> Vec<VerificationResult> {
        let mut results = Vec::new();

        // Verify each step
        for step in &chain.steps {
            results.extend(self.verify_step(step));
        }

        // Contradiction detection across steps
        if self.enable_contradiction_detection {
            results.push(self.check_contradictions(chain));
        }

        results
    }

    /// Check logical consistency of a step
    fn check_logical_consistency(&self, step: &ReasoningStep) -> VerificationResult {
        // Simple heuristic: check if step has content
        if step.thought.is_empty() {
            return VerificationResult::failed(
                VerificationType::LogicalConsistency,
                "Empty reasoning step",
            );
        }

        // Check for obvious logical issues
        let thought_lower = step.thought.to_lowercase();
        if thought_lower.contains("impossible") && thought_lower.contains("therefore true") {
            return VerificationResult::failed(
                VerificationType::LogicalConsistency,
                "Contradiction: impossible implies false, not true",
            );
        }

        VerificationResult::passed(
            VerificationType::LogicalConsistency,
            "Step is logically consistent",
        )
    }

    /// Check facts against knowledge base
    fn check_facts(&self, step: &ReasoningStep) -> VerificationResult {
        // Check each evidence against knowledge base
        for evidence in &step.evidence {
            for (key, known_value) in &self.knowledge_base {
                if evidence.to_lowercase().contains(&key.to_lowercase()) {
                    // Found a fact to check
                    if !evidence.to_lowercase().contains(&known_value.to_lowercase()) {
                        return VerificationResult::failed(
                            VerificationType::FactChecking,
                            format!("Fact mismatch: expected '{}' to contain '{}'", evidence, known_value),
                        );
                    }
                }
            }
        }

        VerificationResult::passed(
            VerificationType::FactChecking,
            "All facts consistent with knowledge base",
        )
    }

    /// Check mathematical calculations
    fn check_calculation(&self, step: &ReasoningStep) -> VerificationResult {
        if let Some(result) = step.computation {
            // Try to parse calculation from thought
            // Simple cases: "2 + 3 = 5", "10 * 2 = 20"
            let thought = &step.thought;

            // Check addition
            if let Some(expected) = self.parse_addition(thought) {
                if (result - expected).abs() < 1e-5 {
                    return VerificationResult::passed(
                        VerificationType::CalculationVerification,
                        "Calculation correct",
                    );
                } else {
                    return VerificationResult::failed(
                        VerificationType::CalculationVerification,
                        format!("Calculation error: expected {}, got {}", expected, result),
                    );
                }
            }

            // Check multiplication
            if let Some(expected) = self.parse_multiplication(thought) {
                if (result - expected).abs() < 1e-5 {
                    return VerificationResult::passed(
                        VerificationType::CalculationVerification,
                        "Calculation correct",
                    );
                } else {
                    return VerificationResult::failed(
                        VerificationType::CalculationVerification,
                        format!("Calculation error: expected {}, got {}", expected, result),
                    );
                }
            }

            // Unable to verify
            VerificationResult::unknown(
                VerificationType::CalculationVerification,
                "Unable to parse calculation",
            )
        } else {
            VerificationResult::unknown(
                VerificationType::CalculationVerification,
                "No computation to verify",
            )
        }
    }

    /// Check for contradictions across chain
    fn check_contradictions(&self, chain: &ChainOfThought) -> VerificationResult {
        // Simple heuristic: look for contradictory statements
        for i in 0..chain.steps.len() {
            for j in (i + 1)..chain.steps.len() {
                let step_i = &chain.steps[i].thought.to_lowercase();
                let step_j = &chain.steps[j].thought.to_lowercase();

                // Check for "X is true" followed by "X is false"
                if step_i.contains("true") && step_j.contains("false") {
                    let words_i: Vec<&str> = step_i.split_whitespace().collect();
                    let words_j: Vec<&str> = step_j.split_whitespace().collect();

                    // Look for common words (potential contradiction)
                    for word_i in &words_i {
                        if words_j.contains(word_i) && word_i.len() > 3 {
                            return VerificationResult::failed(
                                VerificationType::ContradictionDetection,
                                format!("Potential contradiction between steps {} and {}", i + 1, j + 1),
                            );
                        }
                    }
                }
            }
        }

        VerificationResult::passed(
            VerificationType::ContradictionDetection,
            "No contradictions detected",
        )
    }

    /// Parse simple addition: "2 + 3" -> Some(5.0)
    fn parse_addition(&self, text: &str) -> Option<f32> {
        // Look for pattern: number + number
        let parts: Vec<&str> = text.split('+').collect();
        if parts.len() == 2 {
            let a = parts[0].trim().split_whitespace().last()?.parse::<f32>().ok()?;
            let b = parts[1].trim().split_whitespace().next()?.parse::<f32>().ok()?;
            return Some(a + b);
        }
        None
    }

    /// Parse simple multiplication: "2 × 3" or "2 * 3" -> Some(6.0)
    fn parse_multiplication(&self, text: &str) -> Option<f32> {
        // Try × first
        for separator in &["×", "*", "x"] {
            let parts: Vec<&str> = text.split(separator).collect();
            if parts.len() == 2 {
                if let (Ok(a), Ok(b)) = (
                    parts[0].trim().split_whitespace().last()?.parse::<f32>(),
                    parts[1].trim().split_whitespace().next()?.parse::<f32>(),
                ) {
                    return Some(a * b);
                }
            }
        }
        None
    }
}

impl Default for ReasoningVerifier {
    fn default() -> Self {
        Self::new()
    }
}

/// Self-critique: Generate critique of reasoning
pub struct SelfCritique {
    /// Threshold for low confidence (triggers critique)
    pub low_confidence_threshold: f32,
    /// Maximum refinement iterations
    pub max_iterations: usize,
}

impl SelfCritique {
    pub fn new() -> Self {
        Self {
            low_confidence_threshold: 0.6,
            max_iterations: 3,
        }
    }

    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.low_confidence_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Generate critique of a reasoning chain
    pub fn critique(&self, chain: &ChainOfThought, verification_results: &[VerificationResult]) -> Critique {
        let mut issues = Vec::new();
        let mut suggestions = Vec::new();

        // Check verification results
        for result in verification_results {
            if result.status.is_failed() {
                issues.push(format!("{:?}: {}", result.verification_type, result.details));
                suggestions.push(format!("Fix {} issue", format!("{:?}", result.verification_type)));
            }
        }

        // Check confidence
        if chain.confidence < self.low_confidence_threshold {
            issues.push(format!("Low confidence: {:.2}", chain.confidence));
            suggestions.push("Add more reasoning steps or evidence".to_string());
        }

        // Check step verification
        for (i, step) in chain.steps.iter().enumerate() {
            if !step.verified {
                issues.push(format!("Step {} not verified", i + 1));
                suggestions.push(format!("Verify step {}", i + 1));
            }
            if step.confidence < self.low_confidence_threshold {
                issues.push(format!("Step {} has low confidence: {:.2}", i + 1, step.confidence));
            }
        }

        let requires_refinement = !issues.is_empty();
        let overall_quality = if issues.is_empty() {
            CritiqueQuality::Good
        } else if issues.len() <= 2 {
            CritiqueQuality::Fair
        } else {
            CritiqueQuality::Poor
        };

        Critique {
            quality: overall_quality,
            issues,
            suggestions,
            requires_refinement,
        }
    }

    /// Iterative refinement (simplified - returns whether refinement needed)
    pub fn should_refine(&self, critique: &Critique, iteration: usize) -> bool {
        critique.requires_refinement && iteration < self.max_iterations
    }
}

impl Default for SelfCritique {
    fn default() -> Self {
        Self::new()
    }
}

/// Quality assessment of reasoning
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CritiqueQuality {
    Good,
    Fair,
    Poor,
}

/// Critique of reasoning chain
#[derive(Debug, Clone)]
pub struct Critique {
    pub quality: CritiqueQuality,
    pub issues: Vec<String>,
    pub suggestions: Vec<String>,
    pub requires_refinement: bool,
}

/// Uncertainty quantification
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UncertaintyType {
    /// Epistemic: Model doesn't know (reducible with more data)
    Epistemic,
    /// Aleatoric: Inherent randomness (irreducible)
    Aleatoric,
}

/// Uncertainty estimate
#[derive(Debug, Clone)]
pub struct UncertaintyEstimate {
    pub epistemic: f32, // [0, 1]
    pub aleatoric: f32, // [0, 1]
    pub total: f32,     // [0, 1]
    pub calibrated_confidence: f32, // [0, 1] - calibrated based on uncertainty
}

impl UncertaintyEstimate {
    pub fn new(epistemic: f32, aleatoric: f32) -> Self {
        let epistemic = epistemic.clamp(0.0, 1.0);
        let aleatoric = aleatoric.clamp(0.0, 1.0);
        let total = (epistemic.powi(2) + aleatoric.powi(2)).sqrt().min(1.0);
        let calibrated_confidence = 1.0 - total;

        Self {
            epistemic,
            aleatoric,
            total,
            calibrated_confidence,
        }
    }

    pub fn from_confidence(confidence: f32) -> Self {
        // Simple heuristic: assume equal epistemic and aleatoric
        let uncertainty = 1.0 - confidence.clamp(0.0, 1.0);
        let component = uncertainty / 2.0_f32.sqrt();
        Self::new(component, component)
    }

    pub fn is_high_uncertainty(&self) -> bool {
        self.total > 0.5
    }
}

/// Uncertainty quantifier
pub struct UncertaintyQuantifier {
    /// Calibration curve (maps raw confidence to calibrated confidence)
    calibration_map: HashMap<u8, f32>, // Discretized: 0-100 -> calibrated
}

impl UncertaintyQuantifier {
    pub fn new() -> Self {
        Self {
            calibration_map: HashMap::new(),
        }
    }

    /// Add calibration point
    pub fn add_calibration(&mut self, raw_confidence: f32, calibrated_confidence: f32) {
        let key = (raw_confidence * 100.0) as u8;
        self.calibration_map.insert(key, calibrated_confidence.clamp(0.0, 1.0));
    }

    /// Quantify uncertainty for a reasoning chain
    pub fn quantify(&self, chain: &ChainOfThought) -> UncertaintyEstimate {
        // Epistemic: Based on step confidence variance (model uncertainty)
        let confidences: Vec<f32> = chain.steps.iter().map(|s| s.confidence).collect();
        let mean_confidence = if confidences.is_empty() {
            chain.confidence
        } else {
            confidences.iter().sum::<f32>() / confidences.len() as f32
        };

        let variance = if confidences.len() > 1 {
            confidences.iter()
                .map(|c| (c - mean_confidence).powi(2))
                .sum::<f32>() / confidences.len() as f32
        } else {
            0.0
        };

        let epistemic = variance.sqrt().min(1.0);

        // Aleatoric: Based on verification failures (inherent difficulty)
        let verification_rate = chain.steps.iter()
            .filter(|s| s.verified)
            .count() as f32 / chain.steps.len().max(1) as f32;
        let aleatoric = (1.0 - verification_rate).min(1.0);

        let mut estimate = UncertaintyEstimate::new(epistemic, aleatoric);

        // Apply calibration if available
        if let Some(&calibrated) = self.calibration_map.get(&((chain.confidence * 100.0) as u8)) {
            estimate.calibrated_confidence = calibrated;
        }

        estimate
    }

    /// Calibrate confidence based on historical accuracy
    pub fn calibrate_confidence(&self, raw_confidence: f32) -> f32 {
        let key = (raw_confidence * 100.0) as u8;
        self.calibration_map.get(&key).copied().unwrap_or(raw_confidence)
    }
}

impl Default for UncertaintyQuantifier {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verification_status() {
        let passed = VerificationStatus::Passed;
        assert!(passed.is_passed());
        assert!(!passed.is_failed());

        let failed = VerificationStatus::Failed("Error".to_string());
        assert!(!failed.is_passed());
        assert!(failed.is_failed());
    }

    #[test]
    fn test_verification_result_creation() {
        let result = VerificationResult::passed(
            VerificationType::LogicalConsistency,
            "Test passed",
        );
        assert!(result.status.is_passed());
        assert_eq!(result.confidence, 1.0);
    }

    #[test]
    fn test_verifier_creation() {
        let verifier = ReasoningVerifier::new();
        assert!(verifier.enable_logical_consistency);
        assert!(verifier.enable_fact_checking);
        assert!(verifier.enable_calculation_verification);
        assert!(verifier.enable_contradiction_detection);
    }

    #[test]
    fn test_add_fact() {
        let mut verifier = ReasoningVerifier::new();
        verifier.add_fact("water", "H2O");
        assert_eq!(verifier.knowledge_base.get("water"), Some(&"H2O".to_string()));
    }

    #[test]
    fn test_verify_empty_step() {
        let verifier = ReasoningVerifier::new();
        let step = ReasoningStep::new(1, "");

        let results = verifier.verify_step(&step);
        let logical_check = results.iter()
            .find(|r| r.verification_type == VerificationType::LogicalConsistency);

        assert!(logical_check.is_some());
        assert!(logical_check.unwrap().status.is_failed());
    }

    #[test]
    fn test_verify_valid_step() {
        let verifier = ReasoningVerifier::new();
        let step = ReasoningStep::new(1, "The sky is blue");

        let results = verifier.verify_step(&step);
        let logical_check = results.iter()
            .find(|r| r.verification_type == VerificationType::LogicalConsistency);

        assert!(logical_check.is_some());
        assert!(logical_check.unwrap().status.is_passed());
    }

    #[test]
    fn test_fact_checking() {
        let mut verifier = ReasoningVerifier::new();
        verifier.add_fact("water", "H2O");

        let mut step = ReasoningStep::new(1, "Water is H2O");
        step = step.with_evidence("Water is H2O");

        let results = verifier.verify_step(&step);
        let fact_check = results.iter()
            .find(|r| r.verification_type == VerificationType::FactChecking);

        assert!(fact_check.is_some());
        assert!(fact_check.unwrap().status.is_passed());
    }

    #[test]
    fn test_calculation_verification_addition() {
        let verifier = ReasoningVerifier::new();
        let step = ReasoningStep::new(1, "Calculate 2 + 3")
            .with_computation(5.0);

        let results = verifier.verify_step(&step);
        let calc_check = results.iter()
            .find(|r| r.verification_type == VerificationType::CalculationVerification);

        assert!(calc_check.is_some());
        assert!(calc_check.unwrap().status.is_passed());
    }

    #[test]
    fn test_calculation_verification_wrong() {
        let verifier = ReasoningVerifier::new();
        let step = ReasoningStep::new(1, "Calculate 2 + 3")
            .with_computation(6.0); // Wrong!

        let results = verifier.verify_step(&step);
        let calc_check = results.iter()
            .find(|r| r.verification_type == VerificationType::CalculationVerification);

        assert!(calc_check.is_some());
        assert!(calc_check.unwrap().status.is_failed());
    }

    #[test]
    fn test_calculation_verification_multiplication() {
        let verifier = ReasoningVerifier::new();
        let step = ReasoningStep::new(1, "Calculate 4 × 5")
            .with_computation(20.0);

        let results = verifier.verify_step(&step);
        let calc_check = results.iter()
            .find(|r| r.verification_type == VerificationType::CalculationVerification);

        assert!(calc_check.is_some());
        assert!(calc_check.unwrap().status.is_passed());
    }

    #[test]
    fn test_parse_addition() {
        let verifier = ReasoningVerifier::new();
        assert_eq!(verifier.parse_addition("Calculate 2 + 3"), Some(5.0));
        assert_eq!(verifier.parse_addition("10 + 5"), Some(15.0));
    }

    #[test]
    fn test_parse_multiplication() {
        let verifier = ReasoningVerifier::new();
        assert_eq!(verifier.parse_multiplication("Calculate 3 × 4"), Some(12.0));
        assert_eq!(verifier.parse_multiplication("6 * 7"), Some(42.0));
    }

    #[test]
    fn test_verify_chain() {
        let verifier = ReasoningVerifier::new();
        let chain = ChainOfThought::new("Test problem")
            .add_step(ReasoningStep::new(1, "Step 1"))
            .add_step(ReasoningStep::new(2, "Step 2"));

        let results = verifier.verify_chain(&chain);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_self_critique_creation() {
        let critique = SelfCritique::new();
        assert_eq!(critique.low_confidence_threshold, 0.6);
        assert_eq!(critique.max_iterations, 3);
    }

    #[test]
    fn test_critique_good_chain() {
        let critique_engine = SelfCritique::new();
        let chain = ChainOfThought::new("Problem")
            .add_step(ReasoningStep::new(1, "Step 1").with_confidence(0.9).verify())
            .with_confidence(0.9);

        let critique = critique_engine.critique(&chain, &[]);
        assert_eq!(critique.quality, CritiqueQuality::Good);
        assert!(!critique.requires_refinement);
    }

    #[test]
    fn test_critique_low_confidence() {
        let critique_engine = SelfCritique::new();
        let chain = ChainOfThought::new("Problem")
            .add_step(ReasoningStep::new(1, "Step 1").with_confidence(0.3))
            .with_confidence(0.3);

        let critique = critique_engine.critique(&chain, &[]);
        assert_ne!(critique.quality, CritiqueQuality::Good);
        assert!(critique.requires_refinement);
        assert!(!critique.issues.is_empty());
    }

    #[test]
    fn test_critique_unverified_step() {
        let critique_engine = SelfCritique::new();
        let chain = ChainOfThought::new("Problem")
            .add_step(ReasoningStep::new(1, "Unverified step"))
            .with_confidence(0.8);

        let critique = critique_engine.critique(&chain, &[]);
        assert!(critique.requires_refinement);
        assert!(critique.issues.iter().any(|i| i.contains("not verified")));
    }

    #[test]
    fn test_should_refine() {
        let critique_engine = SelfCritique::new();
        let critique = Critique {
            quality: CritiqueQuality::Poor,
            issues: vec!["Issue".to_string()],
            suggestions: vec!["Fix".to_string()],
            requires_refinement: true,
        };

        assert!(critique_engine.should_refine(&critique, 0));
        assert!(critique_engine.should_refine(&critique, 2));
        assert!(!critique_engine.should_refine(&critique, 3)); // Max iterations
    }

    #[test]
    fn test_uncertainty_estimate_creation() {
        let estimate = UncertaintyEstimate::new(0.3, 0.4);
        assert_eq!(estimate.epistemic, 0.3);
        assert_eq!(estimate.aleatoric, 0.4);
        assert!(estimate.total > 0.0);
        assert!(estimate.calibrated_confidence < 1.0);
    }

    #[test]
    fn test_uncertainty_from_confidence() {
        let estimate = UncertaintyEstimate::from_confidence(0.8);
        assert!(estimate.epistemic > 0.0);
        assert!(estimate.aleatoric > 0.0);
        assert!(!estimate.is_high_uncertainty());
    }

    #[test]
    fn test_high_uncertainty() {
        let estimate = UncertaintyEstimate::new(0.6, 0.6);
        assert!(estimate.is_high_uncertainty());
    }

    #[test]
    fn test_uncertainty_quantifier() {
        let quantifier = UncertaintyQuantifier::new();
        let chain = ChainOfThought::new("Problem")
            .add_step(ReasoningStep::new(1, "Step 1").with_confidence(0.9))
            .add_step(ReasoningStep::new(2, "Step 2").with_confidence(0.8))
            .with_confidence(0.85);

        let estimate = quantifier.quantify(&chain);
        assert!(estimate.epistemic >= 0.0);
        assert!(estimate.aleatoric >= 0.0);
        assert!(estimate.total >= 0.0);
    }

    #[test]
    fn test_calibration() {
        let mut quantifier = UncertaintyQuantifier::new();
        quantifier.add_calibration(0.8, 0.75); // Raw 0.8 -> Calibrated 0.75

        let calibrated = quantifier.calibrate_confidence(0.8);
        assert!((calibrated - 0.75).abs() < 1e-5);
    }

    #[test]
    fn test_quantify_with_unverified_steps() {
        let quantifier = UncertaintyQuantifier::new();
        let chain = ChainOfThought::new("Problem")
            .add_step(ReasoningStep::new(1, "Verified").verify())
            .add_step(ReasoningStep::new(2, "Not verified"))
            .with_confidence(0.7);

        let estimate = quantifier.quantify(&chain);
        // Should have some aleatoric uncertainty due to unverified step
        assert!(estimate.aleatoric > 0.0);
    }
}
