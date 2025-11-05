// Chain-of-Thought (CoT) Reasoning
//
// Chain-of-Thought prompting enables models to break down complex problems
// into intermediate reasoning steps, significantly improving accuracy.
//
// Key insight: "Let's think step by step"
//
// Example:
//   Problem: "Roger has 5 balls. He buys 2 cans with 3 balls each. How many balls does he have?"
//
//   Without CoT:
//     → 11 (often wrong)
//
//   With CoT:
//     1. Initial: 5 balls
//     2. Buys 2 cans
//     3. Each can has 3 balls
//     4. New balls: 2 × 3 = 6
//     5. Total: 5 + 6 = 11 balls ✓
//
// Variants:
// - Self-Consistency: Generate multiple reasoning chains, take majority vote
// - Least-to-Most: Break problem into subproblems
// - Reasoning Tokens: Dedicated tokens for thinking
//
// References:
// - Wei et al. (2022): "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"
// - Wang et al. (2023): "Self-Consistency Improves Chain of Thought Reasoning"

use std::collections::HashMap;

/// A single reasoning step in a chain
#[derive(Debug, Clone)]
pub struct ReasoningStep {
    /// Step number (for ordering)
    pub step_number: usize,
    /// The thought/reasoning at this step
    pub thought: String,
    /// Optional computation result
    pub computation: Option<f32>,
    /// Whether this step has been verified
    pub verified: bool,
    /// Confidence in this step [0, 1]
    pub confidence: f32,
    /// Any supporting facts or evidence
    pub evidence: Vec<String>,
}

impl ReasoningStep {
    pub fn new(step_number: usize, thought: impl Into<String>) -> Self {
        Self {
            step_number,
            thought: thought.into(),
            computation: None,
            verified: false,
            confidence: 1.0,
            evidence: Vec::new(),
        }
    }

    pub fn with_computation(mut self, computation: f32) -> Self {
        self.computation = Some(computation);
        self
    }

    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    pub fn with_evidence(mut self, evidence: impl Into<String>) -> Self {
        self.evidence.push(evidence.into());
        self
    }

    pub fn verify(mut self) -> Self {
        self.verified = true;
        self
    }
}

/// Complete chain of thought reasoning
#[derive(Debug, Clone)]
pub struct ChainOfThought {
    /// Problem or question being reasoned about
    pub problem: String,
    /// Sequence of reasoning steps
    pub steps: Vec<ReasoningStep>,
    /// Final answer
    pub final_answer: String,
    /// Overall confidence [0, 1]
    pub confidence: f32,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

impl ChainOfThought {
    pub fn new(problem: impl Into<String>) -> Self {
        Self {
            problem: problem.into(),
            steps: Vec::new(),
            final_answer: String::new(),
            confidence: 1.0,
            metadata: HashMap::new(),
        }
    }

    pub fn add_step(mut self, step: ReasoningStep) -> Self {
        self.steps.push(step);
        self
    }

    pub fn with_final_answer(mut self, answer: impl Into<String>) -> Self {
        self.final_answer = answer.into();
        self
    }

    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Compute overall confidence (min of all step confidences)
    pub fn compute_confidence(&mut self) {
        if self.steps.is_empty() {
            self.confidence = 0.0;
            return;
        }

        self.confidence = self.steps
            .iter()
            .map(|s| s.confidence)
            .fold(f32::INFINITY, f32::min);
    }

    /// Check if all steps are verified
    pub fn is_fully_verified(&self) -> bool {
        !self.steps.is_empty() && self.steps.iter().all(|s| s.verified)
    }

    /// Get number of steps
    pub fn num_steps(&self) -> usize {
        self.steps.len()
    }
}

/// Self-Consistency: Generate multiple reasoning chains and aggregate
pub struct SelfConsistency {
    /// Number of reasoning chains to generate
    pub num_chains: usize,
    /// Aggregation strategy
    pub aggregation: AggregationStrategy,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AggregationStrategy {
    /// Take most common answer (majority vote)
    MajorityVote,
    /// Take highest confidence answer
    MaxConfidence,
    /// Weight by confidence and take weighted majority
    WeightedVote,
}

impl SelfConsistency {
    pub fn new(num_chains: usize) -> Self {
        Self {
            num_chains,
            aggregation: AggregationStrategy::MajorityVote,
        }
    }

    pub fn with_aggregation(mut self, strategy: AggregationStrategy) -> Self {
        self.aggregation = strategy;
        self
    }

    /// Aggregate multiple reasoning chains into final answer
    pub fn aggregate(&self, chains: &[ChainOfThought]) -> String {
        if chains.is_empty() {
            return String::new();
        }

        match self.aggregation {
            AggregationStrategy::MajorityVote => {
                // Count occurrences of each answer
                let mut counts: HashMap<String, usize> = HashMap::new();
                for chain in chains {
                    *counts.entry(chain.final_answer.clone()).or_insert(0) += 1;
                }

                // Return most common
                counts
                    .into_iter()
                    .max_by_key(|(_, count)| *count)
                    .map(|(answer, _)| answer)
                    .unwrap_or_default()
            }
            AggregationStrategy::MaxConfidence => {
                // Return answer with highest confidence
                chains
                    .iter()
                    .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap())
                    .map(|chain| chain.final_answer.clone())
                    .unwrap_or_default()
            }
            AggregationStrategy::WeightedVote => {
                // Weight answers by confidence
                let mut weighted_counts: HashMap<String, f32> = HashMap::new();
                for chain in chains {
                    *weighted_counts
                        .entry(chain.final_answer.clone())
                        .or_insert(0.0) += chain.confidence;
                }

                weighted_counts
                    .into_iter()
                    .max_by(|(_, w1), (_, w2)| w1.partial_cmp(w2).unwrap())
                    .map(|(answer, _)| answer)
                    .unwrap_or_default()
            }
        }
    }

    /// Compute aggregated confidence
    pub fn aggregate_confidence(&self, chains: &[ChainOfThought]) -> f32 {
        if chains.is_empty() {
            return 0.0;
        }

        match self.aggregation {
            AggregationStrategy::MajorityVote => {
                // Confidence = fraction of chains agreeing with majority
                let majority_answer = self.aggregate(chains);
                let agreeing = chains
                    .iter()
                    .filter(|c| c.final_answer == majority_answer)
                    .count();
                agreeing as f32 / chains.len() as f32
            }
            AggregationStrategy::MaxConfidence => {
                // Confidence = max confidence among chains
                chains
                    .iter()
                    .map(|c| c.confidence)
                    .fold(0.0, f32::max)
            }
            AggregationStrategy::WeightedVote => {
                // Confidence = weighted sum / total weight
                let majority_answer = self.aggregate(chains);
                let weighted_sum: f32 = chains
                    .iter()
                    .filter(|c| c.final_answer == majority_answer)
                    .map(|c| c.confidence)
                    .sum();
                let total_weight: f32 = chains.iter().map(|c| c.confidence).sum();
                if total_weight > 0.0 {
                    weighted_sum / total_weight
                } else {
                    0.0
                }
            }
        }
    }
}

/// Least-to-Most Prompting: Break complex problems into simpler subproblems
pub struct LeastToMost {
    /// Maximum subproblem depth
    pub max_depth: usize,
}

impl LeastToMost {
    pub fn new(max_depth: usize) -> Self {
        Self { max_depth }
    }

    /// Decompose problem into subproblems
    pub fn decompose(&self, problem: &str) -> Vec<String> {
        // Simplified decomposition (in production, use LLM)
        // Example: "What is 15 + 23 × 4?"
        // → ["What is 23 × 4?", "What is 15 + [result]?"]

        let mut subproblems = Vec::new();

        // Simple heuristic: split by keywords
        if problem.contains(" and ") {
            let parts: Vec<&str> = problem.split(" and ").collect();
            for part in parts {
                subproblems.push(part.trim().to_string());
            }
        } else if problem.contains("×") || problem.contains("*") {
            // Multiplication first
            subproblems.push(format!("Compute multiplication in: {}", problem));
            subproblems.push(format!("Complete the calculation"));
        } else {
            // Cannot decompose further
            subproblems.push(problem.to_string());
        }

        subproblems
    }

    /// Solve problem using least-to-most decomposition
    pub fn solve(&self, problem: &str, depth: usize) -> ChainOfThought {
        let mut cot = ChainOfThought::new(problem);

        if depth >= self.max_depth {
            // Base case: solve directly
            cot = cot.add_step(ReasoningStep::new(0, format!("Base case: {}", problem)));
            return cot.with_final_answer("(base case)");
        }

        // Decompose
        let subproblems = self.decompose(problem);

        // Solve subproblems
        for (i, subproblem) in subproblems.iter().enumerate() {
            let step = ReasoningStep::new(i, format!("Subproblem {}: {}", i + 1, subproblem));
            cot = cot.add_step(step);
        }

        cot = cot.with_final_answer(format!("Solved {} subproblems", subproblems.len()));
        cot
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reasoning_step_creation() {
        let step = ReasoningStep::new(1, "Initial: 5 balls")
            .with_computation(5.0)
            .with_confidence(0.9)
            .with_evidence("Given in problem statement");

        assert_eq!(step.step_number, 1);
        assert_eq!(step.thought, "Initial: 5 balls");
        assert_eq!(step.computation, Some(5.0));
        assert_eq!(step.confidence, 0.9);
        assert!(!step.verified);
        assert_eq!(step.evidence.len(), 1);
    }

    #[test]
    fn test_reasoning_step_verification() {
        let step = ReasoningStep::new(1, "2 + 2 = 4").verify();
        assert!(step.verified);
    }

    #[test]
    fn test_chain_of_thought_creation() {
        let cot = ChainOfThought::new("How many balls does Roger have?")
            .add_step(ReasoningStep::new(1, "Initial: 5 balls"))
            .add_step(ReasoningStep::new(2, "Buys 2 cans"))
            .add_step(ReasoningStep::new(3, "Each can has 3 balls"))
            .add_step(ReasoningStep::new(4, "New balls: 2 × 3 = 6").with_computation(6.0))
            .add_step(ReasoningStep::new(5, "Total: 5 + 6 = 11").with_computation(11.0))
            .with_final_answer("11 balls");

        assert_eq!(cot.num_steps(), 5);
        assert_eq!(cot.final_answer, "11 balls");
    }

    #[test]
    fn test_compute_confidence() {
        let mut cot = ChainOfThought::new("Test problem")
            .add_step(ReasoningStep::new(1, "Step 1").with_confidence(0.9))
            .add_step(ReasoningStep::new(2, "Step 2").with_confidence(0.8))
            .add_step(ReasoningStep::new(3, "Step 3").with_confidence(0.95));

        cot.compute_confidence();

        // Should be minimum of all steps
        assert!((cot.confidence - 0.8).abs() < 1e-5);
    }

    #[test]
    fn test_is_fully_verified() {
        let cot1 = ChainOfThought::new("Problem 1")
            .add_step(ReasoningStep::new(1, "Step 1").verify())
            .add_step(ReasoningStep::new(2, "Step 2").verify());

        assert!(cot1.is_fully_verified());

        let cot2 = ChainOfThought::new("Problem 2")
            .add_step(ReasoningStep::new(1, "Step 1").verify())
            .add_step(ReasoningStep::new(2, "Step 2")); // Not verified

        assert!(!cot2.is_fully_verified());
    }

    #[test]
    fn test_self_consistency_majority_vote() {
        let sc = SelfConsistency::new(5).with_aggregation(AggregationStrategy::MajorityVote);

        let chains = vec![
            ChainOfThought::new("Problem").with_final_answer("42"),
            ChainOfThought::new("Problem").with_final_answer("42"),
            ChainOfThought::new("Problem").with_final_answer("42"),
            ChainOfThought::new("Problem").with_final_answer("41"),
            ChainOfThought::new("Problem").with_final_answer("43"),
        ];

        let answer = sc.aggregate(&chains);
        assert_eq!(answer, "42"); // Majority

        let confidence = sc.aggregate_confidence(&chains);
        assert!((confidence - 0.6).abs() < 1e-5); // 3/5 agree
    }

    #[test]
    fn test_self_consistency_max_confidence() {
        let sc = SelfConsistency::new(3).with_aggregation(AggregationStrategy::MaxConfidence);

        let chains = vec![
            ChainOfThought::new("Problem")
                .with_final_answer("41")
                .with_confidence(0.7),
            ChainOfThought::new("Problem")
                .with_final_answer("42")
                .with_confidence(0.95),
            ChainOfThought::new("Problem")
                .with_final_answer("43")
                .with_confidence(0.8),
        ];

        let answer = sc.aggregate(&chains);
        assert_eq!(answer, "42"); // Highest confidence

        let confidence = sc.aggregate_confidence(&chains);
        assert!((confidence - 0.95).abs() < 1e-5);
    }

    #[test]
    fn test_self_consistency_weighted_vote() {
        let sc = SelfConsistency::new(3).with_aggregation(AggregationStrategy::WeightedVote);

        let chains = vec![
            ChainOfThought::new("Problem")
                .with_final_answer("42")
                .with_confidence(0.9),
            ChainOfThought::new("Problem")
                .with_final_answer("42")
                .with_confidence(0.8),
            ChainOfThought::new("Problem")
                .with_final_answer("41")
                .with_confidence(0.5),
        ];

        let answer = sc.aggregate(&chains);
        assert_eq!(answer, "42"); // Weighted majority (0.9 + 0.8 > 0.5)
    }

    #[test]
    fn test_least_to_most_decomposition() {
        let ltm = LeastToMost::new(3);

        let problem1 = "What is 2 + 3 and 4 × 5?";
        let subproblems1 = ltm.decompose(problem1);
        assert_eq!(subproblems1.len(), 2);
        assert!(subproblems1[0].contains("2 + 3"));
        assert!(subproblems1[1].contains("4 × 5"));

        let problem2 = "What is 15 + 23 × 4?";
        let subproblems2 = ltm.decompose(problem2);
        assert!(subproblems2.len() >= 1);
    }

    #[test]
    fn test_least_to_most_solve() {
        let ltm = LeastToMost::new(2);
        let problem = "What is 2 + 3 and 4 × 5?";

        let cot = ltm.solve(problem, 0);

        assert_eq!(cot.problem, problem);
        assert!(cot.num_steps() >= 1);
    }

    #[test]
    fn test_least_to_most_max_depth() {
        let ltm = LeastToMost::new(1);
        let problem = "Complex problem";

        // At max depth, should create base case
        let cot = ltm.solve(problem, 1);
        assert!(cot.steps.len() >= 1);
    }
}
