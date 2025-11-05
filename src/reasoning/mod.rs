// Reasoning Systems Module
//
// This module implements explicit, verifiable reasoning capabilities:
// - Chain-of-Thought: Step-by-step reasoning
// - Working Memory: Cognitive memory systems
// - Tree-of-Thoughts: Multi-path deliberate search
// - Causal Reasoning: Beyond correlation to causation
// - Verification & Critique: Self-verification and uncertainty quantification
//
// Usage:
// ```rust
// use charl::reasoning::{ChainOfThought, ReasoningStep, TreeOfThoughts, SearchStrategy};
//
// // Chain-of-Thought
// let cot = ChainOfThought::new("How many balls does Roger have?")
//     .add_step(ReasoningStep::new(1, "Initial: 5 balls"))
//     .add_step(ReasoningStep::new(2, "Buys 2 cans with 3 balls each"))
//     .add_step(ReasoningStep::new(3, "New balls: 2 × 3 = 6"))
//     .add_step(ReasoningStep::new(4, "Total: 5 + 6 = 11"))
//     .with_final_answer("11 balls");
//
// // Self-Verification
// let verifier = ReasoningVerifier::new();
// let results = verifier.verify_chain(&cot);
//
// // Self-Critique
// let critique_engine = SelfCritique::new();
// let critique = critique_engine.critique(&cot, &results);
//
// // Uncertainty Quantification
// let quantifier = UncertaintyQuantifier::new();
// let uncertainty = quantifier.quantify(&cot);
//
// // Tree-of-Thoughts
// let mut tot = TreeOfThoughts::new("Find 24 from [4,9,10,13]", SearchStrategy::BestFirst);
// let solution = tot.search(|node| {
//     // Expand node into children
//     vec![("4+9=13".to_string(), 0.7), ("10-4=6".to_string(), 0.8)]
// });
//
// // Causal Reasoning
// let mut graph = CausalGraph::new();
// graph.add_variable(Variable::new("study_hours"));
// graph.add_variable(Variable::new("grade"));
// graph.add_edge(CausalEdge::new("study_hours", "grade", 0.8));
//
// // Counterfactual: "What if I had studied 2 more hours?"
// let intervention = Intervention::new("study_hours", 4.0);
// let counterfactual_graph = intervention.apply(&graph);
// ```

pub mod causal;
pub mod chain_of_thought;
pub mod tree_of_thoughts;
pub mod verification;
pub mod working_memory;

// Re-export main types
pub use chain_of_thought::{
    AggregationStrategy, ChainOfThought, LeastToMost, ReasoningStep, SelfConsistency,
};

pub use working_memory::{
    LongTermMemory, MemoryItem, MemoryType, ShortTermMemory, WorkingMemorySystem,
};

pub use tree_of_thoughts::{SearchStrategy, ThoughtNode, TreeOfThoughts};

pub use causal::{CausalEdge, CausalGraph, Confounder, Counterfactual, Intervention, Variable};

pub use verification::{
    Critique, CritiqueQuality, ReasoningVerifier, SelfCritique, UncertaintyEstimate,
    UncertaintyQuantifier, UncertaintyType, VerificationResult, VerificationStatus,
    VerificationType,
};
