// Meta-Learning & Curriculum Learning Module
//
// This module implements meta-learning algorithms that enable models to:
// - Learn from few examples (few-shot learning)
// - Adapt quickly to new tasks (rapid adaptation)
// - Learn better initialization points (MAML, Reptile)
// - Organize training with curriculum strategies
//
// Components:
// - MAML: Model-Agnostic Meta-Learning
// - Prototypical Networks: Distance-based few-shot classification
// - Curriculum Learning: Progressive difficulty scheduling
//
// Usage:
// ```rust
// use charl::meta_learning::{MAML, MetaTask, Episode, PrototypicalNetwork};
//
// // Meta-learning with MAML
// let mut maml = MAML::new(vec![(10, 5)], 0.01, 0.001, 5);
//
// let task = MetaTask::new("task1")
//     .add_support(vec![1.0], vec![2.0])
//     .add_query(vec![2.0], vec![4.0]);
//
// let adapted = maml.adapt(&task, &loss_fn);
//
// // Few-shot classification with Prototypical Networks
// let net = PrototypicalNetwork::new(128, DistanceMetric::Euclidean);
//
// let episode = Episode::new(5, 1) // 5-way 1-shot
//     .add_support(vec![...], 0)
//     .add_query(vec![...], 0);
//
// let accuracy = net.evaluate_episode(&episode, &embed_fn);
//
// // Curriculum learning
// let mut scheduler = CurriculumScheduler::new(
//     CurriculumStrategy::Linear,
//     1000, // total steps
//     1.0,  // progression rate
// );
//
// for step in 0..1000 {
//     scheduler.step(Some(accuracy));
//     let threshold = scheduler.get_threshold();
//     // Train on examples with difficulty <= threshold
// }
// ```

pub mod maml;
pub mod prototypical;
pub mod curriculum;

// Re-export main types
pub use maml::{
    MAML, Reptile, MetaSGD,
    MetaTask, ModelParams,
};

pub use prototypical::{
    PrototypicalNetwork, MatchingNetwork,
    Episode, DistanceMetric,
};

pub use curriculum::{
    CurriculumScheduler, CurriculumStrategy,
    DifficultyScorer, DifficultyMetric,
    SelfPacedLearner, TeacherStudentCurriculum,
    TrainingExample,
};
