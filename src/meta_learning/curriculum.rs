// Curriculum Learning - Training with progressively difficult examples
//
// Curriculum learning mimics human education: start with easy examples,
// gradually increase difficulty. Leads to faster convergence and better generalization.
//
// Strategies:
// 1. Task Difficulty Estimation
// 2. Curriculum Scheduling (easy → difficult)
// 3. Self-paced learning (model chooses its own curriculum)
// 4. Teacher-student curriculum
//
// References:
// - Bengio et al. (2009): "Curriculum Learning"
// - Kumar et al. (2010): "Self-Paced Learning for Latent Variable Models"

use std::collections::HashMap;

/// Training example with features
#[derive(Debug, Clone)]
pub struct TrainingExample {
    pub id: String,
    pub input: Vec<f32>,
    pub target: Vec<f32>,
    pub metadata: HashMap<String, f32>,
}

impl TrainingExample {
    pub fn new(id: impl Into<String>, input: Vec<f32>, target: Vec<f32>) -> Self {
        Self {
            id: id.into(),
            input,
            target,
            metadata: HashMap::new(),
        }
    }

    pub fn with_metadata(mut self, key: impl Into<String>, value: f32) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
}

/// Difficulty estimation methods
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DifficultyMetric {
    /// Based on model's current loss on example
    LossBased,
    /// Based on prediction confidence/uncertainty
    UncertaintyBased,
    /// Based on variance across multiple predictions
    VarianceBased,
    /// Based on manual difficulty labels
    ManualLabels,
    /// Based on example complexity (e.g., length, rarity)
    ComplexityBased,
}

/// Difficulty scorer
pub struct DifficultyScorer {
    pub metric: DifficultyMetric,
    /// Cached difficulty scores
    scores: HashMap<String, f32>,
}

impl DifficultyScorer {
    pub fn new(metric: DifficultyMetric) -> Self {
        Self {
            metric,
            scores: HashMap::new(),
        }
    }

    /// Estimate difficulty of an example
    ///
    /// Returns difficulty score [0, 1] where 0 = easiest, 1 = hardest
    pub fn estimate_difficulty(
        &mut self,
        example: &TrainingExample,
        model_loss: Option<f32>,
    ) -> f32 {
        // Check cache first
        if let Some(&score) = self.scores.get(&example.id) {
            return score;
        }

        let difficulty = match self.metric {
            DifficultyMetric::LossBased => {
                // Higher loss = more difficult
                model_loss.unwrap_or(0.5).min(10.0) / 10.0
            }
            DifficultyMetric::UncertaintyBased => {
                // Use prediction variance as proxy for uncertainty
                if let Some(variance) = example.metadata.get("prediction_variance") {
                    variance.min(1.0)
                } else {
                    0.5
                }
            }
            DifficultyMetric::VarianceBased => {
                // Variance across ensemble predictions
                if let Some(variance) = example.metadata.get("ensemble_variance") {
                    variance.min(1.0)
                } else {
                    0.5
                }
            }
            DifficultyMetric::ManualLabels => {
                // Use pre-assigned difficulty labels
                example.metadata.get("difficulty").copied().unwrap_or(0.5)
            }
            DifficultyMetric::ComplexityBased => {
                // Estimate based on input complexity
                let input_norm: f32 = example.input.iter().map(|x| x.abs()).sum();
                (input_norm / example.input.len() as f32).min(1.0)
            }
        };

        // Cache the score
        self.scores.insert(example.id.clone(), difficulty);
        difficulty
    }

    /// Update difficulty score (e.g., after training iteration)
    pub fn update_difficulty(&mut self, example_id: impl Into<String>, new_score: f32) {
        self.scores
            .insert(example_id.into(), new_score.clamp(0.0, 1.0));
    }

    /// Get all difficulty scores
    pub fn get_scores(&self) -> &HashMap<String, f32> {
        &self.scores
    }
}

/// Curriculum scheduling strategy
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CurriculumStrategy {
    /// Linear progression: linearly increase difficulty threshold
    Linear,
    /// Exponential progression: exponentially increase difficulty
    Exponential,
    /// Step-wise: discrete difficulty steps
    Stepwise,
    /// Adaptive: adjust based on model performance
    Adaptive,
}

/// Curriculum scheduler
pub struct CurriculumScheduler {
    pub strategy: CurriculumStrategy,
    /// Current difficulty threshold [0, 1]
    pub current_threshold: f32,
    /// Maximum difficulty threshold
    pub max_threshold: f32,
    /// Progression rate
    pub progression_rate: f32,
    /// Current training step
    pub current_step: usize,
    /// Total training steps
    pub total_steps: usize,
}

impl CurriculumScheduler {
    /// Create new curriculum scheduler
    pub fn new(strategy: CurriculumStrategy, total_steps: usize, progression_rate: f32) -> Self {
        Self {
            strategy,
            current_threshold: 0.0, // Start with easiest examples
            max_threshold: 1.0,
            progression_rate,
            current_step: 0,
            total_steps,
        }
    }

    /// Update curriculum for next step
    pub fn step(&mut self, performance: Option<f32>) {
        self.current_step += 1;

        self.current_threshold = match self.strategy {
            CurriculumStrategy::Linear => {
                // Linear increase: threshold = step / total_steps
                let progress = self.current_step as f32 / self.total_steps as f32;
                (progress * self.progression_rate).min(self.max_threshold)
            }
            CurriculumStrategy::Exponential => {
                // Exponential increase: threshold = 1 - exp(-k * step)
                let k = self.progression_rate / self.total_steps as f32;
                let progress = 1.0 - (-k * self.current_step as f32).exp();
                (progress * self.max_threshold).min(self.max_threshold)
            }
            CurriculumStrategy::Stepwise => {
                // Discrete steps: jump every N steps
                let step_size = (self.total_steps as f32 / 5.0) as usize; // 5 difficulty levels
                let level = (self.current_step / step_size.max(1)) as f32;
                (level * 0.2 * self.progression_rate).min(self.max_threshold)
            }
            CurriculumStrategy::Adaptive => {
                // Adjust based on performance
                if let Some(perf) = performance {
                    // If performance is good (>0.8), increase difficulty
                    // If performance is poor (<0.5), maintain or decrease difficulty
                    if perf > 0.8 {
                        self.current_threshold + 0.05 * self.progression_rate
                    } else if perf < 0.5 {
                        self.current_threshold - 0.02 * self.progression_rate
                    } else {
                        self.current_threshold
                    }
                } else {
                    self.current_threshold
                }
            }
        };

        self.current_threshold = self.current_threshold.clamp(0.0, self.max_threshold);
    }

    /// Check if example should be included in current curriculum
    pub fn should_include(&self, difficulty: f32) -> bool {
        difficulty <= self.current_threshold
    }

    /// Get current difficulty threshold
    pub fn get_threshold(&self) -> f32 {
        self.current_threshold
    }

    /// Get progress percentage
    pub fn get_progress(&self) -> f32 {
        self.current_step as f32 / self.total_steps as f32
    }
}

/// Self-paced learning: model selects its own curriculum
pub struct SelfPacedLearner {
    /// Current age parameter (controls how many examples to select)
    pub age: f32,
    /// Age increment per step
    pub age_increment: f32,
    /// Difficulty scorer
    pub scorer: DifficultyScorer,
}

impl SelfPacedLearner {
    pub fn new(initial_age: f32, age_increment: f32, metric: DifficultyMetric) -> Self {
        Self {
            age: initial_age,
            age_increment,
            scorer: DifficultyScorer::new(metric),
        }
    }

    /// Select examples for current curriculum
    ///
    /// Self-pacing criterion: select examples with difficulty × weight < age
    pub fn select_examples<'a>(
        &mut self,
        examples: &'a [TrainingExample],
        losses: &HashMap<String, f32>,
    ) -> Vec<&'a TrainingExample> {
        let mut selected = Vec::new();

        for example in examples {
            let loss = losses.get(&example.id).copied();
            let difficulty = self.scorer.estimate_difficulty(example, loss);

            // Self-pacing weight (could be learned, here it's 1.0)
            let weight = 1.0;

            // Select if difficulty × weight < age
            if difficulty * weight < self.age {
                selected.push(example);
            }
        }

        selected
    }

    /// Update age (increase curriculum difficulty)
    pub fn step(&mut self) {
        self.age += self.age_increment;
    }

    /// Get current age
    pub fn get_age(&self) -> f32 {
        self.age
    }
}

/// Teacher-Student curriculum: use teacher model to guide student training
pub struct TeacherStudentCurriculum {
    /// Teacher difficulty threshold (teacher is trained on harder examples)
    pub teacher_threshold: f32,
    /// Student difficulty threshold (starts easier)
    pub student_threshold: f32,
    /// Gap between teacher and student
    pub threshold_gap: f32,
    /// Progression rate for student
    pub progression_rate: f32,
}

impl TeacherStudentCurriculum {
    pub fn new(_initial_gap: f32, progression_rate: f32) -> Self {
        let teacher_threshold = 1.0;
        let student_threshold = 0.0;
        Self {
            teacher_threshold,
            student_threshold,
            threshold_gap: teacher_threshold - student_threshold, // Calculate actual gap
            progression_rate,
        }
    }

    /// Get examples for teacher
    pub fn get_teacher_examples<'a>(
        &self,
        examples: &'a [TrainingExample],
        scorer: &mut DifficultyScorer,
    ) -> Vec<&'a TrainingExample> {
        examples
            .iter()
            .filter(|ex| {
                let difficulty = scorer.estimate_difficulty(ex, None);
                difficulty <= self.teacher_threshold
            })
            .collect()
    }

    /// Get examples for student (guided by teacher)
    pub fn get_student_examples<'a>(
        &self,
        examples: &'a [TrainingExample],
        scorer: &mut DifficultyScorer,
    ) -> Vec<&'a TrainingExample> {
        examples
            .iter()
            .filter(|ex| {
                let difficulty = scorer.estimate_difficulty(ex, None);
                difficulty <= self.student_threshold
            })
            .collect()
    }

    /// Update curriculum (student catches up to teacher)
    pub fn step(&mut self, student_performance: f32) {
        // Increase student threshold based on performance
        if student_performance > 0.7 {
            self.student_threshold += self.progression_rate;
            self.student_threshold = self.student_threshold.min(self.teacher_threshold);
        }

        // Maintain gap between teacher and student
        self.threshold_gap = self.teacher_threshold - self.student_threshold;
    }

    /// Get current gap
    pub fn get_gap(&self) -> f32 {
        self.threshold_gap
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_example_creation() {
        let example =
            TrainingExample::new("ex1", vec![1.0, 2.0], vec![3.0]).with_metadata("difficulty", 0.5);

        assert_eq!(example.id, "ex1");
        assert_eq!(example.input, vec![1.0, 2.0]);
        assert_eq!(example.target, vec![3.0]);
        assert_eq!(example.metadata.get("difficulty"), Some(&0.5));
    }

    #[test]
    fn test_difficulty_scorer_loss_based() {
        let mut scorer = DifficultyScorer::new(DifficultyMetric::LossBased);
        let example = TrainingExample::new("ex1", vec![1.0], vec![2.0]);

        let difficulty = scorer.estimate_difficulty(&example, Some(3.0));
        assert!((difficulty - 0.3).abs() < 1e-5); // 3.0 / 10.0 = 0.3
    }

    #[test]
    fn test_difficulty_scorer_manual() {
        let mut scorer = DifficultyScorer::new(DifficultyMetric::ManualLabels);
        let example =
            TrainingExample::new("ex1", vec![1.0], vec![2.0]).with_metadata("difficulty", 0.8);

        let difficulty = scorer.estimate_difficulty(&example, None);
        assert!((difficulty - 0.8).abs() < 1e-5);
    }

    #[test]
    fn test_difficulty_scorer_cache() {
        let mut scorer = DifficultyScorer::new(DifficultyMetric::ManualLabels);
        let example =
            TrainingExample::new("ex1", vec![1.0], vec![2.0]).with_metadata("difficulty", 0.7);

        // First call
        let diff1 = scorer.estimate_difficulty(&example, None);
        // Second call (should use cache)
        let diff2 = scorer.estimate_difficulty(&example, None);

        assert_eq!(diff1, diff2);
        assert!(scorer.scores.contains_key("ex1"));
    }

    #[test]
    fn test_curriculum_scheduler_linear() {
        let mut scheduler = CurriculumScheduler::new(CurriculumStrategy::Linear, 100, 1.0);

        assert_eq!(scheduler.get_threshold(), 0.0);

        // Step 50 times
        for _ in 0..50 {
            scheduler.step(None);
        }

        // Should be around 0.5
        assert!((scheduler.get_threshold() - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_curriculum_scheduler_exponential() {
        let mut scheduler = CurriculumScheduler::new(CurriculumStrategy::Exponential, 100, 5.0);

        assert_eq!(scheduler.get_threshold(), 0.0);

        scheduler.step(None);
        let threshold1 = scheduler.get_threshold();

        // Step more
        for _ in 0..20 {
            scheduler.step(None);
        }

        let threshold2 = scheduler.get_threshold();

        // Exponential should grow faster than linear
        assert!(threshold2 > threshold1);
        assert!(threshold2 < 1.0);
    }

    #[test]
    fn test_curriculum_scheduler_adaptive() {
        let mut scheduler = CurriculumScheduler::new(CurriculumStrategy::Adaptive, 100, 1.0);

        let initial = scheduler.get_threshold();

        // Good performance → increase difficulty
        scheduler.step(Some(0.9));
        let after_good = scheduler.get_threshold();
        assert!(after_good > initial);

        // Poor performance → decrease difficulty
        scheduler.step(Some(0.3));
        let after_poor = scheduler.get_threshold();
        assert!(after_poor < after_good);
    }

    #[test]
    fn test_curriculum_scheduler_should_include() {
        let scheduler = CurriculumScheduler::new(CurriculumStrategy::Linear, 100, 1.0);

        // At threshold 0.0, only easiest examples
        assert!(scheduler.should_include(0.0));
        assert!(!scheduler.should_include(0.5));
        assert!(!scheduler.should_include(1.0));
    }

    #[test]
    fn test_self_paced_learner() {
        let mut learner = SelfPacedLearner::new(0.3, 0.1, DifficultyMetric::ManualLabels);

        let examples = vec![
            TrainingExample::new("easy", vec![1.0], vec![1.0]).with_metadata("difficulty", 0.1),
            TrainingExample::new("medium", vec![2.0], vec![2.0]).with_metadata("difficulty", 0.5),
            TrainingExample::new("hard", vec![3.0], vec![3.0]).with_metadata("difficulty", 0.9),
        ];

        let losses = HashMap::new();
        let selected = learner.select_examples(&examples, &losses);

        // At age 0.3, should select easy and maybe medium
        assert!(selected.len() > 0);
        assert!(selected.len() <= 2);
    }

    #[test]
    fn test_self_paced_learner_progression() {
        let mut learner = SelfPacedLearner::new(0.2, 0.2, DifficultyMetric::ManualLabels);

        let initial_age = learner.get_age();
        learner.step();
        let after_step = learner.get_age();

        assert!((after_step - (initial_age + 0.2)).abs() < 1e-5);
    }

    #[test]
    fn test_teacher_student_curriculum() {
        let mut curriculum = TeacherStudentCurriculum::new(0.5, 0.1);

        assert_eq!(curriculum.teacher_threshold, 1.0);
        assert_eq!(curriculum.student_threshold, 0.0);
        assert!((curriculum.get_gap() - 1.0).abs() < 1e-5);

        // Student improves
        curriculum.step(0.8);

        // Student threshold should increase
        assert!(curriculum.student_threshold > 0.0);
        // Gap should decrease
        assert!(curriculum.get_gap() < 1.0);
    }

    #[test]
    fn test_teacher_student_filtering() {
        let mut curriculum = TeacherStudentCurriculum::new(0.5, 0.1);
        curriculum.student_threshold = 0.3;

        let mut scorer = DifficultyScorer::new(DifficultyMetric::ManualLabels);

        let examples = vec![
            TrainingExample::new("easy", vec![1.0], vec![1.0]).with_metadata("difficulty", 0.1),
            TrainingExample::new("medium", vec![2.0], vec![2.0]).with_metadata("difficulty", 0.5),
            TrainingExample::new("hard", vec![3.0], vec![3.0]).with_metadata("difficulty", 0.9),
        ];

        let teacher_ex = curriculum.get_teacher_examples(&examples, &mut scorer);
        let student_ex = curriculum.get_student_examples(&examples, &mut scorer);

        // Teacher sees all
        assert_eq!(teacher_ex.len(), 3);
        // Student sees only easy
        assert_eq!(student_ex.len(), 1);
        assert_eq!(student_ex[0].id, "easy");
    }
}
