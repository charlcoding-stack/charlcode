// Pattern Matching for Fusion Opportunities
// Detects chains of operations that can be fused

use crate::autograd::Op;
use crate::fusion::FusionType;

/// A pattern that can be fused
#[derive(Debug, Clone, PartialEq)]
pub enum FusionPattern {
    /// Add + Mul: (a + b) * c
    AddMul,

    /// Mul + Add: (a * b) + c (also known as FMA - Fused Multiply-Add)
    MulAdd,

    /// Add + Add: (a + b) + c
    AddAdd,

    /// Mul + Mul: (a * b) * c
    MulMul,

    /// Div + Mul: (a / b) * c
    DivMul,

    /// Custom chain of operations
    Chain(Vec<OpType>),
}

/// Simplified operation type for pattern matching
#[derive(Debug, Clone, PartialEq)]
pub enum OpType {
    Add,
    Sub,
    Mul,
    Div,
    Neg,
    Pow,
}

impl OpType {
    /// Convert from autograd Op to OpType
    pub fn from_op(op: &Op) -> Option<Self> {
        match op {
            Op::Add(_, _) => Some(OpType::Add),
            Op::Sub(_, _) => Some(OpType::Sub),
            Op::Mul(_, _) => Some(OpType::Mul),
            Op::Div(_, _) => Some(OpType::Div),
            Op::Neg(_) => Some(OpType::Neg),
            Op::Pow(_, _) => Some(OpType::Pow),
            _ => None, // Leaf, Sum, MatMul not fusible as element-wise
        }
    }

    /// Check if operation is element-wise (fusible)
    pub fn is_element_wise(&self) -> bool {
        matches!(
            self,
            OpType::Add | OpType::Sub | OpType::Mul | OpType::Div | OpType::Neg | OpType::Pow
        )
    }
}

impl FusionPattern {
    /// Detect pattern from a sequence of operations
    pub fn detect(ops: &[OpType]) -> Option<Self> {
        if ops.len() < 2 {
            return None; // Need at least 2 ops to fuse
        }

        // Check for specific 2-op patterns
        if ops.len() == 2 {
            match (&ops[0], &ops[1]) {
                (OpType::Add, OpType::Mul) => return Some(FusionPattern::AddMul),
                (OpType::Mul, OpType::Add) => return Some(FusionPattern::MulAdd),
                (OpType::Add, OpType::Add) => return Some(FusionPattern::AddAdd),
                (OpType::Mul, OpType::Mul) => return Some(FusionPattern::MulMul),
                (OpType::Div, OpType::Mul) => return Some(FusionPattern::DivMul),
                _ => {}
            }
        }

        // General chain of element-wise ops
        if ops.iter().all(|op| op.is_element_wise()) {
            Some(FusionPattern::Chain(ops.to_vec()))
        } else {
            None
        }
    }

    /// Estimate memory savings from fusion (in bytes)
    pub fn memory_savings(&self, tensor_size: usize) -> usize {
        let element_size = 4; // f32 = 4 bytes

        match self {
            // 2-op patterns save 1 intermediate tensor
            FusionPattern::AddMul
            | FusionPattern::MulAdd
            | FusionPattern::AddAdd
            | FusionPattern::MulMul
            | FusionPattern::DivMul => {
                tensor_size * element_size * 2 // Read + Write
            }

            // Chain saves (n-1) intermediate tensors
            FusionPattern::Chain(ops) => {
                let num_intermediates = ops.len().saturating_sub(1);
                tensor_size * element_size * 2 * num_intermediates
            }
        }
    }

    /// Get the number of operations in this pattern
    pub fn num_ops(&self) -> usize {
        match self {
            FusionPattern::AddMul
            | FusionPattern::MulAdd
            | FusionPattern::AddAdd
            | FusionPattern::MulMul
            | FusionPattern::DivMul => 2,
            FusionPattern::Chain(ops) => ops.len(),
        }
    }
}

/// A detected fusion opportunity in a computational graph
#[derive(Debug, Clone)]
pub struct FusionOpportunity {
    /// The pattern that was detected
    pub pattern: FusionPattern,

    /// IDs of nodes involved in the fusion
    pub node_ids: Vec<usize>,

    /// Type of fusion (vertical/horizontal/element-wise)
    pub fusion_type: FusionType,

    /// Estimated memory savings (bytes)
    pub memory_savings: usize,

    /// Estimated speedup multiplier
    pub estimated_speedup: f64,
}

impl FusionOpportunity {
    /// Create a new fusion opportunity
    pub fn new(pattern: FusionPattern, node_ids: Vec<usize>, tensor_size: usize) -> Self {
        let memory_savings = pattern.memory_savings(tensor_size);

        // Estimate speedup based on memory bandwidth reduction
        // Fusing 2 ops: ~2x speedup (eliminate intermediate memory)
        // Fusing 3 ops: ~2.5x speedup
        // Fusing 4+ ops: ~3x speedup (diminishing returns)
        let num_ops = pattern.num_ops();
        let estimated_speedup = match num_ops {
            2 => 2.0,
            3 => 2.5,
            4 => 3.0,
            _ => 3.5,
        };

        FusionOpportunity {
            pattern,
            node_ids,
            fusion_type: FusionType::Vertical, // Most common
            memory_savings,
            estimated_speedup,
        }
    }

    /// Check if this fusion is worth doing
    pub fn is_beneficial(&self, min_memory_savings: usize) -> bool {
        self.memory_savings >= min_memory_savings
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_op_type_from_op() {
        let add_op = Op::Add(1, 2);
        assert_eq!(OpType::from_op(&add_op), Some(OpType::Add));

        let mul_op = Op::Mul(1, 2);
        assert_eq!(OpType::from_op(&mul_op), Some(OpType::Mul));

        let leaf_op = Op::Leaf;
        assert_eq!(OpType::from_op(&leaf_op), None);
    }

    #[test]
    fn test_op_type_is_element_wise() {
        assert!(OpType::Add.is_element_wise());
        assert!(OpType::Mul.is_element_wise());
        assert!(OpType::Div.is_element_wise());
    }

    #[test]
    fn test_pattern_detect_add_mul() {
        let ops = vec![OpType::Add, OpType::Mul];
        let pattern = FusionPattern::detect(&ops);
        assert_eq!(pattern, Some(FusionPattern::AddMul));
    }

    #[test]
    fn test_pattern_detect_mul_add() {
        let ops = vec![OpType::Mul, OpType::Add];
        let pattern = FusionPattern::detect(&ops);
        assert_eq!(pattern, Some(FusionPattern::MulAdd));
    }

    #[test]
    fn test_pattern_detect_chain() {
        let ops = vec![OpType::Add, OpType::Mul, OpType::Sub];
        let pattern = FusionPattern::detect(&ops);
        assert!(pattern.is_some());

        if let Some(FusionPattern::Chain(detected_ops)) = pattern {
            assert_eq!(detected_ops.len(), 3);
        } else {
            panic!("Expected Chain pattern");
        }
    }

    #[test]
    fn test_pattern_memory_savings() {
        let pattern = FusionPattern::AddMul;
        let tensor_size = 1000; // 1000 elements
        let savings = pattern.memory_savings(tensor_size);

        // 1000 elements * 4 bytes * 2 (read+write) = 8000 bytes
        assert_eq!(savings, 8000);
    }

    #[test]
    fn test_pattern_num_ops() {
        assert_eq!(FusionPattern::AddMul.num_ops(), 2);
        assert_eq!(
            FusionPattern::Chain(vec![OpType::Add, OpType::Mul, OpType::Div]).num_ops(),
            3
        );
    }

    #[test]
    fn test_fusion_opportunity_creation() {
        let pattern = FusionPattern::AddMul;
        let node_ids = vec![1, 2, 3];
        let tensor_size = 10000;

        let opportunity = FusionOpportunity::new(pattern, node_ids.clone(), tensor_size);

        assert_eq!(opportunity.node_ids, node_ids);
        assert_eq!(opportunity.memory_savings, 80000); // 10000 * 4 * 2
        assert_eq!(opportunity.estimated_speedup, 2.0);
    }

    #[test]
    fn test_fusion_opportunity_beneficial() {
        let opportunity = FusionOpportunity::new(FusionPattern::AddMul, vec![1, 2, 3], 10000);

        assert!(opportunity.is_beneficial(1024)); // 80KB > 1KB
        assert!(!opportunity.is_beneficial(100_000)); // 80KB < 100KB
    }

    #[test]
    fn test_estimated_speedup_scaling() {
        let opp2 = FusionOpportunity::new(FusionPattern::AddMul, vec![1, 2], 1000);
        assert_eq!(opp2.estimated_speedup, 2.0);

        let opp3 = FusionOpportunity::new(
            FusionPattern::Chain(vec![OpType::Add, OpType::Mul, OpType::Div]),
            vec![1, 2, 3],
            1000,
        );
        assert_eq!(opp3.estimated_speedup, 2.5);
    }
}
