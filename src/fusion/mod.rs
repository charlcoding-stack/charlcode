// Kernel Fusion - Phase 10
// Fuses consecutive operations to reduce memory bandwidth and increase performance
//
// Key concepts:
// - Vertical fusion: Operations in sequence (a + b) * c → fused_add_mul(a, b, c)
// - Horizontal fusion: Independent operations executed in parallel
// - Memory bandwidth reduction: Eliminate intermediate reads/writes
//
// Expected speedup: 2-4x for chains of element-wise operations

pub mod optimizer;
pub mod patterns;

#[cfg(feature = "llvm")]
pub mod llvm_fusion;

pub use optimizer::FusionOptimizer;
pub use patterns::{FusionOpportunity, FusionPattern};

/// Types of fusion opportunities
#[derive(Debug, Clone, PartialEq)]
pub enum FusionType {
    /// Vertical fusion: operations in sequence
    /// Example: (a + b) * c → single kernel
    Vertical,

    /// Horizontal fusion: independent operations
    /// Example: y1 = a + b, y2 = c + d → single kernel with 2 outputs
    Horizontal,

    /// Element-wise fusion: chain of element-wise ops
    /// Example: relu(sigmoid(x * w + b)) → single fused kernel
    ElementWise,
}

/// Fusion strategy configuration
#[derive(Debug, Clone)]
pub struct FusionConfig {
    /// Enable vertical fusion
    pub enable_vertical: bool,

    /// Enable horizontal fusion
    pub enable_horizontal: bool,

    /// Maximum number of operations to fuse
    pub max_ops_per_fusion: usize,

    /// Minimum memory savings to trigger fusion (bytes)
    pub min_memory_savings: usize,
}

impl Default for FusionConfig {
    fn default() -> Self {
        FusionConfig {
            enable_vertical: true,
            enable_horizontal: false, // More complex, disable by default
            max_ops_per_fusion: 5,
            min_memory_savings: 1024, // 1KB minimum
        }
    }
}

impl FusionConfig {
    /// Aggressive fusion strategy
    pub fn aggressive() -> Self {
        FusionConfig {
            enable_vertical: true,
            enable_horizontal: true,
            max_ops_per_fusion: 10,
            min_memory_savings: 0, // Always fuse
        }
    }

    /// Conservative fusion strategy
    pub fn conservative() -> Self {
        FusionConfig {
            enable_vertical: true,
            enable_horizontal: false,
            max_ops_per_fusion: 3,
            min_memory_savings: 4096, // 4KB minimum
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fusion_config_default() {
        let config = FusionConfig::default();
        assert!(config.enable_vertical);
        assert!(!config.enable_horizontal);
        assert_eq!(config.max_ops_per_fusion, 5);
    }

    #[test]
    fn test_fusion_config_aggressive() {
        let config = FusionConfig::aggressive();
        assert!(config.enable_vertical);
        assert!(config.enable_horizontal);
        assert_eq!(config.max_ops_per_fusion, 10);
        assert_eq!(config.min_memory_savings, 0);
    }

    #[test]
    fn test_fusion_config_conservative() {
        let config = FusionConfig::conservative();
        assert!(config.enable_vertical);
        assert!(!config.enable_horizontal);
        assert_eq!(config.max_ops_per_fusion, 3);
        assert_eq!(config.min_memory_savings, 4096);
    }
}
