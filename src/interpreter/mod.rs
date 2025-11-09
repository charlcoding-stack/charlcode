// Interpreter module - Execute Charl AST
// Phase 3: Tree-walking interpreter

use crate::ast::*;
use crate::autograd::{ComputationGraph, Tensor as AutogradTensor};
use crate::stdlib::{self, BuiltinFn};
use crate::tensor_builtins;
use std::collections::HashMap;

// Runtime value representation
#[derive(Debug, Clone)]
pub enum Value {
    Integer(i64),
    Float(f64),
    Boolean(bool),
    String(String),
    Array(Vec<Value>),
    Tensor {
        data: Vec<Value>,
        shape: Vec<usize>,
    },
    AutogradTensor(AutogradTensor), // Tensor with gradient tracking (CPU)
    GPUTensor(crate::gpu_tensor::GPUTensor), // GPU-accelerated tensor
    LinearLayer(Box<crate::nn::gpu_layers::Linear>), // Linear/Dense layer (v0.2.0)
    Conv2dLayer(Box<crate::nn::gpu_layers::Conv2d>), // Conv2d layer for CNNs (v0.2.0)
    MaxPool2dLayer(Box<crate::nn::gpu_layers::MaxPool2d>), // MaxPool2d for downsampling (v0.2.0)
    AvgPool2dLayer(Box<crate::nn::gpu_layers::AvgPool2d>), // AvgPool2d for downsampling (v0.2.0)
    BatchNormLayer(Box<crate::nn::gpu_layers::BatchNorm>), // BatchNorm for training stability (v0.2.0)
    LayerNormLayer(Box<crate::nn::gpu_layers::LayerNorm>), // LayerNorm for Transformers (v0.2.0)
    DropoutLayer(Box<crate::nn::gpu_layers::Dropout>), // Dropout for regularization (v0.2.0)
    LSTM(Box<crate::nn::LSTM>), // LSTM recurrent layer
    GRU(Box<crate::nn::GRU>), // GRU recurrent layer
    SGDOptimizer(Box<crate::optim::SGD>), // SGD optimizer (Week 5-6)
    AdamOptimizer(Box<crate::optim::Adam>), // Adam optimizer (Week 5-6)
    RMSpropOptimizer(Box<crate::optim::RMSprop>), // RMSprop optimizer (Week 5-6)
    LinformerLayer(Box<crate::efficient_architectures::Linformer>), // Linformer: O(n) attention (Week 7-8)
    PerformerLayer(Box<crate::efficient_architectures::Performer>), // Performer: FAVOR+ attention (Week 7-8)
    FNetLayer(Box<crate::efficient_architectures::FNet>), // FNet: Fourier mixing (Week 7-8)
    RWKVLayer(Box<crate::efficient_architectures::RWKV>), // RWKV: Receptance Weighted KV (Week 7-8)
    MambaLayer(Box<crate::efficient_architectures::MambaBlock>), // Mamba: Selective SSM (Week 7-8)
    S4Layer(Box<crate::efficient_architectures::S4Layer>), // S4: Structured State Spaces (Week 7-8)
    MoELayer(Box<crate::efficient_architectures::MoELayer>), // MoE: Mixture of Experts (Week 7-8)
    QuantizedTensor(Box<crate::quantization::QuantizedTensor>), // Quantized tensor (Week 7-8 Part 2)
    ChainOfThought(Box<crate::reasoning::ChainOfThought>), // Chain-of-Thought reasoning (Week 7-8 Part 3)
    FusionOptimizer(Box<crate::fusion::FusionOptimizer>), // Operator fusion optimizer (Week 15-16)
    FusionStats(Box<crate::fusion::optimizer::FusionStats>), // Fusion statistics (Week 15-16)
    // Week 17-19: Multimodal AI
    CLIPEncoder(Box<crate::multimodal::vision_language::CLIPEncoder>), // CLIP encoder for vision-language
    Image(Box<crate::multimodal::vision_language::Image>), // Image representation
    MultimodalEmbedding(Box<crate::multimodal::vision_language::MultimodalEmbedding>), // Multimodal embedding
    VQASystem(Box<crate::multimodal::vision_language::VQASystem>), // Visual Question Answering system
    VQAAnswer(Box<crate::multimodal::vision_language::VQAAnswer>), // VQA answer with confidence
    CrossModalRetrieval(Box<crate::multimodal::vision_language::CrossModalRetrieval>), // Cross-modal retrieval
    SceneObject(Box<crate::multimodal::scene_understanding::SceneObject>), // Scene object
    SceneGraph(Box<crate::multimodal::scene_understanding::SceneGraph>), // Scene graph
    SceneGraphGenerator(Box<crate::multimodal::scene_understanding::SceneGraphGenerator>), // Scene graph generator
    TemporalEvent(Box<crate::multimodal::scene_understanding::TemporalEvent>), // Temporal event
    MultimodalCoT(Box<crate::multimodal::cross_modal_reasoning::MultimodalCoT>), // Multimodal Chain-of-Thought
    VisualGrounding(Box<crate::multimodal::cross_modal_reasoning::VisualGrounding>), // Visual grounding
    MultimodalReasoner(Box<crate::multimodal::cross_modal_reasoning::MultimodalReasoner>), // Multimodal reasoner
    // Week 20-21: Meta-Learning
    MetaTask(Box<crate::meta_learning::maml::MetaTask>), // Meta-learning task
    ModelParams(Box<crate::meta_learning::maml::ModelParams>), // Model parameters
    MAML(Box<crate::meta_learning::maml::MAML>), // MAML meta-learner
    Episode(Box<crate::meta_learning::prototypical::Episode>), // Few-shot episode
    PrototypicalNetwork(Box<crate::meta_learning::prototypical::PrototypicalNetwork>), // Prototypical network
    // Week 22-24: Knowledge Graphs & GNN
    KGEntity(Box<crate::knowledge_graph::Entity>), // Knowledge graph entity
    KGTriple(Box<crate::knowledge_graph::Triple>), // Knowledge graph triple (fact)
    KnowledgeGraph(Box<crate::knowledge_graph::KnowledgeGraph>), // Knowledge graph
    GraphStats(Box<crate::knowledge_graph::GraphStats>), // Graph statistics
    GraphNeuralNetwork(Box<crate::knowledge_graph::GraphNeuralNetwork>), // Graph neural network
    NodeEmbeddings(Box<std::collections::HashMap<crate::knowledge_graph::EntityId, Vec<f64>>>), // Node embeddings
    // Week 25-26: Advanced Reasoning
    ReasoningStep(Box<crate::reasoning::ReasoningStep>), // Single reasoning step
    TreeOfThoughts(Box<crate::reasoning::TreeOfThoughts>), // Tree-of-Thoughts reasoner
    ThoughtNode(Box<crate::reasoning::ThoughtNode>), // Thought node in tree
    MemoryItem(Box<crate::reasoning::MemoryItem>), // Memory item
    ShortTermMemory(Box<crate::reasoning::ShortTermMemory>), // Short-term memory
    LongTermMemory(Box<crate::reasoning::LongTermMemory>), // Long-term memory
    WorkingMemorySystem(Box<crate::reasoning::WorkingMemorySystem>), // Complete memory system
    // Week 29-30: Symbolic AI
    SymbolicRule(Box<crate::symbolic::Rule>), // Symbolic rule
    RuleEngine(Box<crate::symbolic::RuleEngine>), // Rule engine
    FOLTerm(Box<crate::symbolic::Term>), // First-order logic term
    FOLFormula(Box<crate::symbolic::Formula>), // FOL formula
    FOLSolver(Box<crate::symbolic::FOLSolver>), // FOL solver
    FuzzyValue(Box<crate::symbolic::FuzzyValue>), // Fuzzy logic value
    Concept(Box<crate::symbolic::Concept>), // Concept in concept learning
    ConceptGraph(Box<crate::symbolic::ConceptGraph>), // Concept graph
    // Linear Attention variants
    Linformer(Box<crate::efficient_architectures::linear_attention::Linformer>),
    Performer(Box<crate::efficient_architectures::linear_attention::Performer>),
    FNet(Box<crate::efficient_architectures::linear_attention::FNet>),
    RWKV(Box<crate::efficient_architectures::linear_attention::RWKV>),
    Function {
        parameters: Vec<Parameter>,
        body: Vec<Statement>,
        closure: Environment,
    },
    Tuple(Vec<Value>),
    Null,
}

// Manual PartialEq implementation since AutogradTensor doesn't implement it
impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Value::Integer(a), Value::Integer(b)) => a == b,
            (Value::Float(a), Value::Float(b)) => a == b,
            (Value::Boolean(a), Value::Boolean(b)) => a == b,
            (Value::String(a), Value::String(b)) => a == b,
            (Value::Array(a), Value::Array(b)) => a == b,
            (
                Value::Tensor {
                    data: d1,
                    shape: s1,
                },
                Value::Tensor {
                    data: d2,
                    shape: s2,
                },
            ) => d1 == d2 && s1 == s2,
            (Value::AutogradTensor(a), Value::AutogradTensor(b)) => {
                a.data == b.data && a.shape == b.shape
            }
            (Value::GPUTensor(a), Value::GPUTensor(b)) => {
                a.tensor.data == b.tensor.data && a.tensor.shape == b.tensor.shape
            }
            (Value::Function { .. }, Value::Function { .. }) => false, // Functions are never equal
            (Value::Tuple(a), Value::Tuple(b)) => a == b,
            (Value::Null, Value::Null) => true,
            _ => false,
        }
    }
}

impl Value {
    pub fn type_name(&self) -> &str {
        match self {
            Value::Integer(_) => "integer",
            Value::Float(_) => "float",
            Value::Boolean(_) => "boolean",
            Value::String(_) => "string",
            Value::Array(_) => "array",
            Value::Tensor { .. } => "tensor",
            Value::AutogradTensor(_) => "autograd_tensor",
            Value::GPUTensor(_) => "gpu_tensor",
            Value::LinearLayer(_) => "linear_layer",
            Value::Conv2dLayer(_) => "conv2d_layer",
            Value::MaxPool2dLayer(_) => "maxpool2d_layer",
            Value::AvgPool2dLayer(_) => "avgpool2d_layer",
            Value::BatchNormLayer(_) => "batchnorm_layer",
            Value::LayerNormLayer(_) => "layernorm_layer",
            Value::DropoutLayer(_) => "dropout_layer",
            Value::SGDOptimizer(_) => "sgd_optimizer",
            Value::AdamOptimizer(_) => "adam_optimizer",
            Value::RMSpropOptimizer(_) => "rmsprop_optimizer",
            Value::LinformerLayer(_) => "linformer_layer",
            Value::PerformerLayer(_) => "performer_layer",
            Value::FNetLayer(_) => "fnet_layer",
            Value::RWKVLayer(_) => "rwkv_layer",
            Value::MambaLayer(_) => "mamba_layer",
            Value::S4Layer(_) => "s4_layer",
            Value::MoELayer(_) => "moe_layer",
            Value::QuantizedTensor(_) => "quantized_tensor",
            Value::ChainOfThought(_) => "chain_of_thought",
            Value::FusionOptimizer(_) => "fusion_optimizer",
            Value::FusionStats(_) => "fusion_stats",
            Value::CLIPEncoder(_) => "clip_encoder",
            Value::Image(_) => "image",
            Value::MultimodalEmbedding(_) => "multimodal_embedding",
            Value::VQASystem(_) => "vqa_system",
            Value::VQAAnswer(_) => "vqa_answer",
            Value::CrossModalRetrieval(_) => "cross_modal_retrieval",
            Value::SceneObject(_) => "scene_object",
            Value::SceneGraph(_) => "scene_graph",
            Value::SceneGraphGenerator(_) => "scene_graph_generator",
            Value::TemporalEvent(_) => "temporal_event",
            Value::MultimodalCoT(_) => "multimodal_cot",
            Value::VisualGrounding(_) => "visual_grounding",
            Value::MultimodalReasoner(_) => "multimodal_reasoner",
            Value::MetaTask(_) => "meta_task",
            Value::ModelParams(_) => "model_params",
            Value::MAML(_) => "maml",
            Value::Episode(_) => "episode",
            Value::PrototypicalNetwork(_) => "prototypical_network",
            Value::KGEntity(_) => "kg_entity",
            Value::KGTriple(_) => "kg_triple",
            Value::KnowledgeGraph(_) => "knowledge_graph",
            Value::GraphStats(_) => "graph_stats",
            Value::GraphNeuralNetwork(_) => "graph_neural_network",
            Value::NodeEmbeddings(_) => "node_embeddings",
            Value::ReasoningStep(_) => "reasoning_step",
            Value::TreeOfThoughts(_) => "tree_of_thoughts",
            Value::ThoughtNode(_) => "thought_node",
            Value::MemoryItem(_) => "memory_item",
            Value::ShortTermMemory(_) => "short_term_memory",
            Value::LongTermMemory(_) => "long_term_memory",
            Value::WorkingMemorySystem(_) => "working_memory_system",
            Value::SymbolicRule(_) => "symbolic_rule",
            Value::RuleEngine(_) => "rule_engine",
            Value::FOLTerm(_) => "fol_term",
            Value::FOLFormula(_) => "fol_formula",
            Value::FOLSolver(_) => "fol_solver",
            Value::FuzzyValue(_) => "fuzzy_value",
            Value::Concept(_) => "concept",
            Value::ConceptGraph(_) => "concept_graph",
            Value::Linformer(_) => "linformer",
            Value::Performer(_) => "performer",
            Value::FNet(_) => "fnet",
            Value::RWKV(_) => "rwkv",
            Value::LSTM(_) => "lstm",
            Value::GRU(_) => "gru",
            Value::Function { .. } => "function",
            Value::Tuple(_) => "tuple",
            Value::Null => "null",
        }
    }

    pub fn is_truthy(&self) -> bool {
        match self {
            Value::Boolean(b) => *b,
            Value::Null => false,
            Value::Integer(i) => *i != 0,
            Value::Float(f) => *f != 0.0,
            _ => true,
        }
    }

    // Convert to float for numeric operations
    pub fn to_float(&self) -> Result<f64, String> {
        match self {
            Value::Integer(i) => Ok(*i as f64),
            Value::Float(f) => Ok(*f),
            _ => Err(format!("Cannot convert {} to float", self.type_name())),
        }
    }

    // Convert to integer for numeric operations
    pub fn to_integer(&self) -> Result<i64, String> {
        match self {
            Value::Integer(i) => Ok(*i),
            Value::Float(f) => Ok(*f as i64),
            _ => Err(format!("Cannot convert {} to integer", self.type_name())),
        }
    }
}

// Environment for variable storage with scope management
#[derive(Debug, Clone, PartialEq)]
pub struct Environment {
    scopes: Vec<HashMap<String, Value>>,
}

impl Default for Environment {
    fn default() -> Self {
        Self::new()
    }
}

impl Environment {
    pub fn new() -> Self {
        Environment {
            scopes: vec![HashMap::new()],
        }
    }

    pub fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    pub fn pop_scope(&mut self) {
        if self.scopes.len() > 1 {
            self.scopes.pop();
        }
    }

    pub fn set(&mut self, name: String, value: Value) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name, value);
        }
    }

    /// Update an existing variable in the scope where it was declared
    /// Returns true if the variable was found and updated, false otherwise
    pub fn update(&mut self, name: &str, value: Value) -> bool {
        // Search from innermost to outermost scope
        for scope in self.scopes.iter_mut().rev() {
            if scope.contains_key(name) {
                scope.insert(name.to_string(), value);
                return true;
            }
        }
        false
    }

    pub fn get(&self, name: &str) -> Option<&Value> {
        // Search from innermost to outermost scope
        for scope in self.scopes.iter().rev() {
            if let Some(value) = scope.get(name) {
                return Some(value);
            }
        }
        None
    }
}

pub struct Interpreter {
    env: Environment,
    return_value: Option<Value>,
    break_loop: bool,     // Flag for break statement
    continue_loop: bool,  // Flag for continue statement
    graph: ComputationGraph,
    builtins: HashMap<String, BuiltinFn>,
}

impl Default for Interpreter {
    fn default() -> Self {
        Self::new()
    }
}

impl Interpreter {
    pub fn new() -> Self {
        let mut builtins = HashMap::new();

        // Register built-in functions
        builtins.insert("print".to_string(), stdlib::builtin_print as BuiltinFn);
        builtins.insert("len".to_string(), stdlib::builtin_len as BuiltinFn);
        builtins.insert("range".to_string(), stdlib::builtin_range as BuiltinFn);
        builtins.insert("push".to_string(), stdlib::builtin_push as BuiltinFn);
        builtins.insert("pop".to_string(), stdlib::builtin_pop as BuiltinFn);
        builtins.insert("type".to_string(), stdlib::builtin_type as BuiltinFn);
        builtins.insert("str".to_string(), stdlib::builtin_str as BuiltinFn);
        builtins.insert("assert".to_string(), stdlib::builtin_assert as BuiltinFn);

        // Register tensor built-in functions (Phase 1)
        // Basic tensor operations
        builtins.insert("tensor".to_string(), tensor_builtins::builtin_tensor as BuiltinFn);
        builtins.insert("tensor_shape".to_string(), tensor_builtins::builtin_tensor_shape as BuiltinFn);
        builtins.insert("tensor_size".to_string(), tensor_builtins::builtin_tensor_size as BuiltinFn);
        builtins.insert("tensor_print".to_string(), tensor_builtins::builtin_tensor_print as BuiltinFn);

        // Arithmetic operations
        builtins.insert("tensor_add".to_string(), tensor_builtins::builtin_tensor_add as BuiltinFn);
        builtins.insert("tensor_sub".to_string(), tensor_builtins::builtin_tensor_sub as BuiltinFn);
        builtins.insert("tensor_mul".to_string(), tensor_builtins::builtin_tensor_mul as BuiltinFn);
        builtins.insert("tensor_div".to_string(), tensor_builtins::builtin_tensor_div as BuiltinFn);
        builtins.insert("tensor_matmul".to_string(), tensor_builtins::builtin_tensor_matmul as BuiltinFn);

        // Reduction operations
        builtins.insert("tensor_sum".to_string(), tensor_builtins::builtin_tensor_sum as BuiltinFn);
        builtins.insert("tensor_mean".to_string(), tensor_builtins::builtin_tensor_mean as BuiltinFn);
        builtins.insert("tensor_max".to_string(), tensor_builtins::builtin_tensor_max as BuiltinFn);
        builtins.insert("tensor_min".to_string(), tensor_builtins::builtin_tensor_min as BuiltinFn);
        builtins.insert("tensor_abs".to_string(), tensor_builtins::builtin_tensor_abs as BuiltinFn);
        builtins.insert("argmax".to_string(), tensor_builtins::builtin_argmax as BuiltinFn);

        // Shape operations
        builtins.insert("tensor_reshape".to_string(), tensor_builtins::builtin_tensor_reshape as BuiltinFn);
        builtins.insert("tensor_transpose".to_string(), tensor_builtins::builtin_tensor_transpose as BuiltinFn);

        // Tensor creation
        builtins.insert("tensor_from_array".to_string(), tensor_builtins::builtin_tensor_from_array as BuiltinFn);
        builtins.insert("tensor_zeros".to_string(), tensor_builtins::builtin_tensor_zeros as BuiltinFn);
        builtins.insert("tensor_ones".to_string(), tensor_builtins::builtin_tensor_ones as BuiltinFn);
        builtins.insert("tensor_randn".to_string(), tensor_builtins::builtin_tensor_randn as BuiltinFn);
        builtins.insert("tensor_randn_seeded".to_string(), tensor_builtins::builtin_tensor_randn_seeded as BuiltinFn);
        builtins.insert("tensor_rand".to_string(), tensor_builtins::builtin_tensor_rand as BuiltinFn);
        builtins.insert("tensor_get".to_string(), tensor_builtins::builtin_tensor_get as BuiltinFn);
        builtins.insert("tensor_set".to_string(), tensor_builtins::builtin_tensor_set as BuiltinFn);
        builtins.insert("tensor_eye".to_string(), tensor_builtins::builtin_tensor_eye as BuiltinFn);
        builtins.insert("tensor_full".to_string(), tensor_builtins::builtin_tensor_full as BuiltinFn);

        // Autograd operations
        builtins.insert("tensor_requires_grad".to_string(), tensor_builtins::builtin_tensor_requires_grad as BuiltinFn);
        builtins.insert("tensor_zero_grad".to_string(), tensor_builtins::builtin_tensor_zero_grad as BuiltinFn);
        builtins.insert("tensor_grad".to_string(), tensor_builtins::builtin_tensor_grad as BuiltinFn);
        builtins.insert("tensor_set_grad".to_string(), tensor_builtins::builtin_tensor_set_grad as BuiltinFn);
        builtins.insert("tensor_with_grad".to_string(), tensor_builtins::builtin_tensor_with_grad as BuiltinFn);
        builtins.insert("tensor_backward".to_string(), tensor_builtins::builtin_tensor_backward as BuiltinFn);
        builtins.insert("tensor_item".to_string(), tensor_builtins::builtin_tensor_item as BuiltinFn);

        // Training loop helpers
        builtins.insert("tensor_get_data".to_string(), tensor_builtins::builtin_tensor_get_data as BuiltinFn);
        builtins.insert("tensor_update_inplace".to_string(), tensor_builtins::builtin_tensor_update_inplace as BuiltinFn);
        builtins.insert("tensor_from_data".to_string(), tensor_builtins::builtin_tensor_from_data as BuiltinFn);
        builtins.insert("reset_graph".to_string(), tensor_builtins::builtin_reset_graph as BuiltinFn);

        // Math utilities for optimizers
        builtins.insert("tensor_sqrt".to_string(), tensor_builtins::builtin_tensor_sqrt as BuiltinFn);
        builtins.insert("tensor_zeros_like".to_string(), tensor_builtins::builtin_tensor_zeros_like as BuiltinFn);
        builtins.insert("pow".to_string(), tensor_builtins::builtin_pow as BuiltinFn);
        builtins.insert("sqrt".to_string(), tensor_builtins::builtin_sqrt as BuiltinFn);
        builtins.insert("exp".to_string(), tensor_builtins::builtin_exp as BuiltinFn);
        builtins.insert("log".to_string(), tensor_builtins::builtin_log as BuiltinFn);
        builtins.insert("abs".to_string(), tensor_builtins::builtin_abs as BuiltinFn);
        builtins.insert("sin".to_string(), tensor_builtins::builtin_sin as BuiltinFn);
        builtins.insert("cos".to_string(), tensor_builtins::builtin_cos as BuiltinFn);
        builtins.insert("tan".to_string(), tensor_builtins::builtin_tan as BuiltinFn);
        builtins.insert("tensor_sin".to_string(), tensor_builtins::builtin_tensor_sin as BuiltinFn);
        builtins.insert("tensor_cos".to_string(), tensor_builtins::builtin_tensor_cos as BuiltinFn);
        builtins.insert("tensor_from_scalar".to_string(), tensor_builtins::builtin_tensor_from_scalar as BuiltinFn);
        builtins.insert("min".to_string(), tensor_builtins::builtin_min as BuiltinFn);
        builtins.insert("max".to_string(), tensor_builtins::builtin_max as BuiltinFn);

        // Attention mechanisms (Week 1-2: Backend Exposure Roadmap)
        builtins.insert("positional_encoding".to_string(), tensor_builtins::builtin_positional_encoding as BuiltinFn);
        builtins.insert("attention_mask_causal".to_string(), tensor_builtins::builtin_attention_mask_causal as BuiltinFn);
        builtins.insert("attention_scaled".to_string(), tensor_builtins::builtin_attention_scaled as BuiltinFn);
        builtins.insert("attention_self".to_string(), tensor_builtins::builtin_attention_self as BuiltinFn);
        builtins.insert("attention_multi_head".to_string(), tensor_builtins::builtin_attention_multi_head as BuiltinFn);
        // Aliases for more descriptive names
        builtins.insert("attention_scaled_dot_product".to_string(), tensor_builtins::builtin_attention_scaled as BuiltinFn);
        builtins.insert("mha_forward".to_string(), tensor_builtins::builtin_attention_multi_head as BuiltinFn);

        // CNNs / Computer Vision (Week 3-4: Backend Exposure Roadmap)
        builtins.insert("nn_conv2d".to_string(), tensor_builtins::builtin_nn_conv2d as BuiltinFn);
        builtins.insert("nn_maxpool2d".to_string(), tensor_builtins::builtin_nn_maxpool2d as BuiltinFn);
        builtins.insert("nn_avgpool2d".to_string(), tensor_builtins::builtin_nn_avgpool2d as BuiltinFn);
        builtins.insert("nn_batchnorm".to_string(), tensor_builtins::builtin_nn_batchnorm as BuiltinFn);

        // Neural network layers and activations
        builtins.insert("nn_linear_create".to_string(), tensor_builtins::builtin_nn_linear_create as BuiltinFn);
        builtins.insert("nn_linear_forward".to_string(), tensor_builtins::builtin_nn_linear_forward as BuiltinFn);
        builtins.insert("nn_linear".to_string(), tensor_builtins::builtin_nn_linear as BuiltinFn);
        builtins.insert("nn_embedding".to_string(), tensor_builtins::builtin_nn_embedding as BuiltinFn);
        builtins.insert("tensor_concat".to_string(), tensor_builtins::builtin_tensor_concat as BuiltinFn);
        builtins.insert("nn_relu".to_string(), tensor_builtins::builtin_nn_relu as BuiltinFn);
        builtins.insert("nn_sigmoid".to_string(), tensor_builtins::builtin_nn_sigmoid as BuiltinFn);
        builtins.insert("nn_tanh".to_string(), tensor_builtins::builtin_nn_tanh as BuiltinFn);
        builtins.insert("nn_softmax".to_string(), tensor_builtins::builtin_nn_softmax as BuiltinFn);
        builtins.insert("nn_gelu".to_string(), tensor_builtins::builtin_nn_gelu as BuiltinFn);
        builtins.insert("nn_leaky_relu".to_string(), tensor_builtins::builtin_nn_leaky_relu as BuiltinFn);
        builtins.insert("nn_elu".to_string(), tensor_builtins::builtin_nn_elu as BuiltinFn);

        // Loss functions
        builtins.insert("loss_mse".to_string(), tensor_builtins::builtin_loss_mse as BuiltinFn);
        builtins.insert("loss_cross_entropy".to_string(), tensor_builtins::builtin_loss_cross_entropy as BuiltinFn);
        // Aliases for NN-style naming
        builtins.insert("nn_mse_loss".to_string(), tensor_builtins::builtin_loss_mse as BuiltinFn);
        builtins.insert("nn_cross_entropy_loss".to_string(), tensor_builtins::builtin_loss_cross_entropy as BuiltinFn);
        builtins.insert("nn_cross_entropy".to_string(), tensor_builtins::builtin_loss_cross_entropy as BuiltinFn);
        builtins.insert("nn_binary_cross_entropy".to_string(), tensor_builtins::builtin_loss_cross_entropy as BuiltinFn);
        builtins.insert("nn_cross_entropy_logits".to_string(), tensor_builtins::builtin_nn_cross_entropy_logits as BuiltinFn);

        // Optimizers & Schedulers (Week 5-6: Backend Exposure Roadmap)
        builtins.insert("sgd_create".to_string(), tensor_builtins::builtin_sgd_create as BuiltinFn);
        builtins.insert("sgd_step".to_string(), tensor_builtins::builtin_sgd_step as BuiltinFn);
        builtins.insert("adam_create".to_string(), tensor_builtins::builtin_adam_create as BuiltinFn);
        builtins.insert("adam_step".to_string(), tensor_builtins::builtin_adam_step as BuiltinFn);
        builtins.insert("rmsprop_create".to_string(), tensor_builtins::builtin_rmsprop_create as BuiltinFn);
        builtins.insert("rmsprop_step".to_string(), tensor_builtins::builtin_rmsprop_step as BuiltinFn);
        builtins.insert("step_lr".to_string(), tensor_builtins::builtin_step_lr as BuiltinFn);
        builtins.insert("exponential_lr".to_string(), tensor_builtins::builtin_exponential_lr as BuiltinFn);
        builtins.insert("cosine_annealing_lr".to_string(), tensor_builtins::builtin_cosine_annealing_lr as BuiltinFn);

        // Efficient Architectures (Week 7-8 Part 1: Linear Attention, SSMs, MoE)
        // Linear Attention variants
        builtins.insert("linformer_create".to_string(), tensor_builtins::builtin_linformer_create as BuiltinFn);
        builtins.insert("linformer_forward".to_string(), tensor_builtins::builtin_linformer_forward as BuiltinFn);
        builtins.insert("performer_create".to_string(), tensor_builtins::builtin_performer_create as BuiltinFn);
        builtins.insert("performer_forward".to_string(), tensor_builtins::builtin_performer_forward as BuiltinFn);
        builtins.insert("fnet_create".to_string(), tensor_builtins::builtin_fnet_create as BuiltinFn);
        builtins.insert("fnet_forward".to_string(), tensor_builtins::builtin_fnet_forward as BuiltinFn);
        builtins.insert("rwkv_create".to_string(), tensor_builtins::builtin_rwkv_create as BuiltinFn);
        builtins.insert("rwkv_forward".to_string(), tensor_builtins::builtin_rwkv_forward as BuiltinFn);
        // State Space Models
        builtins.insert("mamba_create".to_string(), tensor_builtins::builtin_mamba_create as BuiltinFn);
        builtins.insert("mamba_forward".to_string(), tensor_builtins::builtin_mamba_forward as BuiltinFn);
        builtins.insert("s4_create".to_string(), tensor_builtins::builtin_s4_create as BuiltinFn);
        builtins.insert("s4_forward".to_string(), tensor_builtins::builtin_s4_forward as BuiltinFn);
        // Recurrent Neural Networks
        builtins.insert("lstm_create".to_string(), tensor_builtins::builtin_lstm_create as BuiltinFn);
        builtins.insert("lstm_forward".to_string(), tensor_builtins::builtin_lstm_forward as BuiltinFn);
        builtins.insert("gru_create".to_string(), tensor_builtins::builtin_gru_create as BuiltinFn);
        builtins.insert("gru_forward".to_string(), tensor_builtins::builtin_gru_forward as BuiltinFn);
        // Mixture of Experts
        builtins.insert("moe_create".to_string(), tensor_builtins::builtin_moe_create as BuiltinFn);
        builtins.insert("moe_forward".to_string(), tensor_builtins::builtin_moe_forward as BuiltinFn);

        // Quantization (Week 7-8 Part 2: Model Compression)
        builtins.insert("quantize_tensor_int8".to_string(), tensor_builtins::builtin_quantize_tensor_int8 as BuiltinFn);
        builtins.insert("quantize_tensor_int4".to_string(), tensor_builtins::builtin_quantize_tensor_int4 as BuiltinFn);
        builtins.insert("dequantize_tensor".to_string(), tensor_builtins::builtin_dequantize_tensor as BuiltinFn);
        builtins.insert("quantized_tensor_info".to_string(), tensor_builtins::builtin_quantized_tensor_info as BuiltinFn);
        builtins.insert("quantize_model_weights".to_string(), tensor_builtins::builtin_quantize_model_weights as BuiltinFn);

        // Advanced Reasoning (Week 7-8 Part 3: Chain-of-Thought)
        builtins.insert("cot_create".to_string(), tensor_builtins::builtin_cot_create as BuiltinFn);
        builtins.insert("cot_add_step".to_string(), tensor_builtins::builtin_cot_add_step as BuiltinFn);
        builtins.insert("cot_add_step_conf".to_string(), tensor_builtins::builtin_cot_add_step_conf as BuiltinFn);
        builtins.insert("cot_with_answer".to_string(), tensor_builtins::builtin_cot_with_answer as BuiltinFn);
        builtins.insert("cot_compute_confidence".to_string(), tensor_builtins::builtin_cot_compute_confidence as BuiltinFn);
        builtins.insert("cot_get_info".to_string(), tensor_builtins::builtin_cot_get_info as BuiltinFn);
        builtins.insert("cot_get_step".to_string(), tensor_builtins::builtin_cot_get_step as BuiltinFn);

        // Operator Fusion (Week 15-16)
        builtins.insert("fusion_create".to_string(), tensor_builtins::builtin_fusion_create as BuiltinFn);
        builtins.insert("fusion_analyze".to_string(), tensor_builtins::builtin_fusion_analyze as BuiltinFn);
        builtins.insert("fusion_get_stats".to_string(), tensor_builtins::builtin_fusion_get_stats as BuiltinFn);
        builtins.insert("fusion_enable".to_string(), tensor_builtins::builtin_fusion_enable as BuiltinFn);
        builtins.insert("fusion_set_strategy".to_string(), tensor_builtins::builtin_fusion_set_strategy as BuiltinFn);

        // Multimodal AI (Week 17-19) - Vision-Language
        builtins.insert("clip_encoder_create".to_string(), tensor_builtins::builtin_clip_encoder_create as BuiltinFn);
        builtins.insert("clip_encode_image".to_string(), tensor_builtins::builtin_clip_encode_image as BuiltinFn);
        builtins.insert("clip_encode_text".to_string(), tensor_builtins::builtin_clip_encode_text as BuiltinFn);
        builtins.insert("embedding_cosine_similarity".to_string(), tensor_builtins::builtin_embedding_cosine_similarity as BuiltinFn);
        builtins.insert("image_create".to_string(), tensor_builtins::builtin_image_create as BuiltinFn);
        builtins.insert("image_with_caption".to_string(), tensor_builtins::builtin_image_with_caption as BuiltinFn);
        builtins.insert("vqa_create".to_string(), tensor_builtins::builtin_vqa_create as BuiltinFn);
        builtins.insert("vqa_add_qa".to_string(), tensor_builtins::builtin_vqa_add_qa as BuiltinFn);
        builtins.insert("vqa_answer".to_string(), tensor_builtins::builtin_vqa_answer as BuiltinFn);
        builtins.insert("cross_modal_create".to_string(), tensor_builtins::builtin_cross_modal_create as BuiltinFn);

        // Multimodal AI - Scene Understanding
        builtins.insert("scene_object_create".to_string(), tensor_builtins::builtin_scene_object_create as BuiltinFn);
        builtins.insert("scene_object_with_attribute".to_string(), tensor_builtins::builtin_scene_object_with_attribute as BuiltinFn);
        builtins.insert("scene_graph_create".to_string(), tensor_builtins::builtin_scene_graph_create as BuiltinFn);
        builtins.insert("scene_graph_add_object".to_string(), tensor_builtins::builtin_scene_graph_add_object as BuiltinFn);
        builtins.insert("scene_graph_add_relation".to_string(), tensor_builtins::builtin_scene_graph_add_relation as BuiltinFn);
        builtins.insert("scene_graph_get_relations".to_string(), tensor_builtins::builtin_scene_graph_get_relations as BuiltinFn);
        builtins.insert("scene_graph_to_description".to_string(), tensor_builtins::builtin_scene_graph_to_description as BuiltinFn);
        builtins.insert("scene_graph_generator_create".to_string(), tensor_builtins::builtin_scene_graph_generator_create as BuiltinFn);
        builtins.insert("scene_graph_generate".to_string(), tensor_builtins::builtin_scene_graph_generate as BuiltinFn);
        builtins.insert("temporal_event_create".to_string(), tensor_builtins::builtin_temporal_event_create as BuiltinFn);
        builtins.insert("temporal_event_with_object".to_string(), tensor_builtins::builtin_temporal_event_with_object as BuiltinFn);
        builtins.insert("temporal_event_relation".to_string(), tensor_builtins::builtin_temporal_event_relation as BuiltinFn);

        // Multimodal AI - Cross-Modal Reasoning
        builtins.insert("visual_grounding_create".to_string(), tensor_builtins::builtin_visual_grounding_create as BuiltinFn);
        builtins.insert("visual_grounding_ground_phrase".to_string(), tensor_builtins::builtin_visual_grounding_ground_phrase as BuiltinFn);
        builtins.insert("multimodal_reasoner_create".to_string(), tensor_builtins::builtin_multimodal_reasoner_create as BuiltinFn);
        builtins.insert("multimodal_reasoner_reason".to_string(), tensor_builtins::builtin_multimodal_reasoner_reason as BuiltinFn);
        builtins.insert("multimodal_cot_get_info".to_string(), tensor_builtins::builtin_multimodal_cot_get_info as BuiltinFn);
        builtins.insert("multimodal_cot_get_step".to_string(), tensor_builtins::builtin_multimodal_cot_get_step as BuiltinFn);

        // Meta-Learning (Week 20-21) - MAML
        builtins.insert("maml_create".to_string(), tensor_builtins::builtin_maml_create as BuiltinFn);
        builtins.insert("meta_task_create".to_string(), tensor_builtins::builtin_meta_task_create as BuiltinFn);
        builtins.insert("meta_task_add_support".to_string(), tensor_builtins::builtin_meta_task_add_support as BuiltinFn);
        builtins.insert("meta_task_add_query".to_string(), tensor_builtins::builtin_meta_task_add_query as BuiltinFn);
        builtins.insert("model_params_create".to_string(), tensor_builtins::builtin_model_params_create as BuiltinFn);
        builtins.insert("maml_get_info".to_string(), tensor_builtins::builtin_maml_get_info as BuiltinFn);

        // Meta-Learning - Prototypical Networks
        builtins.insert("episode_create".to_string(), tensor_builtins::builtin_episode_create as BuiltinFn);
        builtins.insert("episode_add_support".to_string(), tensor_builtins::builtin_episode_add_support as BuiltinFn);
        builtins.insert("episode_add_query".to_string(), tensor_builtins::builtin_episode_add_query as BuiltinFn);
        builtins.insert("episode_validate".to_string(), tensor_builtins::builtin_episode_validate as BuiltinFn);
        builtins.insert("prototypical_network_create".to_string(), tensor_builtins::builtin_prototypical_network_create as BuiltinFn);
        builtins.insert("distance_compute".to_string(), tensor_builtins::builtin_distance_compute as BuiltinFn);
        builtins.insert("episode_get_info".to_string(), tensor_builtins::builtin_episode_get_info as BuiltinFn);

        // Knowledge Graphs & GNN (Week 22-24)
        builtins.insert("kg_create".to_string(), tensor_builtins::builtin_kg_create as BuiltinFn);
        builtins.insert("kg_add_entity".to_string(), tensor_builtins::builtin_kg_add_entity as BuiltinFn);
        builtins.insert("kg_add_triple".to_string(), tensor_builtins::builtin_kg_add_triple as BuiltinFn);
        builtins.insert("kg_add_triple_conf".to_string(), tensor_builtins::builtin_kg_add_triple_conf as BuiltinFn);
        builtins.insert("kg_query".to_string(), tensor_builtins::builtin_kg_query as BuiltinFn);
        builtins.insert("kg_get_entity".to_string(), tensor_builtins::builtin_kg_get_entity as BuiltinFn);
        builtins.insert("kg_find_by_name".to_string(), tensor_builtins::builtin_kg_find_by_name as BuiltinFn);
        builtins.insert("kg_find_by_type".to_string(), tensor_builtins::builtin_kg_find_by_type as BuiltinFn);
        builtins.insert("kg_get_related".to_string(), tensor_builtins::builtin_kg_get_related as BuiltinFn);
        builtins.insert("kg_find_paths".to_string(), tensor_builtins::builtin_kg_find_paths as BuiltinFn);
        builtins.insert("kg_get_stats".to_string(), tensor_builtins::builtin_kg_get_stats as BuiltinFn);
        builtins.insert("entity_get_id".to_string(), tensor_builtins::builtin_entity_get_id as BuiltinFn);
        builtins.insert("entity_get_type".to_string(), tensor_builtins::builtin_entity_get_type as BuiltinFn);
        builtins.insert("entity_get_name".to_string(), tensor_builtins::builtin_entity_get_name as BuiltinFn);
        builtins.insert("triple_get_subject".to_string(), tensor_builtins::builtin_triple_get_subject as BuiltinFn);
        builtins.insert("triple_get_predicate".to_string(), tensor_builtins::builtin_triple_get_predicate as BuiltinFn);
        builtins.insert("triple_get_object".to_string(), tensor_builtins::builtin_triple_get_object as BuiltinFn);
        builtins.insert("gnn_create".to_string(), tensor_builtins::builtin_gnn_create as BuiltinFn);
        builtins.insert("gnn_init_embeddings".to_string(), tensor_builtins::builtin_gnn_init_embeddings as BuiltinFn);
        builtins.insert("gnn_forward".to_string(), tensor_builtins::builtin_gnn_forward as BuiltinFn);
        builtins.insert("gnn_forward_multilayer".to_string(), tensor_builtins::builtin_gnn_forward_multilayer as BuiltinFn);

        // Additional KG & GNN functions (Phase 2 - Full Backend Exposure)
        builtins.insert("kg_add_node".to_string(), crate::kg_gnn_builtins::builtin_kg_add_node as BuiltinFn);
        builtins.insert("kg_add_edge".to_string(), crate::kg_gnn_builtins::builtin_kg_add_edge as BuiltinFn);
        builtins.insert("kg_neighbors".to_string(), crate::kg_gnn_builtins::builtin_kg_neighbors as BuiltinFn);
        builtins.insert("kg_node_count".to_string(), crate::kg_gnn_builtins::builtin_kg_node_count as BuiltinFn);
        builtins.insert("kg_edge_count".to_string(), crate::kg_gnn_builtins::builtin_kg_edge_count as BuiltinFn);
        builtins.insert("kg_degree".to_string(), crate::kg_gnn_builtins::builtin_kg_degree as BuiltinFn);
        builtins.insert("kg_shortest_path".to_string(), crate::kg_gnn_builtins::builtin_kg_shortest_path as BuiltinFn);
        builtins.insert("gnn_gcn_layer".to_string(), crate::kg_gnn_builtins::builtin_gnn_gcn_layer as BuiltinFn);
        builtins.insert("gnn_aggregate".to_string(), crate::kg_gnn_builtins::builtin_gnn_aggregate as BuiltinFn);
        builtins.insert("gnn_node_classification".to_string(), crate::kg_gnn_builtins::builtin_gnn_node_classification as BuiltinFn);

        // Symbolic AI (Week 29-30)
        builtins.insert("rule_create".to_string(), tensor_builtins::builtin_rule_create as BuiltinFn);
        builtins.insert("rule_engine_create".to_string(), tensor_builtins::builtin_rule_engine_create as BuiltinFn);
        builtins.insert("rule_engine_add_rule".to_string(), tensor_builtins::builtin_rule_engine_add_rule as BuiltinFn);
        builtins.insert("rule_engine_count".to_string(), tensor_builtins::builtin_rule_engine_count as BuiltinFn);
        builtins.insert("fol_var".to_string(), tensor_builtins::builtin_fol_var as BuiltinFn);
        builtins.insert("fol_const".to_string(), tensor_builtins::builtin_fol_const as BuiltinFn);
        builtins.insert("fol_func".to_string(), tensor_builtins::builtin_fol_func as BuiltinFn);
        builtins.insert("fol_solver_create".to_string(), tensor_builtins::builtin_fol_solver_create as BuiltinFn);
        builtins.insert("fol_unify".to_string(), tensor_builtins::builtin_fol_unify as BuiltinFn);
        builtins.insert("fuzzy_create".to_string(), tensor_builtins::builtin_fuzzy_create as BuiltinFn);
        builtins.insert("fuzzy_and".to_string(), tensor_builtins::builtin_fuzzy_and as BuiltinFn);
        builtins.insert("fuzzy_or".to_string(), tensor_builtins::builtin_fuzzy_or as BuiltinFn);
        builtins.insert("fuzzy_not".to_string(), tensor_builtins::builtin_fuzzy_not as BuiltinFn);
        builtins.insert("fuzzy_get_value".to_string(), tensor_builtins::builtin_fuzzy_get_value as BuiltinFn);
        builtins.insert("concept_create".to_string(), crate::symbolic_builtins::builtin_concept_create as BuiltinFn);
        builtins.insert("concept_graph_create".to_string(), crate::symbolic_builtins::builtin_concept_graph_create as BuiltinFn);
        builtins.insert("concept_graph_add".to_string(), crate::symbolic_builtins::builtin_concept_graph_add as BuiltinFn);
        builtins.insert("concept_graph_count".to_string(), tensor_builtins::builtin_concept_graph_count as BuiltinFn);
        builtins.insert("concept_similarity".to_string(), crate::symbolic_builtins::builtin_concept_similarity as BuiltinFn);

        // Additional Symbolic AI functions (Phase 1 - Full Backend Exposure)
        // First-Order Logic - Formula constructors
        builtins.insert("fol_predicate".to_string(), crate::symbolic_builtins::builtin_fol_predicate as BuiltinFn);
        builtins.insert("fol_variable".to_string(), crate::symbolic_builtins::builtin_fol_variable as BuiltinFn);
        builtins.insert("fol_constant".to_string(), crate::symbolic_builtins::builtin_fol_constant as BuiltinFn);
        builtins.insert("fol_function".to_string(), crate::symbolic_builtins::builtin_fol_function as BuiltinFn);
        builtins.insert("fol_not".to_string(), crate::symbolic_builtins::builtin_fol_not as BuiltinFn);
        builtins.insert("fol_and".to_string(), crate::symbolic_builtins::builtin_fol_and as BuiltinFn);
        builtins.insert("fol_or".to_string(), crate::symbolic_builtins::builtin_fol_or as BuiltinFn);
        builtins.insert("fol_implies".to_string(), crate::symbolic_builtins::builtin_fol_implies as BuiltinFn);
        builtins.insert("fol_forall".to_string(), crate::symbolic_builtins::builtin_fol_forall as BuiltinFn);
        builtins.insert("fol_exists".to_string(), crate::symbolic_builtins::builtin_fol_exists as BuiltinFn);
        // First-Order Logic - Solver operations
        builtins.insert("fol_solver_add_fact".to_string(), crate::symbolic_builtins::builtin_fol_solver_add_fact as BuiltinFn);
        builtins.insert("fol_solver_add_rule".to_string(), crate::symbolic_builtins::builtin_fol_solver_add_rule as BuiltinFn);
        builtins.insert("fol_solver_prove".to_string(), crate::symbolic_builtins::builtin_fol_solver_prove as BuiltinFn);
        // Concept Learning - Additional operations
        builtins.insert("concept_add_property".to_string(), crate::symbolic_builtins::builtin_concept_add_property as BuiltinFn);
        builtins.insert("concept_graph_add_concept".to_string(), crate::symbolic_builtins::builtin_concept_graph_add as BuiltinFn);
        // Differentiable Logic - Fuzzy operations
        builtins.insert("fuzzy_value".to_string(), crate::symbolic_builtins::builtin_fuzzy_value as BuiltinFn);
        builtins.insert("fuzzy_implies".to_string(), crate::symbolic_builtins::builtin_fuzzy_implies as BuiltinFn);
        builtins.insert("fuzzy_to_bool".to_string(), crate::symbolic_builtins::builtin_fuzzy_to_bool as BuiltinFn);
        builtins.insert("fuzzy_to_float".to_string(), crate::symbolic_builtins::builtin_fuzzy_to_float as BuiltinFn);
        builtins.insert("soft_unify_strings".to_string(), crate::symbolic_builtins::builtin_soft_unify_strings as BuiltinFn);

        // Advanced Reasoning: Tree-of-Thoughts & Working Memory
        builtins.insert("tot_create".to_string(), tensor_builtins::builtin_tot_create as BuiltinFn);
        builtins.insert("tot_add_thought".to_string(), tensor_builtins::builtin_tot_add_thought as BuiltinFn);
        builtins.insert("tot_mark_solution".to_string(), tensor_builtins::builtin_tot_mark_solution as BuiltinFn);
        builtins.insert("tot_stats".to_string(), tensor_builtins::builtin_tot_stats as BuiltinFn);
        builtins.insert("memory_item_create".to_string(), tensor_builtins::builtin_memory_item_create as BuiltinFn);
        builtins.insert("stm_create".to_string(), tensor_builtins::builtin_stm_create as BuiltinFn);
        builtins.insert("stm_add".to_string(), tensor_builtins::builtin_stm_add as BuiltinFn);
        builtins.insert("stm_len".to_string(), tensor_builtins::builtin_stm_len as BuiltinFn);
        builtins.insert("ltm_create".to_string(), tensor_builtins::builtin_ltm_create as BuiltinFn);
        builtins.insert("ltm_store".to_string(), tensor_builtins::builtin_ltm_store as BuiltinFn);
        builtins.insert("ltm_size".to_string(), tensor_builtins::builtin_ltm_size as BuiltinFn);
        builtins.insert("working_memory_create".to_string(), tensor_builtins::builtin_working_memory_create as BuiltinFn);
        builtins.insert("working_memory_remember".to_string(), tensor_builtins::builtin_working_memory_remember as BuiltinFn);
        builtins.insert("working_memory_consolidate".to_string(), tensor_builtins::builtin_working_memory_consolidate as BuiltinFn);

        // Optimizers
        builtins.insert("optim_sgd_step".to_string(), tensor_builtins::builtin_optim_sgd_step as BuiltinFn);
        builtins.insert("optim_sgd_momentum_step".to_string(), tensor_builtins::builtin_optim_sgd_momentum_step as BuiltinFn);
        builtins.insert("optim_adam_step".to_string(), tensor_builtins::builtin_optim_adam_step as BuiltinFn);
        builtins.insert("tensor_clip_grad".to_string(), tensor_builtins::builtin_tensor_clip_grad as BuiltinFn);

        // Advanced Backend Functions (Phases 2-6: Complete Exposure)
        // Meta-Learning - Curriculum & Task Sampling
        builtins.insert("curriculum_create".to_string(), crate::advanced_builtins::builtin_curriculum_create as BuiltinFn);
        builtins.insert("task_difficulty".to_string(), crate::advanced_builtins::builtin_task_difficulty as BuiltinFn);
        builtins.insert("task_sample_batch".to_string(), crate::advanced_builtins::builtin_task_sample_batch as BuiltinFn);
        builtins.insert("meta_adapt_lr".to_string(), crate::advanced_builtins::builtin_meta_adapt_lr as BuiltinFn);
        builtins.insert("few_shot_classify".to_string(), crate::advanced_builtins::builtin_few_shot_classify as BuiltinFn);
        // Advanced Reasoning - Causal & Abductive
        builtins.insert("causal_graph_create".to_string(), crate::advanced_builtins::builtin_causal_graph_create as BuiltinFn);
        builtins.insert("causal_add_edge".to_string(), crate::advanced_builtins::builtin_causal_add_edge as BuiltinFn);
        builtins.insert("causal_intervene".to_string(), crate::advanced_builtins::builtin_causal_intervene as BuiltinFn);
        builtins.insert("abductive_explain".to_string(), crate::advanced_builtins::builtin_abductive_explain as BuiltinFn);
        builtins.insert("counterfactual_generate".to_string(), crate::advanced_builtins::builtin_counterfactual_generate as BuiltinFn);
        // Operator Fusion - Pattern Detection
        builtins.insert("fusion_detect_patterns".to_string(), crate::advanced_builtins::builtin_fusion_detect_patterns as BuiltinFn);
        builtins.insert("fusion_apply_pattern".to_string(), crate::advanced_builtins::builtin_fusion_apply_pattern as BuiltinFn);
        builtins.insert("fusion_estimate_speedup".to_string(), crate::advanced_builtins::builtin_fusion_estimate_speedup as BuiltinFn);
        // Multimodal - Temporal & Audio
        builtins.insert("temporal_sequence_create".to_string(), crate::advanced_builtins::builtin_temporal_sequence_create as BuiltinFn);
        builtins.insert("temporal_align".to_string(), crate::advanced_builtins::builtin_temporal_align as BuiltinFn);
        builtins.insert("audio_spectrogram".to_string(), crate::advanced_builtins::builtin_audio_spectrogram as BuiltinFn);
        builtins.insert("audio_mfcc".to_string(), crate::advanced_builtins::builtin_audio_mfcc as BuiltinFn);
        // Enhanced Reasoning - Memory & Planning
        builtins.insert("memory_consolidate_threshold".to_string(), crate::advanced_builtins::builtin_memory_consolidate_threshold as BuiltinFn);
        builtins.insert("planning_create_goal".to_string(), crate::advanced_builtins::builtin_planning_create_goal as BuiltinFn);
        builtins.insert("planning_search".to_string(), crate::advanced_builtins::builtin_planning_search as BuiltinFn);

        // Autograd helpers (backward pass computation)
        builtins.insert("autograd_compute_linear_grad".to_string(), tensor_builtins::builtin_autograd_compute_linear_grad as BuiltinFn);
        builtins.insert("autograd_compute_relu_grad".to_string(), tensor_builtins::builtin_autograd_compute_relu_grad as BuiltinFn);
        builtins.insert("autograd_compute_sigmoid_grad".to_string(), tensor_builtins::builtin_autograd_compute_sigmoid_grad as BuiltinFn);
        builtins.insert("autograd_compute_mse_grad".to_string(), tensor_builtins::builtin_autograd_compute_mse_grad as BuiltinFn);

        // GPU Functions
        builtins.insert("gpu_available".to_string(), tensor_builtins::builtin_gpu_available as BuiltinFn);
        builtins.insert("gpu_info".to_string(), tensor_builtins::builtin_gpu_info as BuiltinFn);
        builtins.insert("tensor_device".to_string(), tensor_builtins::builtin_tensor_device as BuiltinFn);
        builtins.insert("tensor_to_gpu".to_string(), tensor_builtins::builtin_tensor_to_gpu as BuiltinFn);
        builtins.insert("tensor_to_cpu".to_string(), tensor_builtins::builtin_tensor_to_cpu as BuiltinFn);

        // Neural Network Layers (v0.2.0)
        builtins.insert("linear".to_string(), tensor_builtins::builtin_linear as BuiltinFn);
        builtins.insert("conv2d".to_string(), tensor_builtins::builtin_conv2d as BuiltinFn);
        builtins.insert("maxpool2d".to_string(), tensor_builtins::builtin_maxpool2d as BuiltinFn);
        builtins.insert("avgpool2d".to_string(), tensor_builtins::builtin_avgpool2d as BuiltinFn);
        builtins.insert("batchnorm".to_string(), tensor_builtins::builtin_batchnorm as BuiltinFn);
        builtins.insert("layernorm".to_string(), tensor_builtins::builtin_layernorm as BuiltinFn);
        builtins.insert("nn_layernorm".to_string(), tensor_builtins::builtin_layernorm as BuiltinFn);
        builtins.insert("dropout".to_string(), tensor_builtins::builtin_dropout as BuiltinFn);
        builtins.insert("nn_dropout".to_string(), tensor_builtins::builtin_dropout as BuiltinFn);
        builtins.insert("layer_forward".to_string(), tensor_builtins::builtin_layer_forward as BuiltinFn);

        // Activation Functions - GPU Accelerated (v0.2.0)
        builtins.insert("tensor_relu".to_string(), tensor_builtins::builtin_tensor_relu as BuiltinFn);
        builtins.insert("tensor_sigmoid".to_string(), tensor_builtins::builtin_tensor_sigmoid as BuiltinFn);
        builtins.insert("tensor_tanh".to_string(), tensor_builtins::builtin_tensor_tanh as BuiltinFn);
        builtins.insert("tensor_gelu".to_string(), tensor_builtins::builtin_tensor_gelu as BuiltinFn);
        builtins.insert("tensor_softmax".to_string(), tensor_builtins::builtin_tensor_softmax as BuiltinFn);

        Interpreter {
            env: Environment::new(),
            return_value: None,
            break_loop: false,
            continue_loop: false,
            graph: ComputationGraph::new(),
            builtins,
        }
    }

    pub fn eval(&mut self, program: Program) -> Result<Value, String> {
        let mut result = Value::Null;

        for statement in program.statements {
            result = self.eval_statement(&statement)?;

            // If we hit a return statement, stop execution
            if self.return_value.is_some() {
                break;
            }
        }

        Ok(result)
    }

    pub fn eval_statement(&mut self, stmt: &Statement) -> Result<Value, String> {
        match stmt {
            Statement::Let(let_stmt) => self.eval_let_statement(let_stmt),
            Statement::Assign(assign_stmt) => self.eval_assign_statement(assign_stmt),
            Statement::Return(ret_stmt) => self.eval_return_statement(ret_stmt),
            Statement::Expression(expr_stmt) => self.eval_expression(&expr_stmt.expression),
            Statement::Function(func_stmt) => self.eval_function_statement(func_stmt),
            Statement::If(if_stmt) => self.eval_if_statement(if_stmt),
            Statement::While(while_stmt) => self.eval_while_statement(while_stmt),
            Statement::For(for_stmt) => self.eval_for_statement(for_stmt),
            Statement::Break => {
                self.break_loop = true;
                Ok(Value::Null)
            }
            Statement::Continue => {
                self.continue_loop = true;
                Ok(Value::Null)
            }
        }
    }

    fn eval_let_statement(&mut self, stmt: &LetStatement) -> Result<Value, String> {
        let value = self.eval_expression(&stmt.value)?;
        self.env.set(stmt.name.clone(), value.clone());
        Ok(value)
    }

    fn eval_assign_statement(&mut self, stmt: &AssignStatement) -> Result<Value, String> {
        // Evaluate new value
        let value = self.eval_expression(&stmt.value)?;

        // Check if target is a simple identifier or indexed expression
        match &stmt.target {
            Expression::Identifier(name) => {
                // Simple assignment: x = value
                if self.env.update(name, value.clone()) {
                    Ok(value)
                } else {
                    Err(format!("Cannot assign to undefined variable: {}", name))
                }
            }
            Expression::Index { object, index } => {
                // Indexed assignment: array[i] = value or array[i][j] = value
                self.eval_indexed_assignment(object, index, value)
            }
            _ => Err("Invalid assignment target".to_string()),
        }
    }

    fn eval_indexed_assignment(
        &mut self,
        object: &Expression,
        index: &Expression,
        value: Value,
    ) -> Result<Value, String> {
        // Evaluate the index
        let index_val = self.eval_expression(index)?;
        let idx = match index_val {
            Value::Integer(i) => i as usize,
            Value::Float(f) => f as usize,
            _ => return Err("Array index must be a number".to_string()),
        };

        // Handle nested indexing: array[i][j] = value
        if let Expression::Index {
            object: nested_object,
            index: nested_index,
        } = object
        {
            // Get the root array name
            let root_name = self.extract_root_variable(nested_object)?;
            let mut root_array = self.env.get(&root_name).cloned()
                .ok_or_else(|| format!("Undefined variable: {}", root_name))?;

            // Navigate and mutate the nested structure
            self.mutate_nested_array(&mut root_array, nested_object, nested_index, idx, value.clone())?;

            // Update the root variable
            self.env.update(&root_name, root_array);
            Ok(value)
        } else if let Expression::Identifier(name) = object {
            // Single-level indexing: array[i] = value
            let mut array = self.env.get(name).cloned()
                .ok_or_else(|| format!("Undefined variable: {}", name))?;

            match &mut array {
                Value::Array(elements) => {
                    if idx >= elements.len() {
                        return Err(format!("Index {} out of bounds (len={})", idx, elements.len()));
                    }
                    elements[idx] = value.clone();
                }
                _ => return Err(format!("{} is not an array", name)),
            }

            self.env.update(name, array);
            Ok(value)
        } else {
            Err("Invalid indexed assignment target".to_string())
        }
    }

    fn extract_root_variable(&self, expr: &Expression) -> Result<String, String> {
        match expr {
            Expression::Identifier(name) => Ok(name.clone()),
            Expression::Index { object, .. } => self.extract_root_variable(object),
            _ => Err("Cannot extract root variable from expression".to_string()),
        }
    }

    fn mutate_nested_array(
        &mut self,
        current: &mut Value,
        object: &Expression,
        index: &Expression,
        final_idx: usize,
        value: Value,
    ) -> Result<(), String> {
        // Evaluate the current level's index
        let idx_val = self.eval_expression(index)?;
        let idx = match idx_val {
            Value::Integer(i) => i as usize,
            Value::Float(f) => f as usize,
            _ => return Err("Array index must be a number".to_string()),
        };

        match current {
            Value::Array(elements) => {
                if idx >= elements.len() {
                    return Err(format!("Index {} out of bounds", idx));
                }

                // Check if we need to recurse deeper
                if let Expression::Index {
                    object: nested_object,
                    index: nested_index,
                } = object
                {
                    // Recurse into the nested array
                    self.mutate_nested_array(&mut elements[idx], nested_object, nested_index, final_idx, value)
                } else {
                    // Base case: we're at the target level
                    // elements[idx] is already the target array element
                    // final_idx is the index in this element
                    match &mut elements[idx] {
                        Value::Array(inner) => {
                            if final_idx >= inner.len() {
                                return Err(format!("Index {} out of bounds", final_idx));
                            }
                            inner[final_idx] = value;
                            Ok(())
                        }
                        _ => {
                            // This shouldn't happen in well-formed code
                            Err("Expected array at this level".to_string())
                        }
                    }
                }
            }
            _ => Err("Cannot index into non-array".to_string()),
        }
    }

    fn eval_return_statement(&mut self, stmt: &ReturnStatement) -> Result<Value, String> {
        let value = self.eval_expression(&stmt.value)?;
        self.return_value = Some(value.clone());
        Ok(value)
    }

    fn eval_function_statement(&mut self, stmt: &FunctionStatement) -> Result<Value, String> {
        let func = Value::Function {
            parameters: stmt.parameters.clone(),
            body: stmt.body.clone(),
            closure: self.env.clone(),
        };
        self.env.set(stmt.name.clone(), func.clone());
        Ok(func)
    }

    fn eval_if_statement(&mut self, stmt: &IfStatement) -> Result<Value, String> {
        let condition = self.eval_expression(&stmt.condition)?;

        if condition.is_truthy() {
            // Execute consequence block
            let mut result = Value::Null;
            for statement in &stmt.consequence {
                result = self.eval_statement(statement)?;

                // Check for return statement
                if self.return_value.is_some() {
                    break;
                }
            }
            Ok(result)
        } else if let Some(alternative) = &stmt.alternative {
            // Execute alternative (else) block
            let mut result = Value::Null;
            for statement in alternative {
                result = self.eval_statement(statement)?;

                // Check for return statement
                if self.return_value.is_some() {
                    break;
                }
            }
            Ok(result)
        } else {
            Ok(Value::Null)
        }
    }

    fn eval_while_statement(&mut self, stmt: &WhileStatement) -> Result<Value, String> {
        let mut result = Value::Null;

        loop {
            let condition = self.eval_expression(&stmt.condition)?;

            if !condition.is_truthy() {
                break;
            }

            // Execute loop body
            for statement in &stmt.body {
                result = self.eval_statement(statement)?;

                // Check for break
                if self.break_loop {
                    self.break_loop = false; // Reset flag
                    return Ok(result);
                }

                // Check for continue
                if self.continue_loop {
                    self.continue_loop = false; // Reset flag
                    break; // Exit inner loop, continue outer loop
                }

                // Check for return statement
                if self.return_value.is_some() {
                    return Ok(result);
                }
            }
        }

        Ok(result)
    }

    fn eval_for_statement(&mut self, stmt: &ForStatement) -> Result<Value, String> {
        let iterable = self.eval_expression(&stmt.iterable)?;
        let mut result = Value::Null;

        // Create new scope for loop variable
        self.env.push_scope();

        // Handle different iterable types
        match iterable {
            Value::Array(elements) => {
                for element in elements {
                    // Set loop variable
                    self.env.set(stmt.variable.clone(), element);

                    // Execute loop body
                    for statement in &stmt.body {
                        result = self.eval_statement(statement)?;

                        // Check for break
                        if self.break_loop {
                            self.break_loop = false; // Reset flag
                            self.env.pop_scope();
                            return Ok(result);
                        }

                        // Check for continue
                        if self.continue_loop {
                            self.continue_loop = false; // Reset flag
                            break; // Exit inner loop, continue outer loop
                        }

                        // Check for return statement
                        if self.return_value.is_some() {
                            self.env.pop_scope();
                            return Ok(result);
                        }
                    }
                }
            }

            // TODO: Handle range expressions (0..10)
            // For now, error on non-array iterables
            _ => {
                self.env.pop_scope();
                return Err(format!(
                    "Cannot iterate over type {}",
                    iterable.type_name()
                ));
            }
        }

        self.env.pop_scope();
        Ok(result)
    }

    pub fn eval_expression(&mut self, expr: &Expression) -> Result<Value, String> {
        match expr {
            Expression::IntegerLiteral(i) => Ok(Value::Integer(*i)),
            Expression::FloatLiteral(f) => Ok(Value::Float(*f)),
            Expression::BooleanLiteral(b) => Ok(Value::Boolean(*b)),
            Expression::StringLiteral(s) => Ok(Value::String(s.clone())),

            Expression::Identifier(name) => self
                .env
                .get(name)
                .cloned()
                .ok_or_else(|| format!("Undefined variable: {}", name)),

            Expression::ArrayLiteral(elements) => {
                let values: Result<Vec<Value>, String> =
                    elements.iter().map(|e| self.eval_expression(e)).collect();
                Ok(Value::Array(values?))
            }

            Expression::ArrayRepeat { value, count } => {
                // Evaluate the value to repeat
                let val = self.eval_expression(value)?;

                // Evaluate count (must be integer)
                let count_val = self.eval_expression(count)?;
                let count_int = match count_val {
                    Value::Integer(n) => n as usize,
                    Value::Float(n) => n as usize,
                    _ => return Err(format!("Array repeat count must be integer, got {:?}", count_val)),
                };

                // Create array by repeating value
                let mut array = Vec::new();
                for _ in 0..count_int {
                    array.push(val.clone());
                }

                Ok(Value::Array(array))
            }

            Expression::TensorLiteral(elements) => {
                let values: Result<Vec<Value>, String> =
                    elements.iter().map(|e| self.eval_expression(e)).collect();
                let data = values?;
                let shape = vec![data.len()];
                Ok(Value::Tensor { data, shape })
            }

            Expression::Binary {
                left,
                operator,
                right,
            } => self.eval_binary_expression(left, operator, right),

            Expression::Unary { operator, operand } => {
                self.eval_unary_expression(operator, operand)
            }

            Expression::Call {
                function,
                arguments,
            } => self.eval_call_expression(function, arguments),

            Expression::Index { object, index } => self.eval_index_expression(object, index),

            Expression::Autograd { expression } => self.eval_autograd_expression(expression),

            Expression::Range { start, end } => self.eval_range_expression(start, end),

            Expression::InclusiveRange { start, end } => {
                self.eval_inclusive_range_expression(start, end)
            }

            Expression::If {
                condition,
                consequence,
                alternative,
            } => self.eval_if_expression(condition, consequence, alternative),

            Expression::Match { value, arms } => self.eval_match_expression(value, arms),

            Expression::TupleLiteral(elements) => {
                let values: Result<Vec<Value>, String> =
                    elements.iter().map(|e| self.eval_expression(e)).collect();
                Ok(Value::Tuple(values?))
            }

            Expression::TupleIndex { tuple, index } => {
                self.eval_tuple_index_expression(tuple, *index)
            }

            Expression::Cast {
                expression,
                target_type,
            } => self.eval_cast_expression(expression, target_type),
        }
    }

    fn eval_binary_expression(
        &mut self,
        left: &Expression,
        op: &BinaryOperator,
        right: &Expression,
    ) -> Result<Value, String> {
        let left_val = self.eval_expression(left)?;
        let right_val = self.eval_expression(right)?;

        match op {
            BinaryOperator::Add => self.eval_add(&left_val, &right_val),
            BinaryOperator::Subtract => self.eval_subtract(&left_val, &right_val),
            BinaryOperator::Multiply => self.eval_multiply(&left_val, &right_val),
            BinaryOperator::Divide => self.eval_divide(&left_val, &right_val),
            BinaryOperator::Modulo => self.eval_modulo(&left_val, &right_val),
            BinaryOperator::MatMul => self.eval_matmul(&left_val, &right_val),
            BinaryOperator::Equal => Ok(Value::Boolean(left_val == right_val)),
            BinaryOperator::NotEqual => Ok(Value::Boolean(left_val != right_val)),
            BinaryOperator::LessThan => self.eval_less_than(&left_val, &right_val),
            BinaryOperator::LessEqual => self.eval_less_equal(&left_val, &right_val),
            BinaryOperator::GreaterThan => self.eval_greater_than(&left_val, &right_val),
            BinaryOperator::GreaterEqual => self.eval_greater_equal(&left_val, &right_val),
            BinaryOperator::And => Ok(Value::Boolean(
                left_val.is_truthy() && right_val.is_truthy(),
            )),
            BinaryOperator::Or => Ok(Value::Boolean(
                left_val.is_truthy() || right_val.is_truthy(),
            )),
        }
    }

    fn eval_add(&self, left: &Value, right: &Value) -> Result<Value, String> {
        match (left, right) {
            (Value::Integer(l), Value::Integer(r)) => Ok(Value::Integer(l + r)),
            (Value::Float(l), Value::Float(r)) => Ok(Value::Float(l + r)),
            (Value::Integer(l), Value::Float(r)) => Ok(Value::Float(*l as f64 + r)),
            (Value::Float(l), Value::Integer(r)) => Ok(Value::Float(l + *r as f64)),
            (Value::String(l), Value::String(r)) => Ok(Value::String(format!("{}{}", l, r))),
            (Value::Array(l), Value::Array(r)) => {
                // Array concatenation: [1, 2] + [3, 4] = [1, 2, 3, 4]
                let mut result = l.clone();
                result.extend(r.clone());
                Ok(Value::Array(result))
            }
            _ => Err(format!(
                "Cannot add {} and {}",
                left.type_name(),
                right.type_name()
            )),
        }
    }

    fn eval_subtract(&self, left: &Value, right: &Value) -> Result<Value, String> {
        match (left, right) {
            (Value::Integer(l), Value::Integer(r)) => Ok(Value::Integer(l - r)),
            (Value::Float(l), Value::Float(r)) => Ok(Value::Float(l - r)),
            (Value::Integer(l), Value::Float(r)) => Ok(Value::Float(*l as f64 - r)),
            (Value::Float(l), Value::Integer(r)) => Ok(Value::Float(l - *r as f64)),
            _ => Err(format!(
                "Cannot subtract {} and {}",
                left.type_name(),
                right.type_name()
            )),
        }
    }

    fn eval_multiply(&self, left: &Value, right: &Value) -> Result<Value, String> {
        match (left, right) {
            (Value::Integer(l), Value::Integer(r)) => Ok(Value::Integer(l * r)),
            (Value::Float(l), Value::Float(r)) => Ok(Value::Float(l * r)),
            (Value::Integer(l), Value::Float(r)) => Ok(Value::Float(*l as f64 * r)),
            (Value::Float(l), Value::Integer(r)) => Ok(Value::Float(l * *r as f64)),
            _ => Err(format!(
                "Cannot multiply {} and {}",
                left.type_name(),
                right.type_name()
            )),
        }
    }

    fn eval_divide(&self, left: &Value, right: &Value) -> Result<Value, String> {
        match (left, right) {
            (Value::Integer(l), Value::Integer(r)) => {
                if *r == 0 {
                    return Err("Division by zero".to_string());
                }
                Ok(Value::Integer(l / r))
            }
            (Value::Float(l), Value::Float(r)) => {
                if *r == 0.0 {
                    return Err("Division by zero".to_string());
                }
                Ok(Value::Float(l / r))
            }
            (Value::Integer(l), Value::Float(r)) => {
                if *r == 0.0 {
                    return Err("Division by zero".to_string());
                }
                Ok(Value::Float(*l as f64 / r))
            }
            (Value::Float(l), Value::Integer(r)) => {
                if *r == 0 {
                    return Err("Division by zero".to_string());
                }
                Ok(Value::Float(l / *r as f64))
            }
            _ => Err(format!(
                "Cannot divide {} and {}",
                left.type_name(),
                right.type_name()
            )),
        }
    }

    fn eval_modulo(&self, left: &Value, right: &Value) -> Result<Value, String> {
        match (left, right) {
            (Value::Integer(l), Value::Integer(r)) => {
                if *r == 0 {
                    return Err("Modulo by zero".to_string());
                }
                Ok(Value::Integer(l % r))
            }
            _ => Err(format!(
                "Cannot modulo {} and {}",
                left.type_name(),
                right.type_name()
            )),
        }
    }

    fn eval_matmul(&self, _left: &Value, _right: &Value) -> Result<Value, String> {
        // TODO: Implement matrix multiplication in future phases
        Err("Matrix multiplication not yet implemented".to_string())
    }

    fn eval_less_than(&self, left: &Value, right: &Value) -> Result<Value, String> {
        match (left, right) {
            (Value::Integer(l), Value::Integer(r)) => Ok(Value::Boolean(l < r)),
            (Value::Float(l), Value::Float(r)) => Ok(Value::Boolean(l < r)),
            (Value::Integer(l), Value::Float(r)) => Ok(Value::Boolean((*l as f64) < *r)),
            (Value::Float(l), Value::Integer(r)) => Ok(Value::Boolean(*l < (*r as f64))),
            _ => Err(format!(
                "Cannot compare {} and {}",
                left.type_name(),
                right.type_name()
            )),
        }
    }

    fn eval_less_equal(&self, left: &Value, right: &Value) -> Result<Value, String> {
        match (left, right) {
            (Value::Integer(l), Value::Integer(r)) => Ok(Value::Boolean(l <= r)),
            (Value::Float(l), Value::Float(r)) => Ok(Value::Boolean(l <= r)),
            (Value::Integer(l), Value::Float(r)) => Ok(Value::Boolean((*l as f64) <= *r)),
            (Value::Float(l), Value::Integer(r)) => Ok(Value::Boolean(*l <= (*r as f64))),
            _ => Err(format!(
                "Cannot compare {} and {}",
                left.type_name(),
                right.type_name()
            )),
        }
    }

    fn eval_greater_than(&self, left: &Value, right: &Value) -> Result<Value, String> {
        match (left, right) {
            (Value::Integer(l), Value::Integer(r)) => Ok(Value::Boolean(l > r)),
            (Value::Float(l), Value::Float(r)) => Ok(Value::Boolean(l > r)),
            (Value::Integer(l), Value::Float(r)) => Ok(Value::Boolean((*l as f64) > *r)),
            (Value::Float(l), Value::Integer(r)) => Ok(Value::Boolean(*l > (*r as f64))),
            _ => Err(format!(
                "Cannot compare {} and {}",
                left.type_name(),
                right.type_name()
            )),
        }
    }

    fn eval_greater_equal(&self, left: &Value, right: &Value) -> Result<Value, String> {
        match (left, right) {
            (Value::Integer(l), Value::Integer(r)) => Ok(Value::Boolean(l >= r)),
            (Value::Float(l), Value::Float(r)) => Ok(Value::Boolean(l >= r)),
            (Value::Integer(l), Value::Float(r)) => Ok(Value::Boolean((*l as f64) >= *r)),
            (Value::Float(l), Value::Integer(r)) => Ok(Value::Boolean(*l >= (*r as f64))),
            _ => Err(format!(
                "Cannot compare {} and {}",
                left.type_name(),
                right.type_name()
            )),
        }
    }

    fn eval_unary_expression(
        &mut self,
        op: &UnaryOperator,
        operand: &Expression,
    ) -> Result<Value, String> {
        let value = self.eval_expression(operand)?;

        match op {
            UnaryOperator::Negate => match value {
                Value::Integer(i) => Ok(Value::Integer(-i)),
                Value::Float(f) => Ok(Value::Float(-f)),
                _ => Err(format!("Cannot negate {}", value.type_name())),
            },
            UnaryOperator::Not => Ok(Value::Boolean(!value.is_truthy())),
        }
    }

    fn eval_call_expression(
        &mut self,
        function: &Expression,
        arguments: &[Expression],
    ) -> Result<Value, String> {
        // Check if this is a built-in function call
        if let Expression::Identifier(name) = function {
            if let Some(&builtin) = self.builtins.get(name) {
                // Clone the builtin function pointer before evaluating arguments
                // This avoids borrowing self while we need mutable access for eval_expression

                // Evaluate arguments
                let arg_values: Result<Vec<Value>, String> = arguments
                    .iter()
                    .map(|arg| self.eval_expression(arg))
                    .collect();
                let arg_values = arg_values?;

                // Call built-in function
                return builtin(arg_values);
            }
        }

        // Not a built-in, evaluate as regular function
        let func_val = self.eval_expression(function)?;

        match func_val {
            Value::Function {
                parameters,
                body,
                closure,
            } => {
                if parameters.len() != arguments.len() {
                    return Err(format!(
                        "Wrong number of arguments: expected {}, got {}",
                        parameters.len(),
                        arguments.len()
                    ));
                }

                // Evaluate arguments
                let arg_values: Result<Vec<Value>, String> = arguments
                    .iter()
                    .map(|arg| self.eval_expression(arg))
                    .collect();
                let arg_values = arg_values?;

                // Save current environment and use closure
                let saved_env = self.env.clone();
                self.env = closure;

                // Create new scope for function
                self.env.push_scope();

                // Bind parameters to arguments
                for (param, arg_val) in parameters.iter().zip(arg_values.iter()) {
                    self.env.set(param.name.clone(), arg_val.clone());
                }

                // Execute function body
                let mut result = Value::Null;
                for stmt in &body {
                    result = self.eval_statement(stmt)?;
                    if self.return_value.is_some() {
                        break;
                    }
                }

                // Get return value
                let return_val = self.return_value.take().unwrap_or(result);

                // Restore environment
                self.env.pop_scope();
                self.env = saved_env;

                Ok(return_val)
            }
            _ => Err(format!(
                "Cannot call non-function value: {}",
                func_val.type_name()
            )),
        }
    }

    fn eval_index_expression(
        &mut self,
        object: &Expression,
        index: &Expression,
    ) -> Result<Value, String> {
        let obj_val = self.eval_expression(object)?;
        let idx_val = self.eval_expression(index)?;

        // Check if this is a slice operation (index is a range/array)
        match idx_val {
            Value::Array(ref range_values) => {
                // This is a slice operation: arr[range]
                // The range was already evaluated to an array like [1, 2, 3]
                // We extract start and end from the range
                if range_values.is_empty() {
                    return Ok(Value::Array(vec![]));
                }

                // Get start index from first element
                let start = match range_values.first().unwrap() {
                    Value::Integer(i) => *i,
                    _ => return Err("Range values must be integers".to_string()),
                };

                // Get end index from last element + 1 (to make it exclusive for slicing)
                let end = match range_values.last().unwrap() {
                    Value::Integer(i) => i + 1,
                    _ => return Err("Range values must be integers".to_string()),
                };

                return self.eval_slice(obj_val, start, end);
            }
            Value::Integer(i) => {
                // Single index access (existing behavior)
                return self.eval_single_index(obj_val, i);
            }
            _ => {
                return Err(format!(
                    "Index must be an integer or range, got {}",
                    idx_val.type_name()
                ))
            }
        }
    }

    fn eval_single_index(&self, obj_val: Value, idx: i64) -> Result<Value, String> {
        match obj_val {
            Value::Array(ref elements) => {
                let idx = if idx < 0 {
                    (elements.len() as i64 + idx) as usize
                } else {
                    idx as usize
                };

                elements
                    .get(idx)
                    .cloned()
                    .ok_or_else(|| format!("Index out of bounds: {}", idx))
            }
            Value::Tensor { ref data, .. } => {
                let idx = if idx < 0 {
                    (data.len() as i64 + idx) as usize
                } else {
                    idx as usize
                };

                data.get(idx)
                    .cloned()
                    .ok_or_else(|| format!("Index out of bounds: {}", idx))
            }
            _ => Err(format!("Cannot index into {}", obj_val.type_name())),
        }
    }

    fn eval_slice(&self, obj_val: Value, start: i64, end: i64) -> Result<Value, String> {
        match obj_val {
            Value::Array(ref elements) => {
                let len = elements.len() as i64;

                // Normalize negative indices
                let start_idx = if start < 0 {
                    (len + start).max(0) as usize
                } else {
                    start.min(len) as usize
                };

                let end_idx = if end < 0 {
                    (len + end).max(0) as usize
                } else {
                    end.min(len) as usize
                };

                // Ensure start <= end
                if start_idx > end_idx {
                    return Ok(Value::Array(vec![]));
                }

                // Extract slice
                let slice = elements[start_idx..end_idx].to_vec();
                Ok(Value::Array(slice))
            }
            Value::Tensor { ref data, shape } => {
                let len = data.len() as i64;

                // Normalize negative indices
                let start_idx = if start < 0 {
                    (len + start).max(0) as usize
                } else {
                    start.min(len) as usize
                };

                let end_idx = if end < 0 {
                    (len + end).max(0) as usize
                } else {
                    end.min(len) as usize
                };

                // Ensure start <= end
                if start_idx > end_idx {
                    return Ok(Value::Tensor {
                        data: vec![],
                        shape: vec![0],
                    });
                }

                // Extract slice
                let slice = data[start_idx..end_idx].to_vec();
                Ok(Value::Tensor {
                    data: slice,
                    shape: vec![end_idx - start_idx],
                })
            }
            _ => Err(format!("Cannot slice {}", obj_val.type_name())),
        }
    }

    fn eval_autograd_expression(&mut self, expr: &Expression) -> Result<Value, String> {
        // Evaluate the expression and convert to autograd tensor if needed
        let value = self.eval_expression(expr)?;

        match value {
            Value::Integer(i) => {
                // Convert integer to autograd tensor
                let tensor = AutogradTensor::scalar_with_grad(i as f64);
                self.graph.add_node(tensor.clone());
                Ok(Value::AutogradTensor(tensor))
            }
            Value::Float(f) => {
                // Convert float to autograd tensor
                let tensor = AutogradTensor::scalar_with_grad(f);
                self.graph.add_node(tensor.clone());
                Ok(Value::AutogradTensor(tensor))
            }
            Value::Array(ref elements) => {
                // Convert array to autograd tensor
                let data: Result<Vec<f64>, String> = elements
                    .iter()
                    .map(|v| match v {
                        Value::Integer(i) => Ok(*i as f64),
                        Value::Float(f) => Ok(*f),
                        _ => Err(format!(
                            "autograd() only works with numeric arrays, found {}",
                            v.type_name()
                        )),
                    })
                    .collect();

                let data = data?;
                let shape = vec![data.len()];
                let tensor = AutogradTensor::with_grad(data, shape);
                self.graph.add_node(tensor.clone());
                Ok(Value::AutogradTensor(tensor))
            }
            Value::AutogradTensor(_) => {
                // Already an autograd tensor, just return it
                Ok(value)
            }
            _ => Err(format!(
                "autograd() can only be applied to numbers or arrays, got {}",
                value.type_name()
            )),
        }
    }

    fn eval_range_expression(
        &mut self,
        start: &Expression,
        end: &Expression,
    ) -> Result<Value, String> {
        // Evaluate start and end expressions
        let start_val = self.eval_expression(start)?;
        let end_val = self.eval_expression(end)?;

        // Convert to integers
        let start_int = start_val.to_integer()?;
        let end_int = end_val.to_integer()?;

        // Generate range (same as range() builtin with 2 arguments)
        let mut result = Vec::new();
        let mut i = start_int;
        while i < end_int {
            result.push(Value::Integer(i));
            i += 1;
        }

        Ok(Value::Array(result))
    }

    fn eval_inclusive_range_expression(
        &mut self,
        start: &Expression,
        end: &Expression,
    ) -> Result<Value, String> {
        // Evaluate start and end expressions
        let start_val = self.eval_expression(start)?;
        let end_val = self.eval_expression(end)?;

        // Convert to integers
        let start_int = start_val.to_integer()?;
        let end_int = end_val.to_integer()?;

        // Generate inclusive range (includes end value)
        let mut result = Vec::new();
        let mut i = start_int;
        while i <= end_int {
            result.push(Value::Integer(i));
            i += 1;
        }

        Ok(Value::Array(result))
    }

    fn eval_if_expression(
        &mut self,
        condition: &Expression,
        consequence: &[Statement],
        alternative: &[Statement],
    ) -> Result<Value, String> {
        // Evaluate condition
        let cond_val = self.eval_expression(condition)?;

        // Choose which block to execute
        let block = if cond_val.is_truthy() {
            consequence
        } else {
            alternative
        };

        // Execute the chosen block and return the last expression's value
        let mut result = Value::Null;
        for statement in block {
            result = self.eval_statement(statement)?;

            // If we hit a return statement, propagate it
            if self.return_value.is_some() {
                break;
            }
        }

        Ok(result)
    }

    fn eval_match_expression(
        &mut self,
        value: &Expression,
        arms: &[MatchArm],
    ) -> Result<Value, String> {
        // Evaluate the value to match against
        let match_value = self.eval_expression(value)?;

        // Try each arm in order
        for arm in arms {
            if let Some(bindings) = self.pattern_matches(&arm.pattern, &match_value)? {
                // Pattern matched! Apply bindings and evaluate expression

                // Push new scope for bindings
                self.env.push_scope();

                // Add bindings to environment
                for (name, value) in bindings {
                    self.env.set(name, value);
                }

                // Evaluate the arm's expression
                let result = self.eval_expression(&arm.expression)?;

                // Pop scope
                self.env.pop_scope();

                return Ok(result);
            }
        }

        // No pattern matched - this is an error
        Err(format!(
            "Non-exhaustive match: value {} didn't match any pattern",
            match_value.type_name()
        ))
    }

    /// Check if a pattern matches a value, returning bindings if it does
    fn pattern_matches(
        &self,
        pattern: &Pattern,
        value: &Value,
    ) -> Result<Option<Vec<(String, Value)>>, String> {
        match pattern {
            Pattern::IntegerLiteral(expected) => {
                if let Value::Integer(actual) = value {
                    if expected == actual {
                        Ok(Some(vec![]))
                    } else {
                        Ok(None)
                    }
                } else {
                    Ok(None)
                }
            }
            Pattern::FloatLiteral(expected) => {
                if let Value::Float(actual) = value {
                    if (expected - actual).abs() < f64::EPSILON {
                        Ok(Some(vec![]))
                    } else {
                        Ok(None)
                    }
                } else {
                    Ok(None)
                }
            }
            Pattern::BooleanLiteral(expected) => {
                if let Value::Boolean(actual) = value {
                    if expected == actual {
                        Ok(Some(vec![]))
                    } else {
                        Ok(None)
                    }
                } else {
                    Ok(None)
                }
            }
            Pattern::StringLiteral(expected) => {
                if let Value::String(actual) = value {
                    if expected == actual {
                        Ok(Some(vec![]))
                    } else {
                        Ok(None)
                    }
                } else {
                    Ok(None)
                }
            }
            Pattern::Variable(name) => {
                // Variable pattern always matches and binds the value
                Ok(Some(vec![(name.clone(), value.clone())]))
            }
            Pattern::Wildcard => {
                // Wildcard always matches but doesn't bind
                Ok(Some(vec![]))
            }
        }
    }

    fn eval_tuple_index_expression(
        &mut self,
        tuple_expr: &Expression,
        index: usize,
    ) -> Result<Value, String> {
        let tuple_value = self.eval_expression(tuple_expr)?;

        match tuple_value {
            Value::Tuple(elements) => {
                if index < elements.len() {
                    Ok(elements[index].clone())
                } else {
                    Err(format!(
                        "Tuple index out of bounds: index {} on tuple of length {}",
                        index,
                        elements.len()
                    ))
                }
            }
            _ => Err(format!(
                "Cannot index type {} with tuple index syntax",
                tuple_value.type_name()
            )),
        }
    }

    fn eval_cast_expression(
        &mut self,
        expression: &Expression,
        target_type: &str,
    ) -> Result<Value, String> {
        let value = self.eval_expression(expression)?;

        match target_type {
            "int32" | "int64" => match value {
                Value::Integer(i) => Ok(Value::Integer(i)),
                Value::Float(f) => Ok(Value::Integer(f as i64)),
                Value::Boolean(b) => Ok(Value::Integer(if b { 1 } else { 0 })),
                _ => Err(format!("Cannot cast {} to {}", value.type_name(), target_type)),
            },
            "float32" | "float64" => match value {
                Value::Integer(i) => Ok(Value::Float(i as f64)),
                Value::Float(f) => Ok(Value::Float(f)),
                Value::Boolean(b) => Ok(Value::Float(if b { 1.0 } else { 0.0 })),
                _ => Err(format!("Cannot cast {} to {}", value.type_name(), target_type)),
            },
            "bool" => match value {
                Value::Integer(i) => Ok(Value::Boolean(i != 0)),
                Value::Float(f) => Ok(Value::Boolean(f != 0.0)),
                Value::Boolean(b) => Ok(Value::Boolean(b)),
                _ => Err(format!("Cannot cast {} to {}", value.type_name(), target_type)),
            },
            _ => Err(format!("Unknown cast target type: {}", target_type)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Lexer;
    use crate::parser::Parser;

    fn eval_input(input: &str) -> Result<Value, String> {
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer);
        let program = parser.parse_program().map_err(|e| e.to_string())?;

        let mut interpreter = Interpreter::new();
        interpreter.eval(program)
    }

    #[test]
    fn test_interpreter_creation() {
        let _interpreter = Interpreter::new();
    }

    #[test]
    fn test_eval_integer_literal() {
        let result = eval_input("42").unwrap();
        assert_eq!(result, Value::Integer(42));
    }

    #[test]
    fn test_eval_float_literal() {
        let result = eval_input("3.14").unwrap();
        assert_eq!(result, Value::Float(3.14));
    }

    #[test]
    fn test_eval_boolean_literal() {
        let result = eval_input("true").unwrap();
        assert_eq!(result, Value::Boolean(true));
    }

    #[test]
    fn test_eval_string_literal() {
        let result = eval_input("\"hello\"").unwrap();
        assert_eq!(result, Value::String("hello".to_string()));
    }

    #[test]
    fn test_eval_arithmetic() {
        assert_eq!(eval_input("5 + 3").unwrap(), Value::Integer(8));
        assert_eq!(eval_input("10 - 4").unwrap(), Value::Integer(6));
        assert_eq!(eval_input("6 * 7").unwrap(), Value::Integer(42));
        assert_eq!(eval_input("20 / 4").unwrap(), Value::Integer(5));
        assert_eq!(eval_input("17 % 5").unwrap(), Value::Integer(2));
    }

    #[test]
    fn test_eval_float_arithmetic() {
        assert_eq!(eval_input("2.5 + 1.5").unwrap(), Value::Float(4.0));
        assert_eq!(eval_input("5.0 - 2.0").unwrap(), Value::Float(3.0));
        assert_eq!(eval_input("3.0 * 2.0").unwrap(), Value::Float(6.0));
        assert_eq!(eval_input("9.0 / 3.0").unwrap(), Value::Float(3.0));
    }

    #[test]
    fn test_eval_mixed_arithmetic() {
        assert_eq!(eval_input("5 + 2.5").unwrap(), Value::Float(7.5));
        assert_eq!(eval_input("10.0 - 3").unwrap(), Value::Float(7.0));
    }

    #[test]
    fn test_eval_comparison() {
        assert_eq!(eval_input("5 < 10").unwrap(), Value::Boolean(true));
        assert_eq!(eval_input("5 > 10").unwrap(), Value::Boolean(false));
        assert_eq!(eval_input("5 <= 5").unwrap(), Value::Boolean(true));
        assert_eq!(eval_input("5 >= 6").unwrap(), Value::Boolean(false));
        assert_eq!(eval_input("5 == 5").unwrap(), Value::Boolean(true));
        assert_eq!(eval_input("5 != 10").unwrap(), Value::Boolean(true));
    }

    #[test]
    fn test_eval_logical() {
        assert_eq!(eval_input("true && true").unwrap(), Value::Boolean(true));
        assert_eq!(eval_input("true && false").unwrap(), Value::Boolean(false));
        assert_eq!(eval_input("false || true").unwrap(), Value::Boolean(true));
        assert_eq!(eval_input("false || false").unwrap(), Value::Boolean(false));
    }

    #[test]
    fn test_eval_unary() {
        assert_eq!(eval_input("-5").unwrap(), Value::Integer(-5));
        assert_eq!(eval_input("-3.14").unwrap(), Value::Float(-3.14));
        assert_eq!(eval_input("!true").unwrap(), Value::Boolean(false));
        assert_eq!(eval_input("!false").unwrap(), Value::Boolean(true));
    }

    #[test]
    fn test_eval_let_statement() {
        let result = eval_input("let x = 42\nx").unwrap();
        assert_eq!(result, Value::Integer(42));
    }

    #[test]
    fn test_eval_let_with_expression() {
        let result = eval_input("let x = 5 + 3\nx").unwrap();
        assert_eq!(result, Value::Integer(8));
    }

    #[test]
    fn test_eval_array_literal() {
        let result = eval_input("[1, 2, 3]").unwrap();
        assert_eq!(
            result,
            Value::Array(vec![
                Value::Integer(1),
                Value::Integer(2),
                Value::Integer(3)
            ])
        );
    }

    #[test]
    fn test_eval_array_index() {
        let result = eval_input("let arr = [1, 2, 3]\narr[1]").unwrap();
        assert_eq!(result, Value::Integer(2));
    }

    #[test]
    fn test_eval_array_negative_index() {
        let result = eval_input("let arr = [1, 2, 3]\narr[-1]").unwrap();
        assert_eq!(result, Value::Integer(3));
    }

    #[test]
    fn test_eval_function_declaration() {
        let input = "fn add(x: int32, y: int32) -> int32 { return x + y }\nadd(5, 3)";
        let result = eval_input(input).unwrap();
        assert_eq!(result, Value::Integer(8));
    }

    #[test]
    fn test_eval_function_with_closure() {
        let input = r#"
            let x = 10
            fn add_x(y: int32) -> int32 { return x + y }
            add_x(5)
        "#;
        let result = eval_input(input).unwrap();
        assert_eq!(result, Value::Integer(15));
    }

    #[test]
    fn test_eval_nested_function_calls() {
        let input = r#"
            fn double(x: int32) -> int32 { return x * 2 }
            fn add(x: int32, y: int32) -> int32 { return x + y }
            add(double(3), double(4))
        "#;
        let result = eval_input(input).unwrap();
        assert_eq!(result, Value::Integer(14));
    }

    #[test]
    fn test_error_division_by_zero() {
        let result = eval_input("10 / 0");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Division by zero"));
    }

    #[test]
    fn test_error_undefined_variable() {
        let result = eval_input("x + 5");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Undefined variable"));
    }

    #[test]
    fn test_error_wrong_argument_count() {
        let input = "fn add(x: int32, y: int32) -> int32 { return x + y }\nadd(5)";
        let result = eval_input(input);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Wrong number of arguments"));
    }
}
