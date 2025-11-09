// Symbolic AI Builtin Functions
// Phase 1 - Backend Exposure: First-Order Logic, Rule Engine, Concept Learning, Differentiable Logic
//
// This module exposes the Symbolic AI backend to the Charl language.
// It follows the Karpathy paradigm: minimal, composable functions.

use crate::interpreter::Value;
use crate::symbolic::{
    // FOL
    Term, Formula, FOLSolver, unify,
    // Rule Engine
    Rule, RuleEngine,
    // Concept Learning
    Concept, ConceptGraph,
    // Differentiable Logic
    FuzzyValue, FuzzyLogic, soft_unify,
};

/// Builtin function type
pub type BuiltinFn = fn(Vec<Value>) -> Result<Value, String>;

// ===================================================================
// FIRST-ORDER LOGIC (FOL) - 12 functions
// ===================================================================

/// fol_variable(name: string) -> Term
/// Create a FOL variable term
pub fn builtin_fol_variable(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("fol_variable() expects 1 argument: fol_variable(name)".to_string());
    }

    match &args[0] {
        Value::String(name) => {
            let term = Term::variable(name.clone());
            Ok(Value::FOLTerm(Box::new(term)))
        }
        _ => Err(format!(
            "fol_variable() expects a string, got {}",
            args[0].type_name()
        )),
    }
}

/// fol_constant(name: string) -> Term
/// Create a FOL constant term
pub fn builtin_fol_constant(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("fol_constant() expects 1 argument: fol_constant(name)".to_string());
    }

    match &args[0] {
        Value::String(name) => {
            let term = Term::constant(name.clone());
            Ok(Value::FOLTerm(Box::new(term)))
        }
        _ => Err(format!(
            "fol_constant() expects a string, got {}",
            args[0].type_name()
        )),
    }
}

/// fol_function(name: string, args: [Term]) -> Term
/// Create a FOL function term
pub fn builtin_fol_function(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("fol_function() expects 2 arguments: fol_function(name, args)".to_string());
    }

    let name = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err(format!(
            "fol_function() expects name to be string, got {}",
            args[0].type_name()
        )),
    };

    let term_args = match &args[1] {
        Value::Array(arr) => {
            let mut terms = Vec::new();
            for val in arr {
                match val {
                    Value::FOLTerm(term) => terms.push((**term).clone()),
                    _ => return Err(format!(
                        "fol_function() expects array of Terms, got {}",
                        val.type_name()
                    )),
                }
            }
            terms
        }
        _ => return Err(format!(
            "fol_function() expects args to be array, got {}",
            args[1].type_name()
        )),
    };

    let term = Term::function(name, term_args);
    Ok(Value::FOLTerm(Box::new(term)))
}

/// fol_predicate(name: string, args: [Term]) -> Formula
/// Create a FOL predicate formula
pub fn builtin_fol_predicate(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("fol_predicate() expects 2 arguments: fol_predicate(name, args)".to_string());
    }

    let name = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err(format!(
            "fol_predicate() expects name to be string, got {}",
            args[0].type_name()
        )),
    };

    let term_args = match &args[1] {
        Value::Array(arr) => {
            let mut terms = Vec::new();
            for val in arr {
                match val {
                    Value::FOLTerm(term) => terms.push((**term).clone()),
                    _ => return Err(format!(
                        "fol_predicate() expects array of Terms, got {}",
                        val.type_name()
                    )),
                }
            }
            terms
        }
        _ => return Err(format!(
            "fol_predicate() expects args to be array, got {}",
            args[1].type_name()
        )),
    };

    let formula = Formula::predicate(name, term_args);
    Ok(Value::FOLFormula(Box::new(formula)))
}

/// fol_not(formula: Formula) -> Formula
/// Create a negation formula: ¬φ
pub fn builtin_fol_not(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("fol_not() expects 1 argument: fol_not(formula)".to_string());
    }

    match &args[0] {
        Value::FOLFormula(formula) => {
            let result = Formula::not((**formula).clone());
            Ok(Value::FOLFormula(Box::new(result)))
        }
        _ => Err(format!(
            "fol_not() expects a Formula, got {}",
            args[0].type_name()
        )),
    }
}

/// fol_and(left: Formula, right: Formula) -> Formula
/// Create a conjunction formula: φ ∧ ψ
pub fn builtin_fol_and(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("fol_and() expects 2 arguments: fol_and(left, right)".to_string());
    }

    match (&args[0], &args[1]) {
        (Value::FOLFormula(left), Value::FOLFormula(right)) => {
            let result = Formula::and((**left).clone(), (**right).clone());
            Ok(Value::FOLFormula(Box::new(result)))
        }
        _ => Err("fol_and() expects two Formulas".to_string()),
    }
}

/// fol_or(left: Formula, right: Formula) -> Formula
/// Create a disjunction formula: φ ∨ ψ
pub fn builtin_fol_or(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("fol_or() expects 2 arguments: fol_or(left, right)".to_string());
    }

    match (&args[0], &args[1]) {
        (Value::FOLFormula(left), Value::FOLFormula(right)) => {
            let result = Formula::or((**left).clone(), (**right).clone());
            Ok(Value::FOLFormula(Box::new(result)))
        }
        _ => Err("fol_or() expects two Formulas".to_string()),
    }
}

/// fol_implies(left: Formula, right: Formula) -> Formula
/// Create an implication formula: φ → ψ
pub fn builtin_fol_implies(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("fol_implies() expects 2 arguments: fol_implies(antecedent, consequent)".to_string());
    }

    match (&args[0], &args[1]) {
        (Value::FOLFormula(left), Value::FOLFormula(right)) => {
            let result = Formula::implies((**left).clone(), (**right).clone());
            Ok(Value::FOLFormula(Box::new(result)))
        }
        _ => Err("fol_implies() expects two Formulas".to_string()),
    }
}

/// fol_forall(variable: string, formula: Formula) -> Formula
/// Create a universal quantification formula: ∀x φ
pub fn builtin_fol_forall(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("fol_forall() expects 2 arguments: fol_forall(variable, formula)".to_string());
    }

    let var = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err(format!(
            "fol_forall() expects variable to be string, got {}",
            args[0].type_name()
        )),
    };

    match &args[1] {
        Value::FOLFormula(formula) => {
            let result = Formula::forall(var, (**formula).clone());
            Ok(Value::FOLFormula(Box::new(result)))
        }
        _ => Err(format!(
            "fol_forall() expects a Formula, got {}",
            args[1].type_name()
        )),
    }
}

/// fol_exists(variable: string, formula: Formula) -> Formula
/// Create an existential quantification formula: ∃x φ
pub fn builtin_fol_exists(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("fol_exists() expects 2 arguments: fol_exists(variable, formula)".to_string());
    }

    let var = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err(format!(
            "fol_exists() expects variable to be string, got {}",
            args[0].type_name()
        )),
    };

    match &args[1] {
        Value::FOLFormula(formula) => {
            let result = Formula::exists(var, (**formula).clone());
            Ok(Value::FOLFormula(Box::new(result)))
        }
        _ => Err(format!(
            "fol_exists() expects a Formula, got {}",
            args[1].type_name()
        )),
    }
}

/// fol_unify(term1: Term, term2: Term) -> bool
/// Unify two terms and return true if they unify
pub fn builtin_fol_unify(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("fol_unify() expects 2 arguments: fol_unify(term1, term2)".to_string());
    }

    match (&args[0], &args[1]) {
        (Value::FOLTerm(t1), Value::FOLTerm(t2)) => {
            let result = unify(t1, t2);
            Ok(Value::Boolean(result.is_success()))
        }
        _ => Err("fol_unify() expects two Terms".to_string()),
    }
}

/// fol_solver_create() -> FOLSolver
/// Create a new FOL solver with empty knowledge base
pub fn builtin_fol_solver_create(args: Vec<Value>) -> Result<Value, String> {
    if !args.is_empty() {
        return Err("fol_solver_create() expects 0 arguments".to_string());
    }

    let solver = FOLSolver::new();
    Ok(Value::FOLSolver(Box::new(solver)))
}

/// fol_solver_add_fact(solver: FOLSolver, formula: Formula) -> FOLSolver
/// Add a fact to the knowledge base
pub fn builtin_fol_solver_add_fact(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("fol_solver_add_fact() expects 2 arguments: fol_solver_add_fact(solver, formula)".to_string());
    }

    let mut solver = match &args[0] {
        Value::FOLSolver(s) => (**s).clone(),
        _ => return Err(format!(
            "fol_solver_add_fact() expects FOLSolver, got {}",
            args[0].type_name()
        )),
    };

    match &args[1] {
        Value::FOLFormula(formula) => {
            solver.add_fact((**formula).clone());
            Ok(Value::FOLSolver(Box::new(solver)))
        }
        _ => Err(format!(
            "fol_solver_add_fact() expects Formula, got {}",
            args[1].type_name()
        )),
    }
}

/// fol_solver_add_rule(solver: FOLSolver, head: Formula, body: [Formula]) -> FOLSolver
/// Add a rule to the knowledge base
pub fn builtin_fol_solver_add_rule(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 {
        return Err("fol_solver_add_rule() expects 3 arguments: fol_solver_add_rule(solver, head, body)".to_string());
    }

    let mut solver = match &args[0] {
        Value::FOLSolver(s) => (**s).clone(),
        _ => return Err(format!(
            "fol_solver_add_rule() expects FOLSolver, got {}",
            args[0].type_name()
        )),
    };

    let head = match &args[1] {
        Value::FOLFormula(f) => (**f).clone(),
        _ => return Err(format!(
            "fol_solver_add_rule() expects head to be Formula, got {}",
            args[1].type_name()
        )),
    };

    let body = match &args[2] {
        Value::Array(arr) => {
            let mut formulas = Vec::new();
            for val in arr {
                match val {
                    Value::FOLFormula(f) => formulas.push((**f).clone()),
                    _ => return Err(format!(
                        "fol_solver_add_rule() expects body to be array of Formulas, got {}",
                        val.type_name()
                    )),
                }
            }
            formulas
        }
        _ => return Err(format!(
            "fol_solver_add_rule() expects body to be array, got {}",
            args[2].type_name()
        )),
    };

    solver.add_rule(head, body);
    Ok(Value::FOLSolver(Box::new(solver)))
}

/// fol_solver_prove(solver: FOLSolver, goal: Formula) -> bool
/// Prove that a goal follows from the knowledge base
pub fn builtin_fol_solver_prove(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("fol_solver_prove() expects 2 arguments: fol_solver_prove(solver, goal)".to_string());
    }

    let mut solver = match &args[0] {
        Value::FOLSolver(s) => (**s).clone(),
        _ => return Err(format!(
            "fol_solver_prove() expects FOLSolver, got {}",
            args[0].type_name()
        )),
    };

    match &args[1] {
        Value::FOLFormula(goal) => {
            let result = solver.prove(goal);
            Ok(Value::Boolean(result))
        }
        _ => Err(format!(
            "fol_solver_prove() expects Formula, got {}",
            args[1].type_name()
        )),
    }
}

// ===================================================================
// RULE ENGINE - 8 functions
// ===================================================================

/// rule_engine_create() -> RuleEngine
/// Create a new rule engine
pub fn builtin_rule_engine_create(args: Vec<Value>) -> Result<Value, String> {
    if !args.is_empty() {
        return Err("rule_engine_create() expects 0 arguments".to_string());
    }

    let engine = RuleEngine::new();
    Ok(Value::RuleEngine(Box::new(engine)))
}

/// rule_create(name: string) -> Rule
/// Create a new rule with given name
pub fn builtin_rule_create(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("rule_create() expects 1 argument: rule_create(name)".to_string());
    }

    match &args[0] {
        Value::String(name) => {
            let rule = Rule::new(name.clone());
            Ok(Value::SymbolicRule(Box::new(rule)))
        }
        _ => Err(format!(
            "rule_create() expects a string, got {}",
            args[0].type_name()
        )),
    }
}

// Note: More complex rule building functions will be added as needed
// For now, users can use the simple API above

// ===================================================================
// CONCEPT LEARNING - 10 functions
// ===================================================================

/// concept_create(name: string) -> Concept
/// concept_create(name: string, properties: [string]) -> Concept (legacy)
/// Create a new concept (accepts 1 or 2 arguments for compatibility)
pub fn builtin_concept_create(args: Vec<Value>) -> Result<Value, String> {
    if args.is_empty() || args.len() > 2 {
        return Err("concept_create() expects 1-2 arguments: concept_create(name) or concept_create(name, properties)".to_string());
    }

    let name = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err(format!(
            "concept_create() expects name to be string, got {}",
            args[0].type_name()
        )),
    };

    let mut concept = Concept::new(name);

    // If properties array provided (legacy compatibility)
    if args.len() == 2 {
        match &args[1] {
            Value::Array(arr) => {
                for prop in arr {
                    match prop {
                        Value::String(s) => {
                            // Legacy: properties as strings have default strength 1.0
                            concept = concept.with_property(s.clone(), 1.0);
                        }
                        _ => return Err("concept_create() properties must be strings".to_string()),
                    }
                }
            }
            _ => return Err("concept_create() second argument must be array of properties".to_string()),
        }
    }

    Ok(Value::Concept(Box::new(concept)))
}

/// concept_add_property(concept: Concept, property: string, strength: float) -> Concept
/// Add a property to a concept with given strength (0-1)
pub fn builtin_concept_add_property(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 {
        return Err("concept_add_property() expects 3 arguments: concept_add_property(concept, property, strength)".to_string());
    }

    let concept = match &args[0] {
        Value::Concept(c) => (**c).clone(),
        _ => return Err(format!(
            "concept_add_property() expects Concept, got {}",
            args[0].type_name()
        )),
    };

    let property = match &args[1] {
        Value::String(s) => s.clone(),
        _ => return Err(format!(
            "concept_add_property() expects property to be string, got {}",
            args[1].type_name()
        )),
    };

    let strength = match &args[2] {
        Value::Float(f) => *f,
        Value::Integer(i) => *i as f64,
        _ => return Err(format!(
            "concept_add_property() expects strength to be float, got {}",
            args[2].type_name()
        )),
    };

    let result = concept.with_property(property, strength);
    Ok(Value::Concept(Box::new(result)))
}

/// concept_similarity(concept1: Concept, concept2: Concept) -> float
/// Compute similarity between two concepts (0-1)
pub fn builtin_concept_similarity(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("concept_similarity() expects 2 arguments: concept_similarity(c1, c2)".to_string());
    }

    match (&args[0], &args[1]) {
        (Value::Concept(c1), Value::Concept(c2)) => {
            let similarity = c1.similarity(c2);
            Ok(Value::Float(similarity))
        }
        _ => Err("concept_similarity() expects two Concepts".to_string()),
    }
}

/// concept_graph_create() -> ConceptGraph
/// Create a new concept graph
pub fn builtin_concept_graph_create(args: Vec<Value>) -> Result<Value, String> {
    if !args.is_empty() {
        return Err("concept_graph_create() expects 0 arguments".to_string());
    }

    let graph = ConceptGraph::new();
    Ok(Value::ConceptGraph(Box::new(graph)))
}

/// concept_graph_add(graph: ConceptGraph, concept: Concept) -> ConceptGraph
/// Add a concept to the graph
pub fn builtin_concept_graph_add(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("concept_graph_add() expects 2 arguments: concept_graph_add(graph, concept)".to_string());
    }

    let mut graph = match &args[0] {
        Value::ConceptGraph(g) => (**g).clone(),
        _ => return Err(format!(
            "concept_graph_add() expects ConceptGraph, got {}",
            args[0].type_name()
        )),
    };

    match &args[1] {
        Value::Concept(concept) => {
            graph.add_concept((**concept).clone());
            Ok(Value::ConceptGraph(Box::new(graph)))
        }
        _ => Err(format!(
            "concept_graph_add() expects Concept, got {}",
            args[1].type_name()
        )),
    }
}

// ===================================================================
// DIFFERENTIABLE LOGIC - 8 functions
// ===================================================================

/// fuzzy_value(value: float) -> FuzzyValue
/// Create a fuzzy truth value (0-1)
pub fn builtin_fuzzy_value(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("fuzzy_value() expects 1 argument: fuzzy_value(value)".to_string());
    }

    let value = match &args[0] {
        Value::Float(f) => *f,
        Value::Integer(i) => *i as f64,
        Value::Boolean(b) => if *b { 1.0 } else { 0.0 },
        _ => return Err(format!(
            "fuzzy_value() expects float, got {}",
            args[0].type_name()
        )),
    };

    let fuzzy = FuzzyValue::new(value);
    Ok(Value::FuzzyValue(Box::new(fuzzy)))
}

/// fuzzy_not(value: FuzzyValue) -> FuzzyValue
/// Fuzzy negation: ¬p = 1 - p
pub fn builtin_fuzzy_not(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("fuzzy_not() expects 1 argument: fuzzy_not(value)".to_string());
    }

    match &args[0] {
        Value::FuzzyValue(v) => {
            let result = FuzzyLogic::not(**v);
            Ok(Value::FuzzyValue(Box::new(result)))
        }
        _ => Err(format!(
            "fuzzy_not() expects FuzzyValue, got {}",
            args[0].type_name()
        )),
    }
}

/// fuzzy_and(a: FuzzyValue, b: FuzzyValue) -> FuzzyValue
/// Fuzzy AND: a ∧ b (product t-norm)
pub fn builtin_fuzzy_and(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("fuzzy_and() expects 2 arguments: fuzzy_and(a, b)".to_string());
    }

    match (&args[0], &args[1]) {
        (Value::FuzzyValue(a), Value::FuzzyValue(b)) => {
            let result = FuzzyLogic::and(**a, **b);
            Ok(Value::FuzzyValue(Box::new(result)))
        }
        _ => Err("fuzzy_and() expects two FuzzyValues".to_string()),
    }
}

/// fuzzy_or(a: FuzzyValue, b: FuzzyValue) -> FuzzyValue
/// Fuzzy OR: a ∨ b (probabilistic sum)
pub fn builtin_fuzzy_or(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("fuzzy_or() expects 2 arguments: fuzzy_or(a, b)".to_string());
    }

    match (&args[0], &args[1]) {
        (Value::FuzzyValue(a), Value::FuzzyValue(b)) => {
            let result = FuzzyLogic::or(**a, **b);
            Ok(Value::FuzzyValue(Box::new(result)))
        }
        _ => Err("fuzzy_or() expects two FuzzyValues".to_string()),
    }
}

/// fuzzy_implies(a: FuzzyValue, b: FuzzyValue) -> FuzzyValue
/// Fuzzy implication: a → b = ¬a ∨ b
pub fn builtin_fuzzy_implies(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("fuzzy_implies() expects 2 arguments: fuzzy_implies(a, b)".to_string());
    }

    match (&args[0], &args[1]) {
        (Value::FuzzyValue(a), Value::FuzzyValue(b)) => {
            let result = FuzzyLogic::implies(**a, **b);
            Ok(Value::FuzzyValue(Box::new(result)))
        }
        _ => Err("fuzzy_implies() expects two FuzzyValues".to_string()),
    }
}

/// fuzzy_to_bool(value: FuzzyValue) -> bool
/// Convert fuzzy value to boolean (>= 0.5 = true)
pub fn builtin_fuzzy_to_bool(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("fuzzy_to_bool() expects 1 argument: fuzzy_to_bool(value)".to_string());
    }

    match &args[0] {
        Value::FuzzyValue(v) => {
            let result = v.to_bool();
            Ok(Value::Boolean(result))
        }
        _ => Err(format!(
            "fuzzy_to_bool() expects FuzzyValue, got {}",
            args[0].type_name()
        )),
    }
}

/// fuzzy_to_float(value: FuzzyValue) -> float
/// Get the numeric value of a fuzzy truth value
pub fn builtin_fuzzy_to_float(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("fuzzy_to_float() expects 1 argument: fuzzy_to_float(value)".to_string());
    }

    match &args[0] {
        Value::FuzzyValue(v) => {
            let result = v.value();
            Ok(Value::Float(result))
        }
        _ => Err(format!(
            "fuzzy_to_float() expects FuzzyValue, got {}",
            args[0].type_name()
        )),
    }
}

/// soft_unify_strings(term1: string, term2: string) -> FuzzyValue
/// Soft unification: returns degree of match (0-1) instead of binary success/failure
pub fn builtin_soft_unify_strings(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("soft_unify_strings() expects 2 arguments: soft_unify_strings(term1, term2)".to_string());
    }

    match (&args[0], &args[1]) {
        (Value::String(s1), Value::String(s2)) => {
            let result = soft_unify(s1, s2);
            Ok(Value::FuzzyValue(Box::new(result)))
        }
        _ => Err("soft_unify_strings() expects two strings".to_string()),
    }
}
