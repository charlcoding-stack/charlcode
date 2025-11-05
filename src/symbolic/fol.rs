// First-Order Logic (FOL) Solver
// Prolog-like inference engine for symbolic reasoning
//
// This module implements:
// - FOL terms (variables, constants, functions)
// - FOL formulas (predicates, quantifiers, connectives)
// - Unification algorithm
// - Resolution-based theorem proving
// - Integration with knowledge graphs
//
// Usage:
// ```rust
// use charl::symbolic::fol::{Term, Formula, FOLSolver};
//
// // Create terms
// let x = Term::variable("X");
// let john = Term::constant("john");
//
// // Create formula: mortal(X) :- human(X)
// let rule = Formula::implies(
//     Formula::predicate("human", vec![x.clone()]),
//     Formula::predicate("mortal", vec![x.clone()]),
// );
//
// // Prove query
// let mut solver = FOLSolver::new();
// solver.add_rule(rule);
// let result = solver.prove(&Formula::predicate("mortal", vec![john]))?;
// ```

use std::collections::{HashMap, HashSet};
use std::fmt;

/// FOL Term: variables, constants, or function applications
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Term {
    /// Variable (starts with uppercase or _)
    Variable(String),

    /// Constant (specific value)
    Constant(String),

    /// Function application: f(t1, t2, ..., tn)
    Function {
        name: String,
        args: Vec<Term>,
    },
}

impl Term {
    /// Create a variable term
    pub fn variable(name: impl Into<String>) -> Self {
        Term::Variable(name.into())
    }

    /// Create a constant term
    pub fn constant(name: impl Into<String>) -> Self {
        Term::Constant(name.into())
    }

    /// Create a function term
    pub fn function(name: impl Into<String>, args: Vec<Term>) -> Self {
        Term::Function {
            name: name.into(),
            args,
        }
    }

    /// Check if term is a variable
    pub fn is_variable(&self) -> bool {
        matches!(self, Term::Variable(_))
    }

    /// Check if term is ground (contains no variables)
    pub fn is_ground(&self) -> bool {
        match self {
            Term::Variable(_) => false,
            Term::Constant(_) => true,
            Term::Function { args, .. } => args.iter().all(|arg| arg.is_ground()),
        }
    }

    /// Get all variables in this term
    pub fn variables(&self) -> HashSet<String> {
        let mut vars = HashSet::new();
        self.collect_variables(&mut vars);
        vars
    }

    fn collect_variables(&self, vars: &mut HashSet<String>) {
        match self {
            Term::Variable(name) => {
                vars.insert(name.clone());
            }
            Term::Constant(_) => {}
            Term::Function { args, .. } => {
                for arg in args {
                    arg.collect_variables(vars);
                }
            }
        }
    }

    /// Apply substitution to term
    pub fn substitute(&self, subst: &Substitution) -> Term {
        match self {
            Term::Variable(name) => {
                subst.get(name).cloned().unwrap_or_else(|| self.clone())
            }
            Term::Constant(_) => self.clone(),
            Term::Function { name, args } => {
                Term::Function {
                    name: name.clone(),
                    args: args.iter().map(|arg| arg.substitute(subst)).collect(),
                }
            }
        }
    }
}

impl fmt::Display for Term {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Term::Variable(name) => write!(f, "{}", name),
            Term::Constant(name) => write!(f, "{}", name),
            Term::Function { name, args } => {
                write!(f, "{}(", name)?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", arg)?;
                }
                write!(f, ")")
            }
        }
    }
}

/// FOL Formula: logical expressions
#[derive(Debug, Clone, PartialEq)]
pub enum Formula {
    /// Atomic predicate: P(t1, t2, ..., tn)
    Predicate {
        name: String,
        args: Vec<Term>,
    },

    /// Negation: ¬φ
    Not(Box<Formula>),

    /// Conjunction: φ ∧ ψ
    And(Box<Formula>, Box<Formula>),

    /// Disjunction: φ ∨ ψ
    Or(Box<Formula>, Box<Formula>),

    /// Implication: φ → ψ
    Implies(Box<Formula>, Box<Formula>),

    /// Universal quantification: ∀x φ
    ForAll {
        variable: String,
        formula: Box<Formula>,
    },

    /// Existential quantification: ∃x φ
    Exists {
        variable: String,
        formula: Box<Formula>,
    },

    /// Truth constants
    True,
    False,
}

impl Formula {
    /// Create a predicate formula
    pub fn predicate(name: impl Into<String>, args: Vec<Term>) -> Self {
        Formula::Predicate {
            name: name.into(),
            args,
        }
    }

    /// Create a negation
    pub fn not(formula: Formula) -> Self {
        Formula::Not(Box::new(formula))
    }

    /// Create a conjunction
    pub fn and(left: Formula, right: Formula) -> Self {
        Formula::And(Box::new(left), Box::new(right))
    }

    /// Create a disjunction
    pub fn or(left: Formula, right: Formula) -> Self {
        Formula::Or(Box::new(left), Box::new(right))
    }

    /// Create an implication
    pub fn implies(antecedent: Formula, consequent: Formula) -> Self {
        Formula::Implies(Box::new(antecedent), Box::new(consequent))
    }

    /// Create universal quantification
    pub fn forall(variable: impl Into<String>, formula: Formula) -> Self {
        Formula::ForAll {
            variable: variable.into(),
            formula: Box::new(formula),
        }
    }

    /// Create existential quantification
    pub fn exists(variable: impl Into<String>, formula: Formula) -> Self {
        Formula::Exists {
            variable: variable.into(),
            formula: Box::new(formula),
        }
    }

    /// Get all free variables in formula
    pub fn free_variables(&self) -> HashSet<String> {
        let mut vars = HashSet::new();
        self.collect_free_variables(&mut vars, &HashSet::new());
        vars
    }

    fn collect_free_variables(&self, free: &mut HashSet<String>, bound: &HashSet<String>) {
        match self {
            Formula::Predicate { args, .. } => {
                for arg in args {
                    for var in arg.variables() {
                        if !bound.contains(&var) {
                            free.insert(var);
                        }
                    }
                }
            }
            Formula::Not(f) => f.collect_free_variables(free, bound),
            Formula::And(left, right) | Formula::Or(left, right) | Formula::Implies(left, right) => {
                left.collect_free_variables(free, bound);
                right.collect_free_variables(free, bound);
            }
            Formula::ForAll { variable, formula } | Formula::Exists { variable, formula } => {
                let mut new_bound = bound.clone();
                new_bound.insert(variable.clone());
                formula.collect_free_variables(free, &new_bound);
            }
            Formula::True | Formula::False => {}
        }
    }

    /// Apply substitution to formula
    pub fn substitute(&self, subst: &Substitution) -> Formula {
        match self {
            Formula::Predicate { name, args } => {
                Formula::Predicate {
                    name: name.clone(),
                    args: args.iter().map(|arg| arg.substitute(subst)).collect(),
                }
            }
            Formula::Not(f) => Formula::Not(Box::new(f.substitute(subst))),
            Formula::And(left, right) => {
                Formula::And(
                    Box::new(left.substitute(subst)),
                    Box::new(right.substitute(subst)),
                )
            }
            Formula::Or(left, right) => {
                Formula::Or(
                    Box::new(left.substitute(subst)),
                    Box::new(right.substitute(subst)),
                )
            }
            Formula::Implies(left, right) => {
                Formula::Implies(
                    Box::new(left.substitute(subst)),
                    Box::new(right.substitute(subst)),
                )
            }
            Formula::ForAll { variable, formula } => {
                // Don't substitute bound variables
                let mut new_subst = subst.clone();
                new_subst.remove(variable);
                Formula::ForAll {
                    variable: variable.clone(),
                    formula: Box::new(formula.substitute(&new_subst)),
                }
            }
            Formula::Exists { variable, formula } => {
                let mut new_subst = subst.clone();
                new_subst.remove(variable);
                Formula::Exists {
                    variable: variable.clone(),
                    formula: Box::new(formula.substitute(&new_subst)),
                }
            }
            Formula::True | Formula::False => self.clone(),
        }
    }
}

impl fmt::Display for Formula {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Formula::Predicate { name, args } => {
                write!(f, "{}(", name)?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", arg)?;
                }
                write!(f, ")")
            }
            Formula::Not(formula) => write!(f, "¬{}", formula),
            Formula::And(left, right) => write!(f, "({} ∧ {})", left, right),
            Formula::Or(left, right) => write!(f, "({} ∨ {})", left, right),
            Formula::Implies(left, right) => write!(f, "({} → {})", left, right),
            Formula::ForAll { variable, formula } => write!(f, "∀{} {}", variable, formula),
            Formula::Exists { variable, formula } => write!(f, "∃{} {}", variable, formula),
            Formula::True => write!(f, "⊤"),
            Formula::False => write!(f, "⊥"),
        }
    }
}

/// Substitution: maps variables to terms
pub type Substitution = HashMap<String, Term>;

/// Unification result
#[derive(Debug, Clone)]
pub enum UnificationResult {
    Success(Substitution),
    Failure,
}

impl UnificationResult {
    pub fn is_success(&self) -> bool {
        matches!(self, UnificationResult::Success(_))
    }

    pub fn substitution(&self) -> Option<&Substitution> {
        match self {
            UnificationResult::Success(subst) => Some(subst),
            UnificationResult::Failure => None,
        }
    }
}

/// Unify two terms
pub fn unify(t1: &Term, t2: &Term) -> UnificationResult {
    let mut subst = HashMap::new();
    if unify_terms(t1, t2, &mut subst) {
        UnificationResult::Success(subst)
    } else {
        UnificationResult::Failure
    }
}

fn unify_terms(t1: &Term, t2: &Term, subst: &mut Substitution) -> bool {
    // Apply current substitution
    let t1 = t1.substitute(subst);
    let t2 = t2.substitute(subst);

    // Same term
    if t1 == t2 {
        return true;
    }

    match (&t1, &t2) {
        // Variable unification
        (Term::Variable(v), t) | (t, Term::Variable(v)) => {
            // Occurs check
            if t.variables().contains(v) {
                return false;
            }
            subst.insert(v.clone(), t.clone());
            true
        }

        // Function unification
        (Term::Function { name: n1, args: args1 }, Term::Function { name: n2, args: args2 }) => {
            if n1 != n2 || args1.len() != args2.len() {
                return false;
            }

            for (arg1, arg2) in args1.iter().zip(args2.iter()) {
                if !unify_terms(arg1, arg2, subst) {
                    return false;
                }
            }

            true
        }

        // Mismatch
        _ => false,
    }
}

/// Clause: Horn clause for resolution
#[derive(Debug, Clone)]
pub struct Clause {
    /// Head of the clause (consequent)
    pub head: Formula,

    /// Body of the clause (antecedents)
    pub body: Vec<Formula>,
}

impl Clause {
    /// Create a fact (clause with no body)
    pub fn fact(head: Formula) -> Self {
        Clause {
            head,
            body: Vec::new(),
        }
    }

    /// Create a rule (clause with body)
    pub fn rule(head: Formula, body: Vec<Formula>) -> Self {
        Clause { head, body }
    }

    /// Check if this is a fact (no body)
    pub fn is_fact(&self) -> bool {
        self.body.is_empty()
    }

    /// Get all variables in clause
    pub fn variables(&self) -> HashSet<String> {
        let mut vars = self.head.free_variables();
        for formula in &self.body {
            vars.extend(formula.free_variables());
        }
        vars
    }

    /// Rename variables to avoid conflicts
    pub fn rename_variables(&self, suffix: &str) -> Clause {
        let vars = self.variables();
        let mut subst = HashMap::new();

        for var in vars {
            subst.insert(var.clone(), Term::variable(format!("{}_{}", var, suffix)));
        }

        Clause {
            head: self.head.substitute(&subst),
            body: self.body.iter().map(|f| f.substitute(&subst)).collect(),
        }
    }
}

impl fmt::Display for Clause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.head)?;
        if !self.body.is_empty() {
            write!(f, " :- ")?;
            for (i, formula) in self.body.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", formula)?;
            }
        }
        write!(f, ".")
    }
}

/// FOL Solver: Prolog-like inference engine
pub struct FOLSolver {
    /// Knowledge base (clauses)
    clauses: Vec<Clause>,

    /// Renaming counter for variable conflicts
    rename_counter: usize,
}

impl FOLSolver {
    /// Create a new FOL solver
    pub fn new() -> Self {
        FOLSolver {
            clauses: Vec::new(),
            rename_counter: 0,
        }
    }

    /// Add a clause to the knowledge base
    pub fn add_clause(&mut self, clause: Clause) {
        self.clauses.push(clause);
    }

    /// Add a fact
    pub fn add_fact(&mut self, head: Formula) {
        self.add_clause(Clause::fact(head));
    }

    /// Add a rule
    pub fn add_rule(&mut self, head: Formula, body: Vec<Formula>) {
        self.add_clause(Clause::rule(head, body));
    }

    /// Query the knowledge base
    /// Returns all substitutions that satisfy the query
    pub fn query(&mut self, goal: &Formula) -> Vec<Substitution> {
        let mut results = Vec::new();
        self.solve(vec![goal.clone()], &HashMap::new(), &mut results);
        results
    }

    /// Prove that a goal follows from the knowledge base
    pub fn prove(&mut self, goal: &Formula) -> bool {
        !self.query(goal).is_empty()
    }

    /// SLD resolution: solve goals with current substitution
    fn solve(&mut self, goals: Vec<Formula>, subst: &Substitution, results: &mut Vec<Substitution>) {
        // Base case: all goals satisfied
        if goals.is_empty() {
            results.push(subst.clone());
            return;
        }

        // Select first goal
        let goal = &goals[0];
        let remaining_goals = &goals[1..];

        // Try to unify with each clause
        for clause in self.clauses.clone() {
            // Rename variables to avoid conflicts
            let renamed_clause = clause.rename_variables(&self.rename_counter.to_string());
            self.rename_counter += 1;

            // Try to unify goal with clause head
            let unification = unify_formulas(goal, &renamed_clause.head);

            if let UnificationResult::Success(mgu) = unification {
                // Compose substitutions
                let mut new_subst = subst.clone();
                for (var, term) in mgu {
                    new_subst.insert(var, term.substitute(&new_subst));
                }

                // Add clause body to goals
                let mut new_goals = renamed_clause.body.clone();
                new_goals.extend(remaining_goals.iter().cloned());

                // Apply substitution to new goals
                let new_goals: Vec<_> = new_goals.iter()
                    .map(|g| g.substitute(&new_subst))
                    .collect();

                // Recursively solve new goals
                self.solve(new_goals, &new_subst, results);
            }
        }
    }
}

impl Default for FOLSolver {
    fn default() -> Self {
        Self::new()
    }
}

/// Unify two formulas (must be predicates)
fn unify_formulas(f1: &Formula, f2: &Formula) -> UnificationResult {
    match (f1, f2) {
        (Formula::Predicate { name: n1, args: args1 },
         Formula::Predicate { name: n2, args: args2 }) => {
            if n1 != n2 || args1.len() != args2.len() {
                return UnificationResult::Failure;
            }

            let mut subst = HashMap::new();
            for (arg1, arg2) in args1.iter().zip(args2.iter()) {
                if !unify_terms(arg1, arg2, &mut subst) {
                    return UnificationResult::Failure;
                }
            }

            UnificationResult::Success(subst)
        }
        _ => UnificationResult::Failure,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_term_creation() {
        let var = Term::variable("X");
        assert!(var.is_variable());
        assert!(!var.is_ground());

        let const_term = Term::constant("john");
        assert!(!const_term.is_variable());
        assert!(const_term.is_ground());

        let func = Term::function("father", vec![var.clone(), const_term.clone()]);
        assert!(!func.is_ground());
    }

    #[test]
    fn test_term_variables() {
        let x = Term::variable("X");
        let y = Term::variable("Y");
        let func = Term::function("f", vec![x.clone(), y.clone()]);

        let vars = func.variables();
        assert_eq!(vars.len(), 2);
        assert!(vars.contains("X"));
        assert!(vars.contains("Y"));
    }

    #[test]
    fn test_unify_variables() {
        let x = Term::variable("X");
        let john = Term::constant("john");

        let result = unify(&x, &john);
        assert!(result.is_success());

        let subst = result.substitution().unwrap();
        assert_eq!(subst.get("X"), Some(&john));
    }

    #[test]
    fn test_unify_functions() {
        let x = Term::variable("X");
        let y = Term::variable("Y");

        let f1 = Term::function("f", vec![x.clone(), Term::constant("a")]);
        let f2 = Term::function("f", vec![Term::constant("b"), y.clone()]);

        let result = unify(&f1, &f2);
        assert!(result.is_success());

        let subst = result.substitution().unwrap();
        assert_eq!(subst.get("X"), Some(&Term::constant("b")));
        assert_eq!(subst.get("Y"), Some(&Term::constant("a")));
    }

    #[test]
    fn test_occurs_check() {
        let x = Term::variable("X");
        let f = Term::function("f", vec![x.clone()]);

        let result = unify(&x, &f);
        assert!(!result.is_success());
    }

    #[test]
    fn test_formula_display() {
        let x = Term::variable("X");
        let john = Term::constant("john");

        let pred = Formula::predicate("mortal", vec![x.clone()]);
        assert_eq!(pred.to_string(), "mortal(X)");

        let implies = Formula::implies(
            Formula::predicate("human", vec![x.clone()]),
            Formula::predicate("mortal", vec![x.clone()]),
        );
        assert_eq!(implies.to_string(), "(human(X) → mortal(X))");
    }

    #[test]
    fn test_simple_query() {
        let mut solver = FOLSolver::new();

        // Fact: human(socrates)
        solver.add_fact(Formula::predicate("human", vec![Term::constant("socrates")]));

        // Query: human(socrates)?
        let query = Formula::predicate("human", vec![Term::constant("socrates")]);
        assert!(solver.prove(&query));

        // Query: human(plato)?
        let query2 = Formula::predicate("human", vec![Term::constant("plato")]);
        assert!(!solver.prove(&query2));
    }

    #[test]
    fn test_rule_inference() {
        let mut solver = FOLSolver::new();

        let x = Term::variable("X");

        // Fact: human(socrates)
        solver.add_fact(Formula::predicate("human", vec![Term::constant("socrates")]));

        // Rule: mortal(X) :- human(X)
        solver.add_rule(
            Formula::predicate("mortal", vec![x.clone()]),
            vec![Formula::predicate("human", vec![x.clone()])],
        );

        // Query: mortal(socrates)?
        let query = Formula::predicate("mortal", vec![Term::constant("socrates")]);
        assert!(solver.prove(&query));
    }

    #[test]
    fn test_variable_substitution() {
        let mut solver = FOLSolver::new();

        let x = Term::variable("X");

        // Fact: parent(tom, bob)
        solver.add_fact(Formula::predicate("parent", vec![
            Term::constant("tom"),
            Term::constant("bob"),
        ]));

        // Query: parent(tom, X)?
        let query = Formula::predicate("parent", vec![
            Term::constant("tom"),
            x.clone(),
        ]);

        let results = solver.query(&query);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].get("X"), Some(&Term::constant("bob")));
    }

    #[test]
    fn test_transitive_relation() {
        let mut solver = FOLSolver::new();

        let x = Term::variable("X");
        let y = Term::variable("Y");
        let z = Term::variable("Z");

        // Facts:
        solver.add_fact(Formula::predicate("parent", vec![
            Term::constant("tom"),
            Term::constant("bob"),
        ]));
        solver.add_fact(Formula::predicate("parent", vec![
            Term::constant("bob"),
            Term::constant("ann"),
        ]));

        // Rule: ancestor(X, Y) :- parent(X, Y)
        solver.add_rule(
            Formula::predicate("ancestor", vec![x.clone(), y.clone()]),
            vec![Formula::predicate("parent", vec![x.clone(), y.clone()])],
        );

        // Rule: ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z)
        solver.add_rule(
            Formula::predicate("ancestor", vec![x.clone(), z.clone()]),
            vec![
                Formula::predicate("parent", vec![x.clone(), y.clone()]),
                Formula::predicate("ancestor", vec![y.clone(), z.clone()]),
            ],
        );

        // Query: ancestor(tom, ann)?
        let query = Formula::predicate("ancestor", vec![
            Term::constant("tom"),
            Term::constant("ann"),
        ]);

        assert!(solver.prove(&query));
    }
}
