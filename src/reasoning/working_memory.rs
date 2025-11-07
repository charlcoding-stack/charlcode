// Working Memory Architecture
//
// Inspired by cognitive psychology (Baddeley's Working Memory Model):
// - Short-term memory: Limited capacity (Miller's 7±2 items)
// - Long-term memory: Unlimited capacity, slower access
// - Memory consolidation: Transfer from short to long-term
//
// Types of long-term memory:
// - Episodic: Specific events ("I had pizza yesterday")
// - Semantic: General knowledge ("Pizza is food")
// - Procedural: Skills ("How to ride a bike")
//
// References:
// - Baddeley & Hitch (1974): "Working Memory"
// - Miller (1956): "The Magical Number Seven, Plus or Minus Two"

use std::collections::{HashMap, VecDeque};

/// Memory item with content and metadata
#[derive(Debug, Clone)]
pub struct MemoryItem {
    pub id: String,
    pub content: String,
    pub memory_type: MemoryType,
    pub timestamp: u64,
    pub access_count: usize,
    pub strength: f32, // [0, 1] - stronger memories less likely to decay
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MemoryType {
    /// Specific events
    Episodic,
    /// General knowledge
    Semantic,
    /// Skills and procedures
    Procedural,
}

impl MemoryItem {
    pub fn new(id: impl Into<String>, content: impl Into<String>, memory_type: MemoryType) -> Self {
        Self {
            id: id.into(),
            content: content.into(),
            memory_type,
            timestamp: 0,
            access_count: 0,
            strength: 1.0,
        }
    }

    pub fn access(&mut self) {
        self.access_count += 1;
        // Strengthen with each access (up to max 1.0)
        self.strength = (self.strength + 0.1).min(1.0);
    }

    pub fn decay(&mut self, decay_rate: f32) {
        self.strength = (self.strength - decay_rate).max(0.0);
    }
}

/// Short-term/Working memory with limited capacity
#[derive(Debug, Clone)]
pub struct ShortTermMemory {
    /// Buffer with capacity limit
    pub buffer: VecDeque<MemoryItem>,
    /// Maximum capacity (Miller's 7±2)
    pub capacity: usize,
    /// Decay rate per timestep
    pub decay_rate: f32,
}

impl ShortTermMemory {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
            decay_rate: 0.05,
        }
    }

    pub fn with_decay_rate(mut self, decay_rate: f32) -> Self {
        self.decay_rate = decay_rate;
        self
    }

    /// Add item to buffer
    pub fn add(&mut self, mut item: MemoryItem) -> Option<MemoryItem> {
        item.access();

        // If at capacity, remove oldest
        let evicted = if self.buffer.len() >= self.capacity {
            self.buffer.pop_front()
        } else {
            None
        };

        self.buffer.push_back(item);
        evicted
    }

    /// Retrieve item by ID
    pub fn get(&mut self, id: &str) -> Option<&mut MemoryItem> {
        self.buffer.iter_mut().find(|item| item.id == id)
    }

    /// Apply decay to all items
    pub fn tick(&mut self) {
        for item in &mut self.buffer {
            item.decay(self.decay_rate);
        }

        // Remove items with zero strength
        self.buffer.retain(|item| item.strength > 0.0);
    }

    /// Get current size
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Get items that should be consolidated (high strength, multiple accesses)
    pub fn get_consolidation_candidates(&self) -> Vec<MemoryItem> {
        self.buffer
            .iter()
            .filter(|item| item.access_count >= 2 && item.strength > 0.7)
            .cloned()
            .collect()
    }
}
#[derive(Debug, Clone)]

/// Long-term memory with unlimited capacity
pub struct LongTermMemory {
    /// Semantic memory (general knowledge)
    pub semantic: HashMap<String, MemoryItem>,
    /// Episodic memory (specific events)
    pub episodic: Vec<MemoryItem>,
    /// Procedural memory (skills)
    pub procedural: HashMap<String, MemoryItem>,
}

impl LongTermMemory {
    pub fn new() -> Self {
        Self {
            semantic: HashMap::new(),
            episodic: Vec::new(),
            procedural: HashMap::new(),
        }
    }

    /// Store memory based on type
    pub fn store(&mut self, item: MemoryItem) {
        match item.memory_type {
            MemoryType::Semantic => {
                self.semantic.insert(item.id.clone(), item);
            }
            MemoryType::Episodic => {
                self.episodic.push(item);
            }
            MemoryType::Procedural => {
                self.procedural.insert(item.id.clone(), item);
            }
        }
    }

    /// Retrieve semantic memory
    pub fn get_semantic(&mut self, id: &str) -> Option<&mut MemoryItem> {
        if let Some(item) = self.semantic.get_mut(id) {
            item.access();
            Some(item)
        } else {
            None
        }
    }

    /// Search episodic memories (most recent first)
    pub fn search_episodic(&self, query: &str) -> Vec<&MemoryItem> {
        self.episodic
            .iter()
            .rev() // Most recent first
            .filter(|item| item.content.contains(query))
            .collect()
    }

    /// Get procedural memory
    pub fn get_procedural(&mut self, id: &str) -> Option<&mut MemoryItem> {
        if let Some(item) = self.procedural.get_mut(id) {
            item.access();
            Some(item)
        } else {
            None
        }
    }

    /// Total memories stored
    pub fn total_size(&self) -> usize {
        self.semantic.len() + self.episodic.len() + self.procedural.len()
    }
}

impl Default for LongTermMemory {
    fn default() -> Self {
        Self::new()
    }
}
#[derive(Debug, Clone)]

/// Complete working memory system
pub struct WorkingMemorySystem {
    pub short_term: ShortTermMemory,
    pub long_term: LongTermMemory,
    pub current_timestep: u64,
}

impl WorkingMemorySystem {
    pub fn new(short_term_capacity: usize) -> Self {
        Self {
            short_term: ShortTermMemory::new(short_term_capacity),
            long_term: LongTermMemory::new(),
            current_timestep: 0,
        }
    }

    /// Add to short-term memory
    pub fn remember(&mut self, item: MemoryItem) {
        // Add to short-term
        if let Some(evicted) = self.short_term.add(item) {
            // If item was evicted and strong enough, consolidate to long-term
            if evicted.strength > 0.5 {
                self.long_term.store(evicted);
            }
        }
    }

    /// Consolidate short-term to long-term
    pub fn consolidate(&mut self) {
        let candidates = self.short_term.get_consolidation_candidates();
        for item in candidates {
            self.long_term.store(item);
        }
    }

    /// Advance time and apply decay
    pub fn tick(&mut self) {
        self.current_timestep += 1;
        self.short_term.tick();
    }

    /// Retrieve from any memory system
    pub fn recall(&mut self, id: &str) -> Option<String> {
        // Try short-term first (faster)
        if let Some(item) = self.short_term.get(id) {
            return Some(item.content.clone());
        }

        // Try long-term semantic
        if let Some(item) = self.long_term.get_semantic(id) {
            return Some(item.content.clone());
        }

        // Try long-term procedural
        if let Some(item) = self.long_term.get_procedural(id) {
            return Some(item.content.clone());
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_item_creation() {
        let item = MemoryItem::new("fact1", "Sky is blue", MemoryType::Semantic);
        assert_eq!(item.id, "fact1");
        assert_eq!(item.content, "Sky is blue");
        assert_eq!(item.memory_type, MemoryType::Semantic);
        assert_eq!(item.access_count, 0);
        assert_eq!(item.strength, 1.0);
    }

    #[test]
    fn test_memory_item_access() {
        let mut item = MemoryItem::new("fact1", "Test", MemoryType::Semantic);
        item.access();
        assert_eq!(item.access_count, 1);
        assert!(item.strength > 1.0 || (item.strength - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_memory_item_decay() {
        let mut item = MemoryItem::new("fact1", "Test", MemoryType::Semantic);
        item.decay(0.1);
        assert!((item.strength - 0.9).abs() < 1e-5);
    }

    #[test]
    fn test_short_term_memory_capacity() {
        let mut stm = ShortTermMemory::new(3);

        stm.add(MemoryItem::new("1", "Item 1", MemoryType::Episodic));
        stm.add(MemoryItem::new("2", "Item 2", MemoryType::Episodic));
        stm.add(MemoryItem::new("3", "Item 3", MemoryType::Episodic));

        assert_eq!(stm.len(), 3);

        // Adding 4th item should evict first
        let evicted = stm.add(MemoryItem::new("4", "Item 4", MemoryType::Episodic));
        assert!(evicted.is_some());
        assert_eq!(evicted.unwrap().id, "1");
        assert_eq!(stm.len(), 3);
    }

    #[test]
    fn test_short_term_memory_decay() {
        let mut stm = ShortTermMemory::new(5).with_decay_rate(0.2);

        stm.add(MemoryItem::new("1", "Item 1", MemoryType::Episodic));
        assert_eq!(stm.len(), 1);

        // After 6 ticks with 0.2 decay rate, item should be removed
        // (item.access() adds 0.1, so starts at 1.1, needs 6 ticks: 1.1 - 6*0.2 = -0.1 < 0)
        for _ in 0..6 {
            stm.tick();
        }
        assert_eq!(stm.len(), 0);
    }

    #[test]
    fn test_short_term_memory_get() {
        let mut stm = ShortTermMemory::new(3);
        stm.add(MemoryItem::new(
            "test",
            "Test content",
            MemoryType::Semantic,
        ));

        let item = stm.get("test");
        assert!(item.is_some());
        assert_eq!(item.unwrap().content, "Test content");
    }

    #[test]
    fn test_consolidation_candidates() {
        let mut stm = ShortTermMemory::new(5);

        let mut item1 = MemoryItem::new("1", "Item 1", MemoryType::Semantic);
        item1.access();
        item1.access(); // 2 accesses
        item1.strength = 0.8; // Strong

        let mut item2 = MemoryItem::new("2", "Item 2", MemoryType::Semantic);
        item2.access(); // Only 1 access

        stm.buffer.push_back(item1);
        stm.buffer.push_back(item2);

        let candidates = stm.get_consolidation_candidates();
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].id, "1");
    }

    #[test]
    fn test_long_term_memory_semantic() {
        let mut ltm = LongTermMemory::new();

        let item = MemoryItem::new("fact1", "Water is H2O", MemoryType::Semantic);
        ltm.store(item);

        assert_eq!(ltm.semantic.len(), 1);

        let retrieved = ltm.get_semantic("fact1");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().content, "Water is H2O");
    }

    #[test]
    fn test_long_term_memory_episodic() {
        let mut ltm = LongTermMemory::new();

        ltm.store(MemoryItem::new(
            "event1",
            "Had pizza for lunch",
            MemoryType::Episodic,
        ));
        ltm.store(MemoryItem::new(
            "event2",
            "Went to the park",
            MemoryType::Episodic,
        ));

        assert_eq!(ltm.episodic.len(), 2);

        let results = ltm.search_episodic("pizza");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "event1");
    }

    #[test]
    fn test_long_term_memory_procedural() {
        let mut ltm = LongTermMemory::new();

        let skill = MemoryItem::new("skill1", "Riding a bike", MemoryType::Procedural);
        ltm.store(skill);

        let retrieved = ltm.get_procedural("skill1");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().content, "Riding a bike");
    }

    #[test]
    fn test_working_memory_system() {
        let mut wm = WorkingMemorySystem::new(3);

        wm.remember(MemoryItem::new("item1", "Content 1", MemoryType::Semantic));
        wm.remember(MemoryItem::new("item2", "Content 2", MemoryType::Semantic));

        let recalled = wm.recall("item1");
        assert!(recalled.is_some());
        assert_eq!(recalled.unwrap(), "Content 1");
    }

    #[test]
    fn test_working_memory_consolidation() {
        let mut wm = WorkingMemorySystem::new(2);

        let mut strong_item = MemoryItem::new("strong", "Strong memory", MemoryType::Semantic);
        strong_item.access();
        strong_item.access();
        strong_item.strength = 0.9;

        wm.remember(strong_item.clone());
        wm.consolidate();

        // Should be in long-term now
        assert_eq!(wm.long_term.semantic.len(), 1);
    }

    #[test]
    fn test_working_memory_tick() {
        let mut wm = WorkingMemorySystem::new(3);

        wm.remember(MemoryItem::new("item1", "Test", MemoryType::Episodic));

        let initial_time = wm.current_timestep;
        wm.tick();

        assert_eq!(wm.current_timestep, initial_time + 1);
    }
}
