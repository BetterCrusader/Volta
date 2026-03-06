use crate::ir::op::Op;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ValueId(pub usize);

impl std::fmt::Display for ValueId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "%{}", self.0)
    }
}

/// A typed attribute value for node metadata.
#[derive(Debug, Clone, PartialEq)]
pub enum AttributeValue {
    Int(i64),
    Float(f64),
    String(String),
    Bool(bool),
    Ints(Vec<i64>),
    Floats(Vec<f64>),
}

impl AttributeValue {
    pub fn as_int(&self) -> Option<i64> {
        if let Self::Int(v) = self {
            Some(*v)
        } else {
            None
        }
    }
    pub fn as_float(&self) -> Option<f64> {
        if let Self::Float(v) = self {
            Some(*v)
        } else {
            None
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        if let Self::String(v) = self {
            Some(v.as_str())
        } else {
            None
        }
    }
    pub fn as_bool(&self) -> Option<bool> {
        if let Self::Bool(v) = self {
            Some(*v)
        } else {
            None
        }
    }
}

impl From<i64> for AttributeValue {
    fn from(v: i64) -> Self {
        Self::Int(v)
    }
}
impl From<f64> for AttributeValue {
    fn from(v: f64) -> Self {
        Self::Float(v)
    }
}
impl From<f32> for AttributeValue {
    fn from(v: f32) -> Self {
        Self::Float(v as f64)
    }
}
impl From<bool> for AttributeValue {
    fn from(v: bool) -> Self {
        Self::Bool(v)
    }
}
impl From<String> for AttributeValue {
    fn from(v: String) -> Self {
        Self::String(v)
    }
}
impl From<&str> for AttributeValue {
    fn from(v: &str) -> Self {
        Self::String(v.to_string())
    }
}

#[derive(Debug, Clone)]
pub struct Node {
    pub id: NodeId,
    pub op: Op,
    pub output: ValueId,
    /// Optional metadata attributes for passes, backends, and diagnostics.
    pub attrs: HashMap<String, AttributeValue>,
}

impl Node {
    #[must_use]
    pub fn new(id: NodeId, op: Op, output: ValueId) -> Self {
        Self {
            id,
            op,
            output,
            attrs: HashMap::new(),
        }
    }

    /// Set a metadata attribute.
    pub fn set_attr(&mut self, key: &str, value: impl Into<AttributeValue>) {
        self.attrs.insert(key.to_string(), value.into());
    }

    /// Get a metadata attribute.
    pub fn get_attr(&self, key: &str) -> Option<&AttributeValue> {
        self.attrs.get(key)
    }
}
