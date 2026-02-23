use std::sync::Arc;

use crate::interop::{InteropError, contract::IrNodeContract};

pub trait OpImportPlugin: Send + Sync {
    fn namespace(&self) -> &str;
    fn can_import(&self, op_type: &str) -> bool;
    /// Validates the given node contract.
    ///
    /// # Errors
    ///
    /// Returns `Err(InteropError)` if the node violates any plugin-specific
    /// structural constraint. The default implementation always returns `Ok`.
    fn validate_node(&self, _node: &IrNodeContract) -> Result<(), InteropError> {
        Ok(())
    }
}

#[derive(Default, Clone)]
pub struct PluginRegistry {
    plugins: Vec<Arc<dyn OpImportPlugin>>,
}

impl PluginRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self {
            plugins: Vec::new(),
        }
    }

    pub fn register(&mut self, plugin: Arc<dyn OpImportPlugin>) {
        self.plugins.push(plugin);
    }

    #[must_use]
    pub fn find(&self, op_type: &str) -> Option<Arc<dyn OpImportPlugin>> {
        self.plugins
            .iter()
            .find(|plugin| plugin.can_import(op_type))
            .map(Arc::clone)
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.plugins.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.plugins.is_empty()
    }
}
