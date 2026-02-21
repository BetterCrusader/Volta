pub mod algebraic_simplification;
pub mod allocation;
pub mod autograd;
pub mod backend;
pub mod backend_capabilities;
pub mod block;
pub mod compiler_flags;
pub mod constant_folding;
pub mod cse;
pub mod cuda;
pub mod dce;
pub mod dead_tensor_elimination;
pub mod elementwise_fusion;
pub mod execution_plan;
pub mod fingerprint;
pub mod freeze_hardening;
pub mod gradient_fusion;
pub mod graph;
pub mod interpreter;
pub mod kernel_grouping;
pub mod layout_optimization;
pub mod lowering;
pub mod memory_planner;
pub mod node;
pub mod op;
pub mod optimizer;
pub mod pass;
pub mod pass_utils;
pub mod plan_cache;
pub mod printer;
pub mod regression_harness;
pub mod runtime;
pub mod schedule_optimization;
pub mod scheduler;
pub mod shape_inference;
pub mod static_memory_budget;
pub mod tensor;
pub mod tensor_constant_propagation;
pub mod train;
pub mod verifier;

pub use algebraic_simplification::AlgebraicSimplificationPass;
pub use allocation::{
    AllocationError, AllocationPlan, BufferId, StorageClass, plan_allocation, verify_allocation,
};
pub use autograd::{AutogradError, GradientGraph, build_reverse_graph};
pub use backend::{Backend, BackendError, CompiledProgram, CpuBackend, CudaBackend, LlvmBackend};
pub use backend_capabilities::{BackendCapabilities, BackendKind, DeterminismLevel};
pub use block::{BasicBlock, BasicBlockId};
pub use compiler_flags::CompilerFlags;
pub use constant_folding::ConstantFoldingPass;
pub use cse::CsePass;
pub use dce::DcePass;
pub use dead_tensor_elimination::DeadTensorEliminationPass;
pub use elementwise_fusion::ElementwiseFusionPass;
pub use execution_plan::{
    ExecutionPlan, ExecutionPlanError, PlacementClass, PlacementHint, build_execution_plan,
};
pub use fingerprint::graph_fingerprint;
pub use gradient_fusion::GradientFusionPass;
pub use graph::{Graph, GraphError, ShapeSignature};
pub use interpreter::{
    ExecutionContext, InterpreterError, RuntimeValue, execute, execute_value,
    execute_value_with_context, execute_value_with_schedule_context, execute_with_context,
    execute_with_schedule_context,
};
pub use kernel_grouping::{KernelGroup, KernelGroupingError, KernelKind, group_kernels};
pub use layout_optimization::LayoutAwareOptimizationPass;
pub use lowering::{LoweringContext, lower_program};
pub use memory_planner::{
    MemoryPlan, MemoryPlanError, ValueLiveness, plan_memory, render_lifetime_heatmap,
};
pub use node::{Node, NodeId, ValueId};
pub use op::{ElementwiseUnaryOp, Op};
pub use optimizer::{OptimizerConfig, OptimizerError, OptimizerState, apply_gradients};
pub use pass::Pass;
pub use pass_utils::run_with_verifier_guard;
pub use plan_cache::{clear_plan_cache, compile_or_get_cached};
pub use printer::print_graph;
pub use runtime::{RuntimeGatewayError, execute_terminal_with_backend, execute_value_with_backend};
pub use schedule_optimization::{
    ScheduleOptimizationError, ScheduleOptimizationResult, optimize_schedule,
};
pub use scheduler::{Schedule, ScheduleError, build_schedule, schedule_hash, verify_schedule};
pub use shape_inference::{ShapeError, ShapeFact, infer_shapes};
pub use static_memory_budget::{
    StaticMemoryBudget, StaticMemoryBudgetError, StaticMemoryBudgetReport,
    evaluate_static_memory_budget,
};
pub use tensor::{Tensor, TensorError};
pub use tensor_constant_propagation::TensorConstantPropagationPass;
pub use train::{
    TrainConfig, TrainError, TrainResult, TrainSample, train_graph, train_graph_with_backend,
};
pub use verifier::{VerifyError, run_verified_pass, verify_graph, verify_with_policy};
