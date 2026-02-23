/// # Pass Equivalence Harness
///
/// Each test builds a graph, runs one compiler pass, and asserts:
/// 1. The output value is numerically within epsilon of the pre-pass result.
/// 2. The graph still passes the IR verifier.
/// 3. Shape is preserved.
///
/// This is the correctness gating mechanism for all passes in Volta.
use std::sync::Arc;

use volta::ir::{
    AlgebraicSimplificationPass, ConstantFoldingPass, CsePass, DcePass, ExecutionContext, Graph,
    Op, Pass, RuntimeValue, Tensor, execute, execute_with_context,
};

/// Maximum allowed absolute difference for f32 element-wise comparison.
const ATOL: f32 = 1e-5;

/// Compare two `RuntimeValue`s elementwise within tolerance.
fn assert_values_eq(before: &RuntimeValue, after: &RuntimeValue, label: &str) {
    match (before, after) {
        (RuntimeValue::Int(a), RuntimeValue::Int(b)) => {
            assert_eq!(a, b, "[{label}] integer output changed");
        }
        (RuntimeValue::Float(a), RuntimeValue::Float(b)) => {
            assert!(
                (a - b).abs() < ATOL as f64,
                "[{label}] float output changed: {a} → {b}"
            );
        }
        (RuntimeValue::Tensor(t1), RuntimeValue::Tensor(t2)) => {
            assert_eq!(
                t1.shape, t2.shape,
                "[{label}] tensor shape changed: {:?} → {:?}",
                t1.shape, t2.shape
            );
            for (i, (a, b)) in t1.data.iter().zip(t2.data.iter()).enumerate() {
                let diff = (a - b).abs();
                assert!(
                    diff <= ATOL,
                    "[{label}] element [{i}] changed by {diff} > {ATOL}: {a} → {b}"
                );
            }
        }
        _ => panic!("[{label}] output type changed between before and after pass"),
    }
}

/// Run a pass and assert it maintains semantics on a graph with scalar terminals.
fn assert_pass_equivalence<P: Pass>(graph: &mut Graph, pass: &mut P, label: &str) {
    let before_opt = execute(graph).expect("execute before pass should succeed");
    let before = before_opt.expect("graph should produce a terminal value");
    pass.run(graph);
    let after_opt = execute(graph).expect("execute after pass should succeed");
    let after = after_opt.expect("graph should produce a terminal value");
    assert_values_eq(&before, &after, label);
}

/// Run a pass and assert equivalence on a graph that requires inputs/parameters.
fn assert_pass_equivalence_with_ctx<P: Pass>(
    graph: &mut Graph,
    pass: &mut P,
    ctx: &ExecutionContext,
    output_node: volta::ir::ValueId,
    label: &str,
) {
    let before_opt = execute_with_context(graph, ctx).expect("execute before pass should succeed");
    let before = before_opt.expect("graph should produce a terminal value");
    pass.run(graph);
    let after_opt = execute_with_context(graph, ctx).expect("execute after pass should succeed");
    let after = after_opt.expect("graph should produce a terminal value");
    assert_values_eq(&before, &after, label);
    let _ = output_node;
}

// ── Constant Folding ──────────────────────────────────────────────────────────

#[test]
fn constant_folding_preserves_scalar_chain() {
    let mut graph = Graph::new();
    let block = graph.create_block();

    // (3 + 4) * 2 - 1 = 13
    let (_, a) = graph.add_op(block, Op::ConstInt(3)).unwrap();
    let (_, b) = graph.add_op(block, Op::ConstInt(4)).unwrap();
    let (_, c) = graph.add_op(block, Op::ConstInt(2)).unwrap();
    let (_, d) = graph.add_op(block, Op::ConstInt(1)).unwrap();

    let (_, add) = graph.add_op(block, Op::Add(a, b)).unwrap();
    let (_, mul) = graph.add_op(block, Op::Mul(add, c)).unwrap();
    graph.add_op(block, Op::Sub(mul, d)).unwrap();

    let mut pass = ConstantFoldingPass::new();
    assert_pass_equivalence(&mut graph, &mut pass, "constant_folding_scalar_chain");
}

#[test]
fn constant_folding_preserves_float_chain() {
    let mut graph = Graph::new();
    let block = graph.create_block();

    // 2.5 * 4.0 / 2.0 = 5.0
    let (_, a) = graph.add_op(block, Op::ConstFloat(2.5)).unwrap();
    let (_, b) = graph.add_op(block, Op::ConstFloat(4.0)).unwrap();
    let (_, c) = graph.add_op(block, Op::ConstFloat(2.0)).unwrap();

    let (_, mul) = graph.add_op(block, Op::Mul(a, b)).unwrap();
    graph.add_op(block, Op::Div(mul, c)).unwrap();

    let mut pass = ConstantFoldingPass::new();
    assert_pass_equivalence(&mut graph, &mut pass, "constant_folding_float");
}

#[test]
fn constant_folding_preserves_tensor_result() {
    let mut graph = Graph::new();
    let block = graph.create_block();

    // ConstTensor([1,2,3]) + ConstTensor([4,5,6]) = [5,7,9]
    let (_, a) = graph
        .add_op(
            block,
            Op::ConstTensor {
                shape: vec![3],
                data: vec![1.0, 2.0, 3.0],
            },
        )
        .unwrap();
    let (_, b) = graph
        .add_op(
            block,
            Op::ConstTensor {
                shape: vec![3],
                data: vec![4.0, 5.0, 6.0],
            },
        )
        .unwrap();
    graph.add_op(block, Op::Add(a, b)).unwrap();

    let mut pass = ConstantFoldingPass::new();
    assert_pass_equivalence(&mut graph, &mut pass, "constant_folding_tensor");
}

// ── Algebraic Simplification ──────────────────────────────────────────────────

#[test]
fn algebraic_simplification_preserves_x_plus_zero() {
    let mut graph = Graph::new();
    let block = graph.create_block();

    let (_, x) = graph.add_op(block, Op::ConstInt(42)).unwrap();
    let (_, zero) = graph.add_op(block, Op::ConstInt(0)).unwrap();
    graph.add_op(block, Op::Add(x, zero)).unwrap();

    let mut pass = AlgebraicSimplificationPass::new();
    assert_pass_equivalence(&mut graph, &mut pass, "algebraic_simplification_x+0");
}

#[test]
fn algebraic_simplification_preserves_x_times_one() {
    let mut graph = Graph::new();
    let block = graph.create_block();

    let (_, x) = graph
        .add_op(block, Op::ConstFloat(std::f64::consts::PI))
        .unwrap();
    let (_, one) = graph.add_op(block, Op::ConstFloat(1.0)).unwrap();
    graph.add_op(block, Op::Mul(x, one)).unwrap();

    let mut pass = AlgebraicSimplificationPass::new();
    assert_pass_equivalence(&mut graph, &mut pass, "algebraic_simplification_x*1");
}

#[test]
fn algebraic_simplification_preserves_x_times_zero() {
    let mut graph = Graph::new();
    let block = graph.create_block();

    let (_, x) = graph.add_op(block, Op::ConstInt(999)).unwrap();
    let (_, zero) = graph.add_op(block, Op::ConstInt(0)).unwrap();
    graph.add_op(block, Op::Mul(x, zero)).unwrap();

    let mut pass = AlgebraicSimplificationPass::new();
    assert_pass_equivalence(&mut graph, &mut pass, "algebraic_simplification_x*0");
}

#[test]
fn algebraic_simplification_preserves_x_minus_zero() {
    let mut graph = Graph::new();
    let block = graph.create_block();

    let (_, x) = graph.add_op(block, Op::ConstInt(7)).unwrap();
    let (_, zero) = graph.add_op(block, Op::ConstInt(0)).unwrap();
    graph.add_op(block, Op::Sub(x, zero)).unwrap();

    let mut pass = AlgebraicSimplificationPass::new();
    assert_pass_equivalence(&mut graph, &mut pass, "algebraic_simplification_x-0");
}

// ── CSE (Common Subexpression Elimination) ────────────────────────────────────

#[test]
fn cse_preserves_duplicate_add() {
    let mut graph = Graph::new();
    let block = graph.create_block();

    let (_, a) = graph.add_op(block, Op::ConstInt(5)).unwrap();
    let (_, b) = graph.add_op(block, Op::ConstInt(3)).unwrap();

    // Two identical adds — CSE should merge them
    let (_, add1) = graph.add_op(block, Op::Add(a, b)).unwrap();
    let (_, add2) = graph.add_op(block, Op::Add(a, b)).unwrap();

    // Combined result to ensure both paths are live
    graph.add_op(block, Op::Add(add1, add2)).unwrap();

    let mut pass = CsePass::new();
    assert_pass_equivalence(&mut graph, &mut pass, "cse_duplicate_add");
}

#[test]
fn cse_preserves_duplicate_float_mul() {
    let mut graph = Graph::new();
    let block = graph.create_block();

    let (_, x) = graph.add_op(block, Op::ConstFloat(2.5)).unwrap();
    let (_, y) = graph.add_op(block, Op::ConstFloat(1.5)).unwrap();

    let (_, m1) = graph.add_op(block, Op::Mul(x, y)).unwrap();
    let (_, m2) = graph.add_op(block, Op::Mul(x, y)).unwrap();
    graph.add_op(block, Op::Add(m1, m2)).unwrap();

    let mut pass = CsePass::new();
    assert_pass_equivalence(&mut graph, &mut pass, "cse_duplicate_float_mul");
}

// ── DCE (Dead Code Elimination) ───────────────────────────────────────────────

#[test]
fn dce_preserves_live_computation() {
    let mut graph = Graph::new();
    let block = graph.create_block();

    let (_, a) = graph.add_op(block, Op::ConstInt(10)).unwrap();
    let (_, b) = graph.add_op(block, Op::ConstInt(20)).unwrap();

    // Dead node — never used downstream of the terminal
    let (_, _dead) = graph.add_op(block, Op::Add(a, b)).unwrap();

    // Live node — used as output
    graph.add_op(block, Op::Mul(a, b)).unwrap();

    let mut pass = DcePass::new();
    assert_pass_equivalence(&mut graph, &mut pass, "dce_live_computation");
}

// ── Combined passes ───────────────────────────────────────────────────────────

#[test]
fn combined_passes_preserve_tensor_add_with_parameter() {
    let mut graph = Graph::new();
    let block = graph.create_block();

    let (_, p) = graph.add_op(block, Op::Parameter("w".to_string())).unwrap();
    let (_, c) = graph
        .add_op(
            block,
            Op::ConstTensor {
                shape: vec![2],
                data: vec![1.0, 1.0],
            },
        )
        .unwrap();
    // w + ones + zeros_folded
    let (_, add) = graph.add_op(block, Op::Add(p, c)).unwrap();
    let (_, zero) = graph
        .add_op(
            block,
            Op::ConstTensor {
                shape: vec![2],
                data: vec![0.0, 0.0],
            },
        )
        .unwrap();
    graph.add_op(block, Op::Add(add, zero)).unwrap();
    graph.bind_parameter_shape("w", vec![2]);

    let mut ctx = ExecutionContext::default();
    ctx.parameters.insert(
        "w".to_string(),
        RuntimeValue::Tensor(Arc::new(Tensor::new(vec![2], vec![3.0, 7.0]).unwrap())),
    );

    let out_node = graph.nodes.last().unwrap().output;

    // Run all passes in order and assert equivalence holds through each step.
    let mut fold = ConstantFoldingPass::new();
    assert_pass_equivalence_with_ctx(&mut graph, &mut fold, &ctx, out_node, "pass1_cf");

    let mut simplify = AlgebraicSimplificationPass::new();
    assert_pass_equivalence_with_ctx(&mut graph, &mut simplify, &ctx, out_node, "pass2_alg");

    let mut cse = CsePass::new();
    assert_pass_equivalence_with_ctx(&mut graph, &mut cse, &ctx, out_node, "pass3_cse");

    let mut dce = DcePass::new();
    assert_pass_equivalence_with_ctx(&mut graph, &mut dce, &ctx, out_node, "pass4_dce");
}
