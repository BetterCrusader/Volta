use std::collections::HashSet;

use volta::ir::{Graph, Op, PlacementClass, build_execution_plan};

#[test]
fn execution_plan_contains_backend_neutral_placement_hints() {
    let mut graph = Graph::new();
    let block = graph.create_block();
    let (_, x) = graph
        .add_op(block, Op::Input("x".to_string()))
        .expect("add input should succeed");
    let (_, w) = graph
        .add_op(block, Op::Parameter("w".to_string()))
        .expect("add parameter should succeed");
    let (_, y) = graph
        .add_op(block, Op::MatMul(x, w))
        .expect("add matmul should succeed");
    let (_, out) = graph
        .add_op(block, Op::Output(y))
        .expect("add output should succeed");

    let plan = build_execution_plan(&graph, &HashSet::new()).expect("plan should build");
    assert!(!plan.placement_hints.is_empty());

    assert_eq!(class_for(&plan, x), PlacementClass::Input);
    assert_eq!(class_for(&plan, w), PlacementClass::Parameter);
    assert_eq!(class_for(&plan, y), PlacementClass::Temporary);
    assert_eq!(class_for(&plan, out), PlacementClass::Output);

    let ids = plan
        .placement_hints
        .iter()
        .map(|hint| hint.value.0)
        .collect::<Vec<_>>();
    let mut sorted = ids.clone();
    sorted.sort_unstable();
    assert_eq!(ids, sorted, "placement hints must be deterministic");
}

fn class_for(plan: &volta::ir::ExecutionPlan, value: volta::ir::ValueId) -> PlacementClass {
    plan.placement_hints
        .iter()
        .find(|hint| hint.value == value)
        .unwrap_or_else(|| panic!("missing placement hint for value {}", value.0))
        .class
}
