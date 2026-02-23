use std::collections::{BTreeSet, HashMap, VecDeque};
use std::hash::{Hash, Hasher};

use crate::ir::{Graph, NodeId, Op, ValueId};

#[derive(Debug, Clone)]
pub struct Schedule {
    pub ordered_nodes: Vec<NodeId>,
}

#[derive(Debug, Clone)]
pub struct ScheduleError {
    pub message: String,
}

#[must_use]
pub fn schedule_hash(schedule: &Schedule) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for node in &schedule.ordered_nodes {
        node.0.hash(&mut hasher);
    }
    hasher.finish()
}

pub fn build_schedule(graph: &Graph) -> Result<Schedule, ScheduleError> {
    let mut value_to_node = HashMap::<ValueId, usize>::new();
    for (index, node) in graph.nodes.iter().enumerate() {
        if matches!(node.op, Op::Removed) {
            continue;
        }
        value_to_node.insert(node.output, index);
    }

    let mut indegree = vec![0usize; graph.nodes.len()];
    let mut edges = vec![Vec::<usize>::new(); graph.nodes.len()];
    let mut active_nodes = BTreeSet::<usize>::new();

    for (index, node) in graph.nodes.iter().enumerate() {
        if matches!(node.op, Op::Removed) {
            continue;
        }
        active_nodes.insert(index);
        for input in node.op.input_values() {
            let Some(dep_index) = value_to_node.get(&input).copied() else {
                return Err(ScheduleError {
                    message: format!(
                        "Cannot schedule node {}: missing producer for ValueId {}",
                        node.id.0, input.0
                    ),
                });
            };
            edges[dep_index].push(index);
            indegree[index] += 1;
        }
    }

    let mut queue = VecDeque::new();
    for index in &active_nodes {
        if indegree[*index] == 0 {
            queue.push_back(*index);
        }
    }

    let mut ordered_nodes = Vec::new();
    while let Some(index) = queue.pop_front() {
        ordered_nodes.push(graph.nodes[index].id);
        for next in &edges[index] {
            indegree[*next] = indegree[*next].saturating_sub(1);
            if indegree[*next] == 0 {
                queue.push_back(*next);
            }
        }
    }

    if ordered_nodes.len() != active_nodes.len() {
        return Err(ScheduleError {
            message: "Schedule build failed: graph contains cycle or unresolved dependency"
                .to_string(),
        });
    }

    Ok(Schedule { ordered_nodes })
}

pub fn verify_schedule(graph: &Graph, schedule: &Schedule) -> Result<(), ScheduleError> {
    let mut value_to_node = HashMap::<ValueId, usize>::new();
    for (index, node) in graph.nodes.iter().enumerate() {
        if matches!(node.op, Op::Removed) {
            continue;
        }
        value_to_node.insert(node.output, index);
    }

    let mut scheduled = BTreeSet::<usize>::new();
    for node_id in &schedule.ordered_nodes {
        if node_id.0 >= graph.nodes.len() {
            return Err(ScheduleError {
                message: format!("Schedule references missing NodeId {}", node_id.0),
            });
        }

        let node = &graph.nodes[node_id.0];
        if matches!(node.op, Op::Removed) {
            return Err(ScheduleError {
                message: format!("Schedule includes removed NodeId {}", node_id.0),
            });
        }

        for input in node.op.input_values() {
            let Some(producer) = value_to_node.get(&input).copied() else {
                return Err(ScheduleError {
                    message: format!(
                        "Scheduled node {} depends on missing ValueId {}",
                        node_id.0, input.0
                    ),
                });
            };
            if !scheduled.contains(&producer) {
                return Err(ScheduleError {
                    message: format!(
                        "Schedule dependency violation: node {} uses ValueId {} before producer {}",
                        node_id.0, input.0, producer
                    ),
                });
            }
        }

        if !scheduled.insert(node_id.0) {
            return Err(ScheduleError {
                message: format!("Schedule contains duplicate NodeId {}", node_id.0),
            });
        }
    }

    let active_count = graph
        .nodes
        .iter()
        .filter(|node| !matches!(node.op, Op::Removed))
        .count();
    if scheduled.len() != active_count {
        return Err(ScheduleError {
            message: format!(
                "Schedule is incomplete: covers {} active nodes out of {}",
                scheduled.len(),
                active_count
            ),
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::ir::{Graph, Op, build_schedule, schedule_hash, verify_schedule};

    #[test]
    fn builds_valid_schedule_for_simple_graph() {
        let mut graph = Graph::new();
        let block = graph.create_block();
        let (_, a) = graph
            .add_op(block, Op::ConstInt(1))
            .expect("add op should succeed");
        let (_, b) = graph
            .add_op(block, Op::ConstInt(2))
            .expect("add op should succeed");
        graph
            .add_op(block, Op::Add(a, b))
            .expect("add op should succeed");

        let schedule = build_schedule(&graph).expect("schedule should build");
        verify_schedule(&graph, &schedule).expect("schedule should verify");
        assert_eq!(schedule.ordered_nodes.len(), 3);
    }

    #[test]
    fn schedule_hash_is_deterministic() {
        let mut graph = Graph::new();
        let block = graph.create_block();
        let (_, a) = graph
            .add_op(block, Op::ConstInt(1))
            .expect("add op should succeed");
        let (_, b) = graph
            .add_op(block, Op::ConstInt(2))
            .expect("add op should succeed");
        graph
            .add_op(block, Op::Add(a, b))
            .expect("add op should succeed");

        let s1 = build_schedule(&graph).expect("schedule should build");
        let s2 = build_schedule(&graph).expect("schedule should build");
        assert_eq!(schedule_hash(&s1), schedule_hash(&s2));
    }
}
