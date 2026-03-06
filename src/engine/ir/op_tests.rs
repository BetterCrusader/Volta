#[cfg(test)]
mod tests {
    use crate::ir::{Graph, Op, RuntimeValue, verify_graph, execute_value};
    use crate::model::{ModelBuilder, BatchNormLayer, TensorShape, Module};

    #[test]
    fn test_reduce_max_backward_cpu() {
        let mut graph = Graph::new();
        let block = graph.create_block();

        let input_data = vec![1.0, 2.0, 3.0, 3.0];
        let (_, input) = graph.add_op(block, Op::ConstTensor { shape: vec![4], data: input_data }).unwrap();

        let (_, omax) = graph.add_op(block, Op::ConstTensor { shape: vec![1], data: vec![3.0] }).unwrap();
        let (_, upstream) = graph.add_op(block, Op::ConstTensor { shape: vec![1], data: vec![1.0] }).unwrap();

        let (_, grad) = graph.add_op(block, Op::ReduceMaxBackward {
            input,
            output_max: omax,
            upstream,
            axis: None,
            keepdims: false,
        }).unwrap();

        graph.add_op(block, Op::Output(grad)).unwrap();
        verify_graph(&graph).unwrap();

        let result = execute_value(&graph, grad).unwrap();

        if let RuntimeValue::Tensor(t) = result {
            assert_eq!(t.shape, vec![4]);
            assert_eq!(t.data.as_ref(), &vec![0.0, 0.0, 0.5, 0.5]);
        } else {
            panic!("Expected tensor");
        }
    }

    #[test]
    fn test_batch_norm_forward_cpu() {
        let mut graph = Graph::new();
        let block = graph.create_block();

        let input_shape = vec![1, 1, 2, 2];
        let input_data = vec![1.0, 2.0, 3.0, 4.0];
        let (_, input) = graph.add_op(block, Op::ConstTensor { shape: input_shape, data: input_data }).unwrap();

        let (_, weight) = graph.add_op(block, Op::ConstTensor { shape: vec![1], data: vec![1.0] }).unwrap();
        let (_, bias) = graph.add_op(block, Op::ConstTensor { shape: vec![1], data: vec![0.0] }).unwrap();
        let (_, mean) = graph.add_op(block, Op::ConstTensor { shape: vec![1], data: vec![2.5] }).unwrap();
        let (_, var) = graph.add_op(block, Op::ConstTensor { shape: vec![1], data: vec![1.25] }).unwrap();

        let (_, out) = graph.add_op(block, Op::BatchNorm {
            input,
            weight,
            bias,
            mean,
            var,
        }).unwrap();

        graph.add_op(block, Op::Output(out)).unwrap();
        verify_graph(&graph).unwrap();

        let result = execute_value(&graph, out).unwrap();

        if let RuntimeValue::Tensor(t) = result {
            assert_eq!(t.shape, vec![1, 1, 2, 2]);
            let eps = 1e-5;
            let std = (1.25f32 + eps).sqrt();
            assert!((t.data[0] - (1.0 - 2.5) / std).abs() < 1e-6);
            assert!((t.data[3] - (4.0 - 2.5) / std).abs() < 1e-6);
        } else {
            panic!("Expected tensor");
        }
    }

    #[test]
    fn test_matmul_determinism() {
        let m = 128;
        let k = 256;
        let n = 128;

        let mut left_data = vec![0.0f32; m * k];
        let mut right_data = vec![0.0f32; k * n];
        for i in 0..left_data.len() { left_data[i] = (i as f32).sin(); }
        for i in 0..right_data.len() { right_data[i] = (i as f32).cos(); }

        let run_matmul = || {
            let mut graph = Graph::new();
            let block = graph.create_block();
            let (_, left) = graph.add_op(block, Op::ConstTensor { shape: vec![m, k], data: left_data.clone() }).unwrap();
            let (_, right) = graph.add_op(block, Op::ConstTensor { shape: vec![k, n], data: right_data.clone() }).unwrap();
            let (_, res) = graph.add_op(block, Op::MatMul(left, right)).unwrap();
            graph.add_op(block, Op::Output(res)).unwrap();
            verify_graph(&graph).unwrap();
            let result = execute_value(&graph, res).unwrap();
            if let RuntimeValue::Tensor(t) = result { t.data.as_ref().to_vec() } else { panic!() }
        };

        let first = run_matmul();
        for _ in 0..5 {
            let next = run_matmul();
            assert_eq!(first, next, "Matmul results must be identical for determinism");
        }
    }

    #[test]
    fn test_batch_norm_layer_build() {
        let mut builder = ModelBuilder::new();
        let input = builder.input_with_shape("in", vec![1, 16, 32, 32]).unwrap();
        let bn = BatchNormLayer::new("bn", 16);
        let (out, shape) = bn.build(&mut builder, input, &TensorShape(vec![1, 16, 32, 32])).unwrap();

        assert_eq!(shape.0, vec![1, 16, 32, 32]);
        let model = builder.finalize(out, shape, None).unwrap();
        assert!(model.parameters.contains_key("bn.weight"));
        assert!(model.parameters.contains_key("bn.bias"));
        assert!(model.parameters.contains_key("bn.mean"));
        assert!(model.parameters.contains_key("bn.var"));
    }
}
