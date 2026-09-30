//! Seeded random-graph properties for optimization, the artifact codec, and
//! execution.
//!
//! Every seed builds a graph through the public builder from a small
//! elementwise, transpose, copy, constant, and effect vocabulary with exact
//! duplicates and per-operation schema versions, then checks that optimization
//! preserves observable results bit for bit and reaches its fixed point in one
//! run, and that the canonical artifact round-trips. A failure names its seed.

#![expect(
    clippy::unwrap_used,
    reason = "generated graphs are valid by construction, so a failure should fail the test"
)]

use std::collections::BTreeMap;
use std::sync::Arc;

use advect_runtime::{
    AttrMap, AttrValue, ConstantKind, DTypeDescriptor, GraphBuilder, GraphStore, Host,
    LinkedExecutionPlan, LinkedOperation, NodeId, NodeMetadata, NumericDType, Operand,
    OutputOwnership, PortableConstant, ValueSpec, optimize,
};

const SEEDS: u64 = 512;

/// `SplitMix64`: small, deterministic, and good enough to shape graphs.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut mixed = self.0;
        mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        mixed ^ (mixed >> 31)
    }

    fn below(&mut self, bound: usize) -> usize {
        usize::try_from(self.next() % u64::try_from(bound).unwrap()).unwrap()
    }

    fn percent(&mut self, chance: usize) -> bool {
        self.below(100) < chance
    }

    fn pick<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        items.get(self.below(items.len())).unwrap()
    }

    /// A value in [-2, 2] on a 1/16 grid.
    fn value(&mut self) -> f64 {
        f64::from(u32::try_from(self.below(65)).unwrap()) / 16.0 - 2.0
    }

    fn values(&mut self, count: usize) -> Vec<f64> {
        (0..count).map(|_| self.value()).collect()
    }
}

fn metadata(shape: Vec<usize>, attrs: AttrMap) -> NodeMetadata {
    NodeMetadata::new(
        attrs,
        shape,
        DTypeDescriptor::from_name("float64").unwrap(),
        None,
        1,
        None,
        None,
        None,
    )
    .unwrap()
}

/// One appended operation, kept so it can be repeated exactly.
#[derive(Clone)]
struct Appended {
    op: &'static str,
    parents: Vec<NodeId>,
    metadata: NodeMetadata,
}

struct Generator {
    rng: Rng,
    builder: GraphBuilder,
    /// Effects pin their ancestors, so only some graphs contain them.
    effects: bool,
    shapes: Vec<Vec<usize>>,
    unused: Vec<NodeId>,
    operations: Vec<Appended>,
    /// Each operation name's schema version, drawn at its first use.
    schema_versions: BTreeMap<&'static str, u32>,
}

impl Generator {
    fn push(&mut self, node_id: NodeId, shape: Vec<usize>) {
        assert_eq!(usize::try_from(node_id).unwrap(), self.shapes.len());
        self.shapes.push(shape);
        self.unused.push(node_id);
    }

    fn append(&mut self, op: &'static str, parents: Vec<NodeId>, metadata: NodeMetadata) {
        let rng = &mut self.rng;
        let schema_version = *self
            .schema_versions
            .entry(op)
            .or_insert_with(|| 1 + u32::try_from(rng.below(3)).unwrap());
        let node_id = self
            .builder
            .append_operation(op, schema_version, &parents, metadata.clone())
            .unwrap();
        self.push(node_id, metadata.shape().to_vec());
        self.unused.retain(|node_id| !parents.contains(node_id));
        self.operations.push(Appended {
            op,
            parents,
            metadata,
        });
    }

    /// A random node, usually one without users so that most nodes stay live.
    fn node(&mut self) -> NodeId {
        if !self.unused.is_empty() && self.rng.percent(80) {
            return *self.rng.pick(&self.unused);
        }
        NodeId::try_from(self.rng.below(self.shapes.len())).unwrap()
    }

    fn shape(&self, node: NodeId) -> Vec<usize> {
        self.shapes
            .get(usize::try_from(node).unwrap())
            .unwrap()
            .clone()
    }

    fn append_constant(&mut self) {
        let node = self.node();
        let shape = self.shape(node);
        let values = self.rng.values(shape.iter().product());
        let bytes = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let constant = PortableConstant::new(
            ConstantKind::Array,
            NumericDType::Float64,
            shape.clone(),
            bytes,
        )
        .unwrap();
        let node_id = self
            .builder
            .append_constant(metadata(shape.clone(), AttrMap::new()), constant)
            .unwrap();
        self.push(node_id, shape);
    }

    fn append_transpose(&mut self, parent: NodeId, axes: &[usize]) {
        let input = self.shape(parent);
        let shape = axes.iter().map(|&axis| *input.get(axis).unwrap()).collect();
        let rank = axes.len();
        let reversed = axes.iter().copied().eq((0..rank).rev());
        let mut attrs = AttrMap::new();
        if !reversed || self.rng.percent(50) {
            let rank = i64::try_from(rank).unwrap();
            let spelled = axes
                .iter()
                .map(|&axis| {
                    let positive = i64::try_from(axis).unwrap();
                    AttrValue::Integer(if self.rng.percent(50) {
                        positive - rank
                    } else {
                        positive
                    })
                })
                .collect();
            let axes = if self.rng.percent(50) {
                AttrValue::List(spelled)
            } else {
                AttrValue::Tuple(spelled)
            };
            attrs.insert("axes".to_owned(), axes);
        } else if self.rng.percent(50) {
            attrs.insert("axes".to_owned(), AttrValue::Null);
        }
        self.append("array.transpose", vec![parent], metadata(shape, attrs));
    }

    fn append_transposes(&mut self, parent: NodeId) {
        let rank = self.shape(parent).len();
        let mut axes = (0..rank).collect::<Vec<_>>();
        if self.rng.percent(30) {
            axes.reverse();
        } else {
            for index in (1..rank).rev() {
                axes.swap(index, self.rng.below(index + 1));
            }
        }
        self.append_transpose(parent, &axes);
        if self.rng.percent(60) {
            let mut inverse = vec![0; rank];
            for (position, &axis) in axes.iter().enumerate() {
                *inverse.get_mut(axis).unwrap() = position;
            }
            let transposed = NodeId::try_from(self.shapes.len() - 1).unwrap();
            self.append_transpose(transposed, &inverse);
        }
    }

    fn append_random_operation(&mut self) {
        const ELEMENTWISE: [&str; 6] = [
            "array.add",
            "array.subtract",
            "array.multiply",
            "array.negative",
            "array.sin",
            "advect.copy",
        ];
        let roll = self.rng.below(100);
        if roll < 12 && !self.operations.is_empty() {
            let repeated = self.rng.pick(&self.operations).clone();
            return self.append(repeated.op, repeated.parents, repeated.metadata);
        }
        if roll < 22 {
            return self.append_constant();
        }
        let parent = self.node();
        let op = match roll {
            ..25 if self.effects => "custom.effect",
            ..45 => return self.append_transposes(parent),
            _ => *self.rng.pick(&ELEMENTWISE),
        };
        let shape = self.shape(parent);
        let parents = if matches!(op, "array.add" | "array.subtract" | "array.multiply") {
            let peers = (0..self.shapes.len())
                .filter(|&index| self.shapes.get(index) == Some(&shape))
                .collect::<Vec<_>>();
            let peer = NodeId::try_from(*self.rng.pick(&peers)).unwrap();
            if self.rng.percent(50) {
                vec![parent, peer]
            } else {
                vec![peer, parent]
            }
        } else {
            vec![parent]
        };
        self.append(op, parents, metadata(shape, AttrMap::new()));
    }
}

/// An unoptimized random graph and one input value per graph input.
fn random_graph(seed: u64) -> (GraphStore, Vec<Array>) {
    let mut rng = Rng(seed);
    let base = (0..=rng.below(3))
        .map(|_| 1 + rng.below(3))
        .collect::<Vec<_>>();
    let effects = rng.percent(50);
    let mut generator = Generator {
        rng,
        builder: GraphBuilder::new(),
        effects,
        shapes: Vec::new(),
        unused: Vec::new(),
        operations: Vec::new(),
        schema_versions: BTreeMap::new(),
    };
    let mut inputs = Vec::new();
    for _ in 0..=generator.rng.below(3) {
        let node_id = generator
            .builder
            .append_input(metadata(base.clone(), AttrMap::new()))
            .unwrap();
        generator.push(node_id, base.clone());
        inputs.push(Array {
            shape: base.clone(),
            data: generator.rng.values(base.iter().product()),
        });
    }
    for _ in 0..20 + generator.rng.below(101) {
        generator.append_random_operation();
    }
    // About half of the sinks are observed, and one output may repeat.
    for sink in generator.unused.clone() {
        if generator.rng.percent(50) {
            generator.builder.append_output(sink).unwrap();
        }
    }
    let output = generator.node();
    generator.builder.append_output(output).unwrap();
    (generator.builder.finish_unoptimized().unwrap(), inputs)
}

#[derive(Clone, Debug)]
struct Array {
    shape: Vec<usize>,
    data: Vec<f64>,
}

impl Array {
    fn bits(&self) -> (&[usize], Vec<u64>) {
        (
            &self.shape,
            self.data.iter().map(|value| value.to_bits()).collect(),
        )
    }
}

#[derive(Debug)]
enum Kernel {
    Unary(fn(f64) -> f64),
    Binary(fn(f64, f64) -> f64),
    Transpose(Vec<usize>),
}

/// Reference host over dense row-major float64 arrays.
struct ArrayHost;

fn transpose_axes(attrs: &AttrMap, rank: usize) -> Result<Vec<usize>, String> {
    let values = match attrs.get("axes") {
        None | Some(AttrValue::Null) => return Ok((0..rank).rev().collect()),
        Some(AttrValue::List(values) | AttrValue::Tuple(values)) => values,
        Some(other) => return Err(format!("invalid transpose axes {other:?}")),
    };
    let rank = i64::try_from(rank).map_err(|error| error.to_string())?;
    values
        .iter()
        .map(|value| match value {
            AttrValue::Integer(axis) => {
                usize::try_from(axis.rem_euclid(rank)).map_err(|error| error.to_string())
            }
            other => Err(format!("invalid transpose axis {other:?}")),
        })
        .collect()
}

fn transpose(input: &Array, axes: &[usize]) -> Array {
    let mut strides = vec![1; input.shape.len()];
    for index in (1..input.shape.len()).rev() {
        let stride = strides.get(index).unwrap() * input.shape.get(index).unwrap();
        *strides.get_mut(index - 1).unwrap() = stride;
    }
    let shape = axes
        .iter()
        .map(|&axis| *input.shape.get(axis).unwrap())
        .collect::<Vec<_>>();
    let mut position = vec![0; shape.len()];
    let mut data = Vec::with_capacity(input.data.len());
    for _ in 0..input.data.len() {
        let offset = position
            .iter()
            .zip(axes)
            .map(|(&index, &axis)| index * strides.get(axis).unwrap())
            .sum::<usize>();
        data.push(*input.data.get(offset).unwrap());
        for (index, &extent) in position.iter_mut().zip(&shape).rev() {
            *index += 1;
            if *index < extent {
                break;
            }
            *index = 0;
        }
    }
    Array { shape, data }
}

impl Host for ArrayHost {
    type Value = Array;
    type LinkedOp = Kernel;
    type Error = String;

    fn link(
        &mut self,
        op: &str,
        _schema_version: u32,
        attrs: &AttrMap,
        outputs: &[ValueSpec],
    ) -> Result<LinkedOperation<Self::LinkedOp>, Self::Error> {
        let kernel = match op {
            "array.add" => Kernel::Binary(|left, right| left + right),
            "array.subtract" => Kernel::Binary(|left, right| left - right),
            "array.multiply" => Kernel::Binary(|left, right| left * right),
            "array.negative" => Kernel::Unary(|value| -value),
            "array.sin" => Kernel::Unary(f64::sin),
            "advect.copy" => Kernel::Unary(|value| value),
            "custom.effect" => Kernel::Unary(|value| 0.5_f64.mul_add(value, 1.0)),
            "array.transpose" => {
                let rank = outputs
                    .first()
                    .ok_or("transpose has no output")?
                    .shape()
                    .len();
                Kernel::Transpose(transpose_axes(attrs, rank)?)
            }
            _ => return Err(format!("unsupported operation {op}")),
        };
        Ok(LinkedOperation::new(
            kernel,
            Vec::new(),
            OutputOwnership::Owned,
        ))
    }

    fn materialize_constant(
        &mut self,
        _node_id: NodeId,
        constant: &PortableConstant,
    ) -> Result<Self::Value, Self::Error> {
        Ok(Array {
            shape: constant.shape().to_vec(),
            data: constant
                .data()
                .as_chunks::<8>()
                .0
                .iter()
                .map(|&bytes| f64::from_le_bytes(bytes))
                .collect(),
        })
    }

    fn retain_value(&mut self, value: &Self::Value) -> Result<Self::Value, Self::Error> {
        Ok(value.clone())
    }

    fn evaluate(
        &mut self,
        operation: &Self::LinkedOp,
        operands: Vec<Operand<'_, Self::Value>>,
    ) -> Result<Self::Value, Self::Error> {
        Ok(match (operation, operands.as_slice()) {
            (Kernel::Unary(apply), [value]) => Array {
                shape: value.value().shape.clone(),
                data: value.value().data.iter().map(|&item| apply(item)).collect(),
            },
            (Kernel::Binary(apply), [left, right]) => {
                let (left, right) = (left.value(), right.value());
                if left.shape != right.shape {
                    return Err("binary operands have different shapes".to_owned());
                }
                Array {
                    shape: left.shape.clone(),
                    data: left
                        .data
                        .iter()
                        .zip(&right.data)
                        .map(|(&left, &right)| apply(left, right))
                        .collect(),
                }
            }
            (Kernel::Transpose(axes), [value]) => transpose(value.value(), axes),
            _ => return Err(format!("{operation:?} received the wrong operand count")),
        })
    }

    fn validate_value(
        &mut self,
        value: &Self::Value,
        outputs: &[ValueSpec],
    ) -> Result<(), Self::Error> {
        match outputs {
            [spec]
                if spec.shape() == value.shape
                    && spec.dtype().canonical() == "float64"
                    && value.shape.iter().product::<usize>() == value.data.len() =>
            {
                Ok(())
            }
            _ => Err(format!("{value:?} does not match {outputs:?}")),
        }
    }
}

fn execute(store: Arc<GraphStore>, inputs: Vec<Array>) -> Vec<Array> {
    let mut host = ArrayHost;
    LinkedExecutionPlan::from_store(store, &mut host)
        .unwrap()
        .execute(&mut host, inputs)
        .unwrap()
}

/// Only `custom.effect` is impure here; the optimizer must keep it.
fn is_impure(op: &str) -> bool {
    op.starts_with("custom.")
}

/// The canonical artifact round-trips byte for byte and node for node.
fn assert_round_trip(seed: u64, store: &GraphStore) -> GraphStore {
    let encoded = store.to_json().unwrap();
    let restored = GraphStore::from_json(&encoded).unwrap();
    assert_eq!(restored.to_json().unwrap(), encoded, "seed {seed}");
    for node_id in store.topological_order() {
        assert_eq!(
            restored.get_node(node_id).unwrap(),
            store.get_node(node_id).unwrap(),
            "seed {seed}"
        );
    }
    restored
}

/// Remapped nodes keep their values' metadata, endpoints remap exactly, and
/// every impure node survives as itself.
fn assert_remap(
    seed: u64,
    source: &GraphStore,
    optimized: &GraphStore,
    old_to_new: &[Option<NodeId>],
) {
    assert_eq!(old_to_new.len(), source.node_count(), "seed {seed}");
    let mapped = |node_id: NodeId| *old_to_new.get(usize::try_from(node_id).unwrap()).unwrap();
    for node in source.nodes() {
        let node = node.unwrap();
        let impure = is_impure(node.schema.name());
        let Some(new_id) = mapped(node.id) else {
            assert!(!impure, "seed {seed}: dropped %{}", node.id);
            continue;
        };
        let new = optimized.node(new_id).unwrap();
        assert_eq!(
            new.metadata.outputs(),
            node.metadata.outputs(),
            "seed {seed}: %{} -> %{new_id}",
            node.id
        );
        if impure {
            assert_eq!(new.schema, node.schema, "seed {seed}: %{}", node.id);
        }
    }
    let remap = |ids: &[NodeId]| ids.iter().map(|&id| mapped(id)).collect::<Option<Vec<_>>>();
    assert_eq!(
        remap(source.inputs()).as_deref(),
        Some(optimized.inputs()),
        "seed {seed}"
    );
    assert_eq!(
        remap(source.outputs()).as_deref(),
        Some(optimized.outputs()),
        "seed {seed}"
    );
    let impure_count = |store: &GraphStore| {
        store
            .nodes()
            .filter(|node| is_impure(node.as_ref().unwrap().schema.name()))
            .count()
    };
    assert_eq!(impure_count(optimized), impure_count(source), "seed {seed}");
}

#[test]
fn random_graph_optimization_is_exact_idempotent_and_round_trips() {
    // Seeds in which each pass rewrote at least one node.
    let mut seeds_rewritten = [0; 3];
    for seed in 0..SEEDS {
        let (source, inputs) = random_graph(seed);
        let outcome = optimize(assert_round_trip(seed, &source)).unwrap();
        for (count, pass) in seeds_rewritten.iter_mut().zip(&outcome.report.passes) {
            *count += u64::from(pass.rewritten_nodes > 0);
        }
        assert_remap(seed, &source, &outcome.store, &outcome.old_to_new);

        // Cleanup reuses deterministic results, moves data exactly, and
        // removes only unobserved nodes, so results agree bit for bit. The
        // rebuilt store is executed as decoded from its own canonical artifact.
        let optimized = Arc::new(assert_round_trip(seed, &outcome.store));
        let expected = execute(Arc::new(source), inputs.clone());
        let actual = execute(Arc::clone(&optimized), inputs);
        assert_eq!(
            actual.iter().map(Array::bits).collect::<Vec<_>>(),
            expected.iter().map(Array::bits).collect::<Vec<_>>(),
            "seed {seed}"
        );

        // One run reaches the fixed point.
        let optimized = Arc::try_unwrap(optimized).unwrap();
        let identity = (0..optimized.node_count())
            .map(|index| Some(NodeId::try_from(index).unwrap()))
            .collect::<Vec<_>>();
        let again = optimize(optimized).unwrap();
        assert_eq!(again.old_to_new, identity, "seed {seed}");
        assert_eq!(again.report.rewritten_nodes, 0, "seed {seed}");
    }
    assert!(
        seeds_rewritten.iter().all(|&count| count > SEEDS / 2),
        "passes rewrote too few graphs: {seeds_rewritten:?}"
    );
}
