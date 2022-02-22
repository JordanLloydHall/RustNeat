use petgraph::{
    graph::{node_index, DiGraph, NodeIndex},
    visit::Bfs,
    Direction,
};
use rand;
use std::{
    collections::{HashSet, VecDeque},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

type InnovationCounter = Arc<AtomicUsize>;

#[derive(Copy, Clone, Debug)]
struct Node {
    activation: Activation,
}

#[derive(Copy, Clone, Debug)]
enum Activation {
    Linear,
    Sigmoid,
}

impl Activation {
    fn apply(&self, x: f32) -> f32 {
        match self {
            Self::Linear => x,
            Self::Sigmoid => 1. / (1. + (-x).exp()),
        }
    }
}

#[derive(Copy, Clone, Debug)]
struct Connection {
    weight: f32,
    innovation_number: usize,
    enabled: bool,
}

#[derive(Clone, Debug)]
struct Network<const I: usize, const O: usize> {
    graph: DiGraph<Node, Connection>,
    innvoation_counter: InnovationCounter,
}

impl<const I: usize, const O: usize> Network<I, O> {
    fn new(innovation_counter: InnovationCounter) -> Self {
        let mut graph = DiGraph::<Node, Connection>::new();

        for i in 0..I {
            graph.add_node(Node {
                activation: Activation::Linear,
            });
        }

        for i in 0..O {
            graph.add_node(Node {
                activation: Activation::Sigmoid,
            });
        }

        let mut counter = 0;

        for i in 0..I {
            for j in 0..O {
                graph.update_edge(
                    NodeIndex::new(i),
                    NodeIndex::new(I + j),
                    Connection {
                        weight: rand::random(),
                        innovation_number: counter,
                        enabled: true,
                    },
                );
                counter += 1;
            }
        }

        innovation_counter.fetch_max(counter, Ordering::SeqCst);

        Network {
            graph,
            innvoation_counter: innovation_counter,
        }
    }

    fn feed_forward(&self, inputs: [f32; I]) -> [f32; O] {
        let mut node_outputs = vec![0f32; self.graph.node_count()];

        for i in 0..I {
            node_outputs[i] = self.graph[NodeIndex::new(i + I)]
                .activation
                .apply(inputs[i]);
        }

        let mut visited = HashSet::<(usize, usize)>::new();
        let mut queue = VecDeque::<(usize, usize)>::new();
        for i in 0..I {
            let nodes = self
                .graph
                .neighbors_directed(NodeIndex::new(i), Direction::Outgoing);
            for e in nodes {
                queue.push_back((i, e.index()));
            }
        }

        while let Some(edge) = queue.pop_front() {
            if visited.contains(&edge) {
                continue;
            }
            visited.insert(edge);

            let (i, o) = edge;
            let edge_index = self
                .graph
                .find_edge(node_index(i), node_index(o))
                .unwrap()
                .index();
            let conn = self.graph.raw_edges()[edge_index].weight;

            if conn.enabled {
                node_outputs[o] += self.graph[NodeIndex::new(i)]
                    .activation
                    .apply(node_outputs[i])
                    * conn.weight;
            }

            let nodes = self
                .graph
                .neighbors_directed(NodeIndex::new(o), Direction::Outgoing);
            for e in nodes {
                queue.push_back((o, e.index()));
            }
        }

        let mut outputs = [0f32; O];
        for (i, v) in node_outputs[I..I + O].iter().enumerate() {
            println!("{:?}", self.graph[NodeIndex::new(i + I)]);
            outputs[i] = self.graph[NodeIndex::new(i + I)].activation.apply(*v);
        }

        outputs
    }

    fn add_connection(&mut self, (i, o): (usize, usize), weight: f32) {
        assert!(self.graph.node_count() >= i);
        assert!(self.graph.node_count() >= o);

        let new_conn;
        if self
            .graph
            .contains_edge(NodeIndex::new(i), NodeIndex::new(o))
        {
            let old_conn_index = self
                .graph
                .find_edge(node_index(i), node_index(o))
                .unwrap()
                .index();
            let old_conn = self.graph.raw_edges()[old_conn_index].weight;
            new_conn = Connection {
                weight: old_conn.weight + weight,
                ..old_conn
            }
        } else {
            new_conn = Connection {
                weight: weight,
                innovation_number: self.innvoation_counter.fetch_add(1, Ordering::SeqCst),
                enabled: true,
            }
        }
        self.graph
            .update_edge(NodeIndex::new(i), NodeIndex::new(o), new_conn);
    }

    fn add_node(&mut self, (i, o): (usize, usize)) {
        assert!(self
            .graph
            .contains_edge(NodeIndex::new(i), NodeIndex::new(o)));

        let old_conn_index = self
            .graph
            .find_edge(node_index(i), node_index(o))
            .unwrap()
            .index();
        let old_conn = self.graph.raw_edges()[old_conn_index].weight;

        self.graph.update_edge(
            NodeIndex::new(i),
            NodeIndex::new(o),
            Connection {
                enabled: false,
                ..old_conn
            },
        );

        let new_node = self.graph.add_node(Node {
            activation: Activation::Linear,
        });

        self.graph.update_edge(
            NodeIndex::new(i),
            new_node,
            Connection {
                innovation_number: self.innvoation_counter.fetch_add(1, Ordering::SeqCst),
                ..old_conn
            },
        );

        self.graph.update_edge(
            new_node,
            NodeIndex::new(o),
            Connection {
                weight: 1.,
                innovation_number: self.innvoation_counter.fetch_add(1, Ordering::SeqCst),
                enabled: true,
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use petgraph::dot::{Config, Dot};
    use super::*;

    #[test]
    fn add_node() {
        let innov = InnovationCounter::new(AtomicUsize::new(0));
        let mut network: Network<2, 2> = Network::new(innov.clone());
        let old_out = network.feed_forward([8.8, 4.4]);
        println!("{:?}", Dot::with_config(&network.graph, &[]));
        network.add_node((0, 2));
        println!("{:?}", Dot::with_config(&network.graph, &[]));
        assert_eq!(innov.load(Ordering::SeqCst), 6);
        assert_eq!(old_out, network.feed_forward([8.8, 4.4]));
        assert_eq!(network.graph.edge_count(), 6);
        assert_eq!(network.graph.node_count(), 5);
    }
    
    #[test]
    fn add_connection_duplicate() {
        let innov = InnovationCounter::new(AtomicUsize::new(0));
        let mut network: Network<2, 2> = Network::new(innov.clone());
        println!("{:?}", Dot::with_config(&network.graph, &[]));
        network.add_connection((0, 2), 0.1);
        println!("{:?}", Dot::with_config(&network.graph, &[]));
        assert_eq!(innov.load(Ordering::SeqCst), 4);
        assert_eq!(network.graph.edge_count(), 4);
        assert_eq!(network.graph.node_count(), 4);
    }

    #[test]
    fn add_connection() {
        let innov = InnovationCounter::new(AtomicUsize::new(0));
        let mut network: Network<2, 2> = Network::new(innov.clone());
        println!("{:?}", Dot::with_config(&network.graph, &[]));
        network.add_connection((2, 0), 0.1);
        println!("{:?}", Dot::with_config(&network.graph, &[]));
        assert_eq!(innov.load(Ordering::SeqCst), 5);
        assert_eq!(network.graph.edge_count(), 5);
        assert_eq!(network.graph.node_count(), 4);
    }
}
