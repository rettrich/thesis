module Training

using Graphs
using GraphNeuralNetworks
using thesis
using Distributions
using DataStructures
using Flux
using LookaheadSearch

export TrainingSample, ReplayBuffer,
       add_to_buffer!, get_data, compute_node_features, create_sample,
       InstanceGenerator, sample_graph

"""
    TrainingSample

Contains all relevant data used for a training sample:

- `gnn_graph`: A `GNNGraph` that can be used to train the GNN. Node features and targets are 
    contained as `gnn_graph.ndata.x` and `gnn_graph.ndata.y`, respectively
- `graph`: The original graph used to create the sample. 
- `S`: The candidate solution used to create the sample. 
"""
struct TrainingSample
    gnn_graph::GNNGraph
    graph::SimpleGraph
    S::Set{Int}

    function TrainingSample(gnn_graph::GNNGraph, graph::SimpleGraph, S::Set{Int})
        new(gnn_graph, graph, S)
    end
end

struct ReplayBuffer
    min_fill::Int
    capacity::Int
    buffer::Deque{TrainingSample}
    lookahead_search::LookaheadSearchFunction # used to compute target values when adding samples to buffer

    function ReplayBuffer(min_fill::Int, capacity::Int, lookahead_search::LookaheadSearchFunction)
        new(min_fill, capacity, Deque{TrainingSample}(), lookahead_search)
    end
end

"""
    add_to_buffer!(buffer, g, S)

Creates a training sample from `g, S` and adds it to the buffer. 

"""
function add_to_buffer!(buffer::ReplayBuffer, g::SimpleGraph, S::Set{Int})
    gnn_graph = create_sample(g, S, buffer.lookahead_search)

    pushfirst!(buffer, TrainingSample(gnn_graph, g, S))

    if length(buffer) > buffer.capacity
        pop!(buffer)
    end
end

"""
    add_to_buffer!(buffer, g, S)

Creates training samples for a graph `g` and multiple candidate solutions `Ss` and adds 
them to the buffer. 

"""
function add_to_buffer!(buffer::ReplayBuffer, g::SimpleGraph, Ss::Vector{Set{Int}})
    for S in Ss
        add_to_buffer!(buffer, g, S)
    end
end

Base.length(buffer::ReplayBuffer) = length(buffer.buffer)

function get_data(buffer::ReplayBuffer)
    data = map(sample -> sample.gnn_graph, collect(buffer.buffer))
    return Flux.DataLoader(data, batchsize=32, shuffle=true, collate=true)
end


function compute_node_features(graph::SimpleGraph, S::Union{Set{Int}, Vector{Int}}, d_S::Vector{Int})
    degrees = degree(graph)
    node_features = Float32.(vcat(degrees', d_S'))
    return node_features
end

function create_sample(graph::SimpleGraph{Int},
                       S::Union{Set{Int}, Vector{Int}},
                       lookahead_func::LookaheadSearchFunction
                       )::GNNGraph

    # node features
    d_S = thesis.LocalSearch.calculate_d_S(graph, S)
    node_features = compute_node_features(graph, S, d_S)

    # use lookahead function to obtain best neighboring solutions
    obj_val, solutions = lookahead_func(graph, S, d_S)
    targets = fill(0.0f0, nv(graph))

    # compute target node labels
    for v in S
    targets[v] = 1.0
    end

    for (in_nodes, out_nodes) in solutions
    # nodes that are in S and not in every best neighboring solution are scored lower
    # so they are considered for swaps
        for u in in_nodes
        targets[u] = 0.0
        end
        # mark nodes that are attractive for swaps with 1
        for v in out_nodes
        targets[v] = 1.0
        end
    end

    # create GNNGraph
    gnn_graph = GNNGraph(graph,
        ndata=(; x = node_features, y = targets)
        )
    add_self_loops(gnn_graph) # add self loops for message passing
    return gnn_graph
end

"""
    InstanceGenerator

Can be used to sample random instances, where `nv_sampler` is used to 
sample the number of vertices, and `density_sampler` is used to 
sample the density of generated graphs. 

"""
struct InstanceGenerator
    nv_sampler::Sampleable
    density_sampler::Sampleable

    function InstanceGenerator(nv_sampler, density_sampler)
        new(nv_sampler, density_sampler)
    end
end

"""
    sample_graph(generator)

Returns a random graph instance, where the number of vertices and the 
density are sampled randomly according to the distributions in the `generator` instance.

"""
function sample_graph(generator::InstanceGenerator)::SimpleGraph
    return thesis.Instances.generate_instance(
        round(Int, rand(generator.nv_sampler)), 
        rand(generator.density_sampler)
        )
end


end # module