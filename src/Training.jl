module Training

using Graphs
using GraphNeuralNetworks
using thesis
using Distributions
using DataStructures
using Statistics
using Flux
using thesis.LookaheadSearch
using thesis.Instances: generate_instance
using thesis.LocalSearch: run_lsbmh, LocalSearchBasedMH, sample_candidate_solutions
using thesis.GNNs: GNNModel, device
using Printf

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
    S::Union{Vector{Int}, Set{Int}}

    function TrainingSample(gnn_graph::GNNGraph, graph::SimpleGraph, S::Union{Vector{Int}, Set{Int}})
        new(gnn_graph, graph, S)
    end
end

"""
    ReplayBuffer

Simple replay buffer implementation realized as a FIFO Queue. 

- `min_fill`: Minimum fill which is required before training is started. 
- `capacity`: Maximum capacity of the buffer.
- `buffer`: A `Deque` of `TrainingSample`s
- `lookahead_func`: 

"""
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
    add_to_buffer!(buffer, g, S, lookahead_func)

Creates a training sample from `g, S` and adds it to the buffer. Target values are computed 
by using `lookahead_func`

- `buffer`: The `ReplayBuffer` instance the sample is added to
- `g`: The input graph 
- `S`: The candidate solution `S ⊆ V`
- `lookahead_func`: The lookahead search function that is used to identify the best neighboring solutions of `S` 
    in order to calculate target values. 

"""
function add_to_buffer!(buffer::ReplayBuffer, g::SimpleGraph, S::Set{Int}, lookahead_func::LookaheadSearchFunction)
    sample = create_sample(g, S, lookahead_func)

    pushfirst!(buffer.buffer, sample)

    if length(buffer) > buffer.capacity
        pop!(buffer.buffer)
    end
end

"""
    add_to_buffer!(buffer, g, S, lookahead_func)

Creates training samples for a graph `g` and multiple candidate solutions `Ss` and adds 
them to the buffer. 

- `buffer`: The `ReplayBuffer` instance the sample is added to
- `g`: The input graph 
- `Ss`: A set of candidate solution, `S ⊆ V` for `S ∈ Ss`
- `lookahead_func`: The lookahead search function that is used to identify the best neighboring solutions of `S` 
    in order to calculate target values. 

"""
function add_to_buffer!(buffer::ReplayBuffer, g::SimpleGraph, Ss::Vector{Set{Int}}, lookahead_func::LookaheadSearchFunction)
    for S in Ss
        add_to_buffer!(buffer, g, S, lookahead_func)
    end
end

Base.length(buffer::ReplayBuffer) = length(buffer.buffer)

function get_data(buffer::ReplayBuffer)
    data = map(sample -> sample.gnn_graph, collect(buffer.buffer))
    return Flux.DataLoader(data, batchsize=32, shuffle=true, collate=true)
end

"""
    create_sample(graph, S, lookahead_func)

Creates a training sample consisting of a graph `graph` and a candidate solution `S`. 
Target values are computed using `lookahead_func` to identify the best neighboring solution(s).
Returns an instance of `TrainingSample`, where the fields `gnn_graph.ndata.x` and `gnn_graph.ndata.y` 
contain the inputs and target values for training, respectively. 

"""
function create_sample(graph::SimpleGraph{Int},
                       S::Union{Set{Int}, Vector{Int}},
                       lookahead_func::LookaheadSearchFunction
                       )::TrainingSample

    # node features
    d_S = thesis.LocalSearch.calculate_d_S(graph, S)
    node_features = thesis.GNNs.compute_node_features(graph, d_S)

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
    gnn_graph = add_self_loops(gnn_graph) # add self loops for message passing
    return TrainingSample(gnn_graph, graph, S)
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
density are sampled randomly according to the distributions in the `instance_generator` instance.

"""
function sample_graph(instance_generator::InstanceGenerator)::SimpleGraph
    return generate_instance(
        round(Int, rand(instance_generator.nv_sampler)), 
        rand(instance_generator.density_sampler)
        )
end

"""
    train!(local_search, instance_generator, gnn; iter, lookahead_func)

Train `gnn` by performing `local_search` on graphs created by `instance_generator` for `iter` iterations. 
Target values for training are computed using `lookahead_func`. 

- `local_search`: A `LocalSearchBasedMH` instance that is used to collect observations
- `instance_generator`: Used to sample random instances 
- `gnn`: The GNN to be trained
- `epochs`: Number of epochs of training
- `lookahead_func`: Lookahead search function used to compute target values for training 

"""
function train!(local_search::LocalSearchBasedMH, instance_generator::InstanceGenerator, gnn::GNNModel; 
               epochs=100, lookahead_func=Ω_1_LookaheadSearchFunction()
               )
    capacity = 2000
    min_fill = 1000
    buffer = ReplayBuffer(min_fill, capacity, lookahead_func)
    ps = Flux.params(gnn.model)

    @printf("Iteration | t_local_search | t_compute_targets | t_train | buffer | loss \n")

    for i = 1:epochs
        start_time = time()
        
        graph = sample_graph(instance_generator)

        solution, swap_history = run_lsbmh(local_search, graph)
        
        after_local_search = time()

        _, samples = sample_candidate_solutions(swap_history, 100)
        add_to_buffer!(buffer, graph, samples, lookahead_func)

        after_add_to_buffer = time()

        if length(buffer) < buffer.min_fill
            @printf("%9i %16.3f %19.3f %9.3f %8i %6.3f\n", 
                    i, after_local_search - start_time, after_add_to_buffer - after_local_search, 0, 
                    length(buffer), NaN)
            continue
        end


        train_loader = get_data(buffer)
        loss(g::GNNGraph) = Flux.logitbinarycrossentropy( vec(gnn(g, g.ndata.x)), g.ndata.y)
        

        losses = []
        for g in first(train_loader, 4)
            g = g |> device
            gs = gradient(ps) do 
                loss(g)
            end
            push!(losses, loss(g))
            Flux.Optimise.update!(gnn.opt, ps, gs)
        end

        after_training = time()

        @printf("%9i %16.3f %19.3f %9.3f %8i %6.3f\n", 
                    i, after_local_search - start_time, after_add_to_buffer - after_local_search, after_training - after_add_to_buffer, 
                    length(buffer), mean(losses))

    end
    return gnn
end


end # module