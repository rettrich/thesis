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
using thesis.GNNs: GNNModel, device, NodeFeature, get_feature_list
using Printf
using Logging, TensorBoardLogger

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
function add_to_buffer!(buffer::ReplayBuffer, g::SimpleGraph, S::Set{Int}, lookahead_func::LookaheadSearchFunction, feature_list::Vector{<:NodeFeature})
    sample, is_local_optimum = create_sample(g, S, lookahead_func, feature_list)

    pushfirst!(buffer.buffer, sample)

    if length(buffer) > buffer.capacity
        pop!(buffer.buffer)
    end
    return is_local_optimum
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
function add_to_buffer!(buffer::ReplayBuffer, g::SimpleGraph, Ss::Vector{Set{Int}}, lookahead_func::LookaheadSearchFunction, feature_list::Vector{<:NodeFeature})
    local_optima = 0
    for S in Ss
        is_local_optimum = add_to_buffer!(buffer, g, S, lookahead_func, feature_list)
        is_local_optimum && (local_optima += 1)
    end
    return local_optima
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
                       lookahead_func::LookaheadSearchFunction,
                       feature_list::Vector{<:NodeFeature}
                       )::Tuple{TrainingSample, Bool}
    if typeof(S) <: Set{Int}
        S_vec = collect(S)
    else
        S_vec = S
    end
    # node features
    d_S = thesis.LocalSearch.calculate_d_S(graph, S)
    node_features = thesis.GNNs.compute_node_features(feature_list, graph, S, d_S)

    # use lookahead function to obtain best neighboring solutions
    obj_val, solutions = lookahead_func(graph, S, d_S)
    targets = fill(0.0f0, nv(graph))

    in_S = fill(0, nv(graph))
    in_S[S_vec] .= 1

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
        ndata=(; x = node_features, y = targets, in_S = in_S)
        )
    gnn_graph = add_self_loops(gnn_graph) # add self loops for message passing

    # return training sample and a boolean indicating whether this sample is a local optimum wrt lookahead search
    return TrainingSample(gnn_graph, graph, S), isempty(solutions)
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

Base.show(io::IO, ::MIME"text/plain", x::InstanceGenerator) = print(io, "V=$(x.nv_sampler)-dens=$(x.density_sampler)")


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
               epochs=200, lookahead_func=Ω_1_LookaheadSearchFunction(), baseline::Union{Nothing, LocalSearchBasedMH}=nothing,
               num_batches=2, logger::Union{Nothing, TBLogger}=nothing
               )
    capacity = 4000
    min_fill = 2000
    buffer = ReplayBuffer(min_fill, capacity, lookahead_func)
    ps = Flux.params(gnn.model)

    t_baseline = 0
    s_base = 0


    @printf("Iteration | encountered |   t_ls | t_base | t_targets | t_train | (opt)buffer | loss | V/density | solution | baseline \n")

    for i = 1:epochs
        
        graph = sample_graph(instance_generator)
        
        t_ls = @elapsed solution, swap_history = run_lsbmh(local_search, graph)
        s_ls = length(solution)

        if !isnothing(baseline)
            t_baseline = @elapsed baseline_sol, _ = run_lsbmh(baseline, graph)
            s_base = length(baseline_sol)
        end

        _, samples = sample_candidate_solutions(swap_history, 100)

        t_targets = @elapsed (local_optima = add_to_buffer!(buffer, graph, samples, lookahead_func, get_feature_list(gnn)))
        t_train = 0
        iter_loss = NaN

        if length(buffer) < buffer.min_fill
            @printf("%9i %13i %8.3f %8.3f %11.3f %9.3f   (%3i)%6i %6.3f %5i/%4.3f %10i %10i\n", 
                    i, length(swap_history),
                    t_ls, t_baseline, t_targets, 0, 
                    local_optima, length(buffer), 
                    iter_loss, 
                    nv(graph), density(graph), 
                    s_ls, s_base)
            if !isnothing(logger)
                with_logger(logger) do 
                    @info("thesis", t_ls, t_baseline, t_targets, t_train, iter_loss, V=nv(graph), dens=density(graph), s_ls, s_base)
                end
            end
            continue
        end


        before_training = time()

        train_loader = get_data(buffer)
        # loss(g::GNNGraph) = Flux.logitbinarycrossentropy( vec(gnn(g, g.ndata.x)), g.ndata.y)

        losses = []
        for g in first(train_loader, num_batches)
            g = g |> device
            gs = gradient(ps) do 
                gnn.loss(g)
            end
            push!(losses, gnn.loss(g))
            Flux.Optimise.update!(gnn.opt, ps, gs)
        end
        iter_loss = mean(losses)

        t_train = time() - before_training

        @printf("%9i %13i %8.3f %8.3f %11.3f %9.3f   (%3i)%6i %6.3f %5i/%4.3f %10i %10i\n", 
                    i, length(swap_history),
                    t_ls, t_baseline, t_targets, t_train, 
                    local_optima, length(buffer), 
                    iter_loss, 
                    nv(graph), density(graph), 
                    s_ls, s_base)

        if !isnothing(logger)
            with_logger(logger) do 
                @info("thesis", t_ls, t_baseline, t_targets, t_train, iter_loss, V=nv(graph), dens=density(graph), s_ls, s_base)
            end
        end

    end
end


end # module