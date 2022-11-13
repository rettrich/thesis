module Training

using Graphs
using GraphNeuralNetworks
using thesis
using Distributions
using DataStructures
using Statistics
using StatsBase
using Flux
using thesis.LookaheadSearch
using thesis.Instances: generate_instance
using thesis.LocalSearch: run_lsbmh, LocalSearchBasedMH, sample_candidate_solutions
using thesis.GNNs: GNNModel, device, NodeFeature, get_feature_list, batch_support, get_decoder_features, evaluate, loss_func_unbatched
using Printf
using Logging, TensorBoardLogger
using BSON

export TrainingSample, ReplayBuffer,
       add_to_buffer!, get_data, create_sample,
       InstanceGenerator, sample_graph

"""
    TrainingSample

Contains all relevant data used for a training sample:

- `gnn_graph`: A `GNNGraph` that can be used to train the GNN. Node features and targets are 
    contained as `gnn_graph.ndata.x` and `gnn_graph.ndata.y`, respectively
- `S`: The candidate solution used to create the sample. 
"""
struct TrainingSample
    gnn_graph::GNNGraph
    S::Union{Vector{Int}, Set{Int}}

    function TrainingSample(gnn_graph::GNNGraph, S::Union{Vector{Int}, Set{Int}})
        new(gnn_graph, S)
    end
end

"""
    ReplayBuffer

Simple replay buffer implementation realized as a FIFO Queue. 

- `min_fill`: Minimum fill which is required before training is started. 
- `capacity`: Maximum capacity of the buffer.
- `buffer`: A `Deque` of `TrainingSample`s

"""
struct ReplayBuffer
    min_fill::Int
    capacity::Int
    buffer::Deque{TrainingSample}

    function ReplayBuffer(min_fill::Int, capacity::Int)
        new(min_fill, capacity, Deque{TrainingSample}())
    end
end

"""
    add_to_buffer!(buffer, g, S, lookahead_func, node_features)

Creates a training sample from `g, S` and adds it to the buffer. Target values are computed 
by using `lookahead_func`

- `buffer`: The `ReplayBuffer` instance the sample is added to
- `g`: The input graph 
- `S`: The candidate solution `S ⊆ V`
- `lookahead_func`: The lookahead search function that is used to identify the best neighboring solutions of `S` 
    in order to calculate target values. 
- `node_features`: Feature matrix for vertices in `g`, used as inputs for training

"""
function add_to_buffer!(buffer::ReplayBuffer, g::SimpleGraph, S::Set{Int}, 
                        lookahead_func::LookaheadSearchFunction, node_features::AbstractMatrix,
                        decoder_features::Union{Nothing, Vector{<:NodeFeature}}=nothing,
                        gnn::Union{GNNModel, Nothing}=nothing)
    sample, is_local_optimum = create_sample(g, S, lookahead_func, node_features, decoder_features, gnn)

    pushfirst!(buffer.buffer, sample)

    if length(buffer) > buffer.capacity
        _ = pop!(buffer.buffer)
    end
    return is_local_optimum
end

"""
    add_to_buffer!(buffer, g, S, lookahead_func, node_features)

Creates training samples for a graph `g` and multiple candidate solutions `Ss` and adds 
them to the buffer. 

- `buffer`: The `ReplayBuffer` instance the sample is added to
- `g`: The input graph 
- `Ss`: A set of candidate solution, `S ⊆ V` for `S ∈ Ss`
- `lookahead_func`: The lookahead search function that is used to identify the best neighboring solutions of `S` 
    in order to calculate target values. 
- `node_features`: Feature matrix for vertices in `g`, used as inputs for training

"""
function add_to_buffer!(buffer::ReplayBuffer, g::SimpleGraph, Ss::Vector{Set{Int}}, 
                        lookahead_func::LookaheadSearchFunction, node_features::AbstractMatrix,
                        decoder_features::Union{Nothing, Vector{<:NodeFeature}}=nothing,
                        gnn::Union{GNNModel, Nothing}=nothing)
    local_optima = 0
    for S in Ss
        is_local_optimum = add_to_buffer!(buffer, g, S, lookahead_func, node_features, decoder_features, gnn)
        is_local_optimum && (local_optima += 1)
    end
    return local_optima
end

Base.length(buffer::ReplayBuffer) = length(buffer.buffer)

function get_data(buffer::ReplayBuffer; batchsize=32)
    data = map(sample -> sample.gnn_graph, collect(buffer.buffer))
    return Flux.DataLoader(data, batchsize=batchsize, shuffle=true, collate=true)
end

"""
    create_sample(graph, S, lookahead_func)

Creates a training sample consisting of a graph `graph` and a candidate solution `S`. 
Target values are computed using `lookahead_func` to identify the best neighboring solution(s).
Returns an instance of `TrainingSample`, where the fields `gnn_graph.ndata.x` and `gnn_graph.ndata.y` 
contain the inputs and target values for training, respectively, and a boolean indicating whether the 
sample is a local optimum with respect to the given lookahead search. 

- `graph`: Graph for the training sample
- `S`: Candidate solution, used to create training sample
- `lookahead_func`: Lookahead search function used to compute target values
- `node_features`: Inputs for the GNN, feature matrix with node features for vertices in `graph`

"""
function create_sample(graph::SimpleGraph{Int},
                       S::Union{Set{Int}, Vector{Int}},
                       lookahead_func::LookaheadSearchFunction,
                       node_features::AbstractMatrix,
                       decoder_features::Union{Nothing, Vector{<:NodeFeature}}=nothing,
                       gnn::Union{GNNModel, Nothing}=nothing,
                       )::Tuple{TrainingSample, Bool}
    if typeof(S) <: Set{Int}
        S_vec = collect(S)
    else
        S_vec = copy(S)
    end

    d_S = thesis.LocalSearch.calculate_d_S(graph, S)

    # use lookahead function to obtain best neighboring solutions
    if isnothing(gnn)
        obj_val, solutions = lookahead_func(graph, S, d_S)
    else
        gnn_g = GNNGraph(graph)
        gnn_g = add_self_loops(gnn_g)
        scores = vec(evaluate(gnn, gnn_g, node_features, S, d_S))
        obj_val, solutions = lookahead_func(graph, S, d_S; scores)
    end
    
    targets = fill(0.0f0, nv(graph))

    in_S = fill(0, nv(graph))
    in_S[S_vec] .= 1 # mark vertices in S for later computation of context

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

    if !isnothing(decoder_features)
        decoder_in = thesis.GNNs.compute_node_features(decoder_features, graph, S, d_S)
        
        # create GNNGraph
        gnn_graph = GNNGraph(graph,
            ndata=(; x = node_features, y = targets, in_S = in_S, decoder_features = decoder_in)
        )
    else
        gnn_graph = GNNGraph(graph,
            ndata=(; x = node_features, y = targets, in_S = in_S)
        )
    end

    gnn_graph = add_self_loops(gnn_graph) # add self loops for message passing

    # return training sample and a boolean indicating whether this sample is a local optimum wrt lookahead search
    return TrainingSample(gnn_graph, S), isempty(solutions)
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
    ensure_connectivity::Bool

    function InstanceGenerator(nv_sampler, density_sampler; ensure_connectivity=false)
        new(nv_sampler, density_sampler, ensure_connectivity)
    end
end

Base.show(io::IO, ::MIME"text/plain", x::InstanceGenerator) = print(io, "V=$(fields_to_string(x.nv_sampler))-dens=$(fields_to_string(x.density_sampler))")

function fields_to_string(x)
    fields = fieldnames(typeof(x))
    props = [getproperty(x, field) for field in fields]
    join(props, "-")
end

"""
    sample_graph(generator)

Returns a random graph instance, where the number of vertices and the 
density are sampled randomly according to the distributions in the `instance_generator` instance.

"""
function sample_graph(instance_generator::InstanceGenerator)::SimpleGraph
    V = round(Int, rand(instance_generator.nv_sampler))
    dens = rand(instance_generator.density_sampler)

    if instance_generator.ensure_connectivity
        generate_connected_instance(V, dens)
    else
        generate_instance(V, dens)
    end
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
- `baseline`: Local search function to compare the gnn based local search to
- `num_samples`: The number of samples generated per epoch
- `num_batches`: The number of batches used for training each epoch
- `batchsize`: Number of samples in each batch
- `warm_up`: Use d_S as scoring function for the lookahead search first `warm_up` iterations, 
    to start using the gnn based scoring function only when it is already somewhat trained.  
- `buffer_capacity`: Maximum capacity of replay buffer.

"""
function train!(local_search::LocalSearchBasedMH, instance_generator::InstanceGenerator, gnn::GNNModel; 
               epochs=200, lookahead_func=Ω_1_LookaheadSearchFunction(), baseline::Union{Nothing, LocalSearchBasedMH}=nothing,
               num_samples::Int = 25, num_batches::Int = 4, batchsize::Int = 8, warm_up::Int = 50, 
               buffer_capacity::Int = 1000,
               logger::Union{Nothing, TBLogger}=nothing
               )
    buffer_capacity = 1000
    min_fill = round(Int, buffer_capacity / 2)
    buffer = ReplayBuffer(min_fill, buffer_capacity)
    ps = Flux.params(gnn)

    t_baseline = 0
    s_base = 0

    # check if gnn supports batching of graphs. for encoder / decoder it is not possible for now, as looping over graphs in batch is not possible
    # https://github.com/CarloLucibello/GraphNeuralNetworks.jl/issues/161
    batchsize = batch_support(gnn) ? batchsize : 1
    num_batches = batch_support(gnn) ? num_batches : (num_batches, batchsize)

    loss(g::GNNGraph, S::Vector{Int}) = loss_func_unbatched(gnn, g, S) 

    @printf("Iteration | encountered |   t_ls | t_base | t_targets | t_train | (opt)buffer | loss | V/density | solution | baseline |   gap | free/total memory\n")

    for i = 1:epochs
        
        graph = sample_graph(instance_generator)
        
        t_ls = @elapsed solution, swap_history = run_lsbmh(local_search, graph)
        s_ls = length(solution)

        if !isnothing(baseline)
            t_baseline = @elapsed baseline_sol, _ = run_lsbmh(baseline, graph)
            s_base = length(baseline_sol)
            gap = (s_ls - s_base) / s_base
        end

        data = sample_candidate_solutions(swap_history, num_samples)

        if use_scoring_vector(lookahead_func) && i >= warm_up
            gnn_for_scores = gnn
        else
            gnn_for_scores = nothing
        end

        t_targets = @elapsed (local_optima = add_to_buffer!(buffer, graph, data.samples, 
                                                            lookahead_func, data.node_features, 
                                                            get_decoder_features(gnn), gnn_for_scores))
        t_train = 0
        iter_loss = NaN

        free_mem = Int(Sys.free_memory()) / 2^30
        total_mem = Int(Sys.total_memory()) / 2^30

        if length(buffer) < buffer.min_fill
            @printf("%9i %13i %8.3f %8.3f %11.3f %9.3f   (%3i)%6i %6.3f %5i/%4.3f %10i %10i %7.3f   %5.3f/%5.3f\n", 
                    i, length(swap_history),
                    t_ls, t_baseline, t_targets, 0, 
                    local_optima, length(buffer), 
                    iter_loss, 
                    nv(graph), density(graph), 
                    s_ls, s_base, gap,
                    free_mem, total_mem)
            if !isnothing(logger)
                with_logger(logger) do 
                    @info("thesis", t_ls, t_baseline, t_targets, t_train, iter_loss, V=nv(graph), dens=density(graph), s_ls, s_base, gap, free_mem, total_mem)
                end
            end
            continue
        end


        before_training = time()

        train_loader = get_data(buffer; batchsize) # batch data accordingly 
        # loss(g::GNNGraph) = Flux.logitbinarycrossentropy( vec(gnn(g, g.ndata.x)), g.ndata.y)

        if batch_support(gnn)
            error("currently not supported")
            # losses = []
            # for g in first(train_loader, num_batches)
            #     g = g |> device
            #     gs = gradient(ps) do 
            #         gnn.loss(g)
            #     end
            #     push!(losses, gnn.loss(g))
            #     Flux.Optimise.update!(gnn.opt, ps, gs)
            # end
            # iter_loss = mean(losses)
        else
            # only unbatched training is possible atm (batch consists of individual samples)
            losses = []
            for _ in 1:num_batches[1]
                graphs = first(train_loader, num_batches[2]) # get batch 
                Ss = [filter(v -> g.ndata.in_S[v]==1, 1:nv(g)) for g in graphs] # obtain candidate solution for each graph
                graphs = [g |> device for g in graphs]
                gs = gradient(ps) do 
                    iter_loss = mean( loss.(graphs, Ss) )
                    iter_loss                    
                end
                push!(losses, iter_loss)
                Flux.Optimise.update!(gnn.opt, ps, gs)
            end
            iter_loss = mean(losses)
        end

        t_train = time() - before_training

        @printf("%9i %13i %8.3f %8.3f %11.3f %9.3f   (%3i)%6i %6.3f %5i/%4.3f %10i %10i %7.3f   %5.3f/%5.3f\n", 
                    i, length(swap_history),
                    t_ls, t_baseline, t_targets, t_train, 
                    local_optima, length(buffer), 
                    iter_loss, 
                    nv(graph), density(graph), 
                    s_ls, s_base, gap,
                    free_mem, total_mem)

        if !isnothing(logger)
            with_logger(logger) do 
                @info("thesis", t_ls, t_baseline, t_targets, t_train, iter_loss, V=nv(graph), dens=density(graph), s_ls, s_base, gap, free_mem, total_mem)
            end
        end

        GC.gc() # maybe this helps with memory errors?

    end


    # idxs = sample(1:length(buffer), 1000; replace=false)
    # training_samples = collect(buffer.buffer)[idxs] # write samples to file
    # BSON.@save "training_samples.bson" training_samples
    

end


end # module