module GNNs

using Flux, Graphs, GraphNeuralNetworks, CUDA
using Statistics
using thesis.NodeRepresentationLearning
# using BSON
# using thesis
# using Statistics
# using Printf
# using MLUtils
# using Logging

export GNNModel, SimpleGNN, Encoder_Decoder_GNNModel, compute_node_features, device,
    NodeFeature, d_S_NodeFeature, DegreeNodeFeature, DeepWalkNodeFeature, EgoNetNodeFeature, PageRankNodeFeature,
    get_feature_list,
    GNNChainFactory, ResGatedGraphConv_GNNChainFactory, GATv2Conv_GNNChainFactory,
    batch_support

# no trailing commas in export!

# device = CUDA.functional() ? Flux.gpu : Flux.cpu
device = Flux.cpu

ENV["JULIA_DEBUG"] = "thesis"


abstract type GNNModel end

Flux.params(gnn::GNNModel) = Flux.params(gnn.model)

(gnn::GNNModel)(gnn_graph::GNNGraph, inputs::AbstractMatrix) = gnn.model(gnn_graph, inputs) # TODO: only relevant for SimpleGNN

batch_support(gnn::GNNModel)::Bool = gnn.batch_support

abstract type NodeFeature end

get_feature_list(gnn::GNNModel) = gnn.node_features
get_loss(gnn::GNNModel) = gnn.loss

AddResidual(l) = Parallel(+, Base.identity, l) # residual connection

abstract type ChainFactory end

"""
    GNNChainFactory

Can be used to create a `GNNChain` of a specific type, e.g. `ResGatedGraphConv` or `GATv2Conv` networks.

"""
abstract type GNNChainFactory <: ChainFactory end

"""
    (::GNNChainFactory)(d_in::Int, dims::Vector{Int}; add_classifier::Bool)

Creates a `GNNChain` of a specific type with a first layer of size `d_in` => `dims[1]`, 
second layer of size `dims[1]` => `dims[2]`, etc. If `add_classifier` is set to true, a Dense layer of size `dims[end] => 1` 
with a sigmoid classifier is added as a final layer. 

"""
(::GNNChainFactory)(d_in::Int, dims::Vector{Int}; add_classifier::Bool)::GNNChain = error("ModelFactory: Abstract Functor called")

struct ResGatedGraphConv_GNNChainFactory <: GNNChainFactory end

function (::ResGatedGraphConv_GNNChainFactory)(d_in::Int, dims::Vector{Int}; add_classifier::Bool = false)::GNNChain
    @assert length(dims) >= 1
    inner_layers_gcn = (AddResidual(ResGatedGraphConv(dims[i] => dims[i+1], relu)) for i in 1:(length(dims)-1))
    inner_layers_batch_norm = (BatchNorm(dims[i+1]) for i in 1:(length(dims)-1))
    inner_layers = collect(Iterators.flatten(zip(inner_layers_gcn, inner_layers_batch_norm)))
    model = GNNChain(
        ResGatedGraphConv(d_in => dims[1], relu),
        BatchNorm(dims[1]),
        inner_layers...,
    )

    if add_classifier
        model = GNNChain(
            model,
            Dense(dims[end] => 1, sigmoid),
        )
    end
    
    return model
end

struct GATv2Conv_GNNChainFactory <: GNNChainFactory end

function (::GATv2Conv_GNNChainFactory)(d_in::Int, dims::Vector{Int}; add_classifier::Bool = false)::GNNChain
    @assert length(dims) >= 1
    inner_layers_gcn = (AddResidual(GATv2Conv(dims[i] => dims[i+1])) for i in 1:(length(dims)-1))
    inner_layers_batch_norm = (BatchNorm(dims[i+1]) for i in 1:(length(dims)-1))
    inner_layers = collect(Iterators.flatten(zip(inner_layers_gcn, inner_layers_batch_norm)))
    
    model = GNNChain(
            Dense(d_in, dims[1]),
            BatchNorm(dims[1]), # ? is this needed for egonet feature???
            inner_layers...,
    )

    if add_classifier
        model = GNNChain(
            model,
            Dense(dims[end] => 1, sigmoid),
        )
    end

    return model
end

struct Dense_ChainFactory <: ChainFactory end

"""
    Dense_classifier(d_in, dims)

Simple Dense Feed Forward Network with specified dimensions and sigmoid classifier at the final layer. 
"""
function (::Dense_ChainFactory)(d_in::Int, dims::Vector{Int}; add_classifier::Bool = true)::Chain
    @assert length(dims) >= 1

    inner_dense = (Dense(dims[i] => dims[i+1], relu) for i in 1:(length(dims)-1))
    inner_batchnorm = (BatchNorm(dims[i+1]) for i in 1:(length(dims)-1))
    inner_layers = collect(Iterators.flatten(zip(inner_dense, inner_batchnorm)))

    model = Chain(
            Dense(d_in => dims[1]),
            BatchNorm(dims[1]),
            inner_layers...,
            Dense(dims[end] => 1, sigmoid)
    )
    return model
end

"""
    SimpleGNN

A graph neural network that classifies nodes in a single forward pass by applying multiple graph convolutional layers. 
Its corresponding ScoringFunction type `SimpleGNN_ScoringFunction` is only used for testing purposes as it is very slow. 
Use `Encoder_Decoder_GNNModel` and its corresponding ScoringFunction type instead. 

"""
struct SimpleGNN <: GNNModel
    num_layers::Int # number of layers
    d_in::Int # dimension of node feature vectors
    dims::Vector{Int} # output dimensions of GCN layers (dims[i] is output dim of layer i)
    model::GNNChain
    node_features::Vector{<:NodeFeature}
    gnn_type::String
    batch_support::Bool
    loss # loss function for training
    opt

    function SimpleGNN(dims::Vector{Int}; 
                       node_features::Vector{<:NodeFeature}=[DegreeNodeFeature(), d_S_NodeFeature()],
                       loss = Flux.logitbinarycrossentropy,
                       opt=Adam(0.001, (0.9, 0.999)), 
                       model_factory::GNNChainFactory = ResGatedGraphConv_GNNChainFactory(),
                       )
        d_in = sum(map(x -> length(x), node_features))
        model = model_factory(d_in, dims; add_classifier=true) |> device

        loss_func(g::GNNGraph) = loss( vec(model(g, g.ndata.x)), g.ndata.y)

        gnn_type = split(string(typeof(model_factory)), "_")[1]

        batch_support = true

        new(length(dims), d_in, dims, model, node_features, gnn_type, batch_support, loss_func, opt)
    end
end

Base.show(io::IO, ::MIME"text/plain", x::SimpleGNN) = print(io, "$(x.gnn_type)-$(x.d_in)-$(join(x.dims, "-"))")

"""
    Encoder_Decoder_GNNModel

A deep neural network based on the encoder / decoder paradigm. The encoder is a GNNChain which should be used to compute 
node embeddings for a graph. During each iteration, the context (based on the current candidate solution) is derived from 
node embeddings and fed through the simpler decoder Chain, which is e.g. a small Dense feed forward network. 

"""
struct Encoder_Decoder_GNNModel <: GNNModel
    d_in::Int
    encoder_dims::Vector{Int}
    decoder_dims::Vector{Int}
    encoder::GNNChain # more expensive gnn chain
    decoder::Chain    # linear time decoder 
    node_features::Vector{<:NodeFeature}
    gnn_type::String
    batch_support::Bool
    loss
    opt

    function Encoder_Decoder_GNNModel(encoder_dims::Vector{Int}, decoder_dims::Vector{Int};
                      encoder_factory::GNNChainFactory = ResGatedGraphConv_GNNChainFactory(),
                      decoder_factory::ChainFactory = Dense_ChainFactory(),  
                      node_features::Vector{<:NodeFeature} = [DegreeNodeFeature()],
                      loss = Flux.binarycrossentropy,
                      opt = Adam(0.001, (0.9, 0.999)),
                      batch_support = false
                      )

        # input dimension is sum of dimensions of node features
        d_in = sum(map(x -> length(x), node_features))

        # encoder: GNN, compute node embeddings
        encoder = encoder_factory(d_in, encoder_dims) |> device

        # decoder from node embeddings + context embedding, used to classify node
        decoder = decoder_factory(encoder_dims[end]*2, decoder_dims) |> device

        gnn_type = "$(split(string(typeof(encoder_factory)), "_")[1])-$(split(string(typeof(decoder_factory)), "_")[1])"

        function loss_func_unbatched(unbatched_g::GNNGraph, S::Vector{Int})
            node_embeddings = encoder(unbatched_g, unbatched_g.ndata.x)
            context = repeat(mean(NNlib.gather(node_embeddings, S), dims=2), 1, nv(unbatched_g))
            decoder_input = vcat(node_embeddings, context)
            output = decoder(decoder_input) 
            loss(vec(output), unbatched_g.ndata.y)
        end

        # This does not work for now, as `getgraph` is not gpu friendly at the moment: https://github.com/CarloLucibello/GraphNeuralNetworks.jl/issues/161
        function loss_func_batched(batched_g::GNNGraph) # graphs are batched by taking union of several graphs which are all disconnected          
            node_embeddings = encoder(batched_g, batched_g.ndata.x) # compute node embeddings for all graphs in g

            offset = 0
            context_embeddings = fill(0f0, size(node_embeddings)) # context embeddings have same size as node embeddings
            
            for i in 1:batched_g.num_graphs 
                # loop over graphs that are batched in g and compute context for each graph
                # context is the mean of all node embeddings of vertices in candidate solution S
                g = getgraph(batched_g, [i])
                
                # obtain column indices for features of vertices in S
                S = filter(v -> g.in_S[v], 1:nv(g))

                # gather mean from corresponding columns in embeddings and repeat for each node in g
                context = repeat(mean(gather(embeddings[:, (1+offset):(nv(g)+offset)], S), dims=2), 1, nv(g))
                
                # write back to context embedding matrix
                NNlib.scatter!(+, context_embeddings, context, collect((1+offset):(nv(g)+offset)))
                offset += nv(batched_g) # increase offset, as vertices are always numbered from 1
            end
            decoder_input = vcat(node_embeddings, context_embeddings)
            output = decoder(decoder_input) 
            loss(vec(output), batched_g.ndata.y) 
        end

        loss_func = batch_support ? loss_func_batched : loss_func_unbatched

        new(d_in, encoder_dims, decoder_dims, encoder, decoder, node_features, gnn_type, batch_support, loss_func, opt)
    end
end

Flux.params(gnn::Encoder_Decoder_GNNModel) = Flux.params(gnn.encoder, gnn.decoder)

Base.show(io::IO, ::MIME"text/plain", x::Encoder_Decoder_GNNModel) = 
    print(io, "$(x.gnn_type)-$(x.d_in)-$(join(x.encoder_dims, "-"))-$(join(x.decoder_dims, "-"))")

"""

Get context embedding of dimension d from node_embeddings, which is a d x n matrix 

"""
function get_context_embeddings(node_embeddings, in_S::Vector{Int})
    repeat(d_S, 1)
    mean(embeddings[:, S], dims=2)
end


Flux.params(gnn::SimpleGNN) = Flux.params(gnn.model)

(gnn::SimpleGNN)(gnn_graph::GNNGraph, inputs::AbstractMatrix) = gnn.model(gnn_graph, inputs)

"""
    (::NodeFeature)(graph::SimpleGraph, S::Union{Vector{Int}, Set{Int}, Nothing} = nothing)

Compute some node feature of a `graph` and optional candidate solution `S` and return it as a vector 
of length of `vertices(graph)`. 

"""
(::NodeFeature)(graph::SimpleGraph, S = nothing, d_S = nothing)::Vector{Float32} = error("NodeFeature: Abstract functor called")

"""
    Base.length(::NodeFeature)

Returns the dimension of the feature vector for a single node for this `NodeFeature`

"""
Base.length(::NodeFeature) = error("NodeFeature: Abstract length called")

struct DegreeNodeFeature <: NodeFeature end

(::DegreeNodeFeature)(graph::SimpleGraph, S = nothing, d_S = nothing) = Float32.(degree(graph))'

Base.length(::DegreeNodeFeature) = 1

struct d_S_NodeFeature <: NodeFeature end

(::d_S_NodeFeature)(graph::SimpleGraph, S = nothing, d_S = nothing) = Float32.(d_S)'

Base.length(::d_S_NodeFeature) = 1

struct DeepWalkNodeFeature <: NodeFeature
    rws::RandomWalkSimulator
    walks_per_node::Int
    embedding_size::Int

    function DeepWalkNodeFeature(; rws=RandomWalkSimulator(50, 1), walks_per_node=100, embedding_size=64)
        new(rws, walks_per_node, embedding_size)
    end
end

function (dnf::DeepWalkNodeFeature)(graph::SimpleGraph, S = nothing, d_S = nothing)
    learn_embeddings(dnf.rws, graph; dnf.walks_per_node)
end

Base.length(x::DeepWalkNodeFeature) = x.embedding_size

# EgoNet features of the `d`-hop neighborhood of a vertex (all vertices reachable from a node by paths of length <= d)
# Features: 
# - Size of egonet (nodes, edges)
# - Number of edges to outside
struct EgoNetNodeFeature <: NodeFeature
    d::Int

    function EgoNetNodeFeature(d::Int = 1)
        new(d)
    end
end

function (enf::EgoNetNodeFeature)(graph::SimpleGraph, S = nothing, d_S = nothing)
    features = []
    for v in vertices(graph)
        N_v = neighborhood(graph, v, enf.d)
        egonet, _ = induced_subgraph(graph, N_v )
        
        num_v = nv(egonet)
        num_e = ne(egonet)

        outgoing_edges = 0

        for e in edges(graph)
            if (src(e) ∈ N_v && dst(e) ∉ N_v) || (dst(e) ∈ N_v && src(e) ∉ N_v)
                outgoing_edges += 1
            end
        end
        push!(features, [num_v, num_e, outgoing_edges])
    end
    feature_matrix = reduce(hcat, features)
    return Float32.(feature_matrix)
end

Base.length(::EgoNetNodeFeature) = 3

struct PageRankNodeFeature <: NodeFeature end

Base.length(::PageRankNodeFeature) = 1

function (::PageRankNodeFeature)(graph::SimpleGraph, S = nothing, d_S = nothing) 
    Float32.(pagerank(graph)')
end

function Base.convert(::Type{Vector{<:NodeFeature}}, a::Vector{Any})
    res::Vector{<:NodeFeature} = [_ for _ in a]
end

function compute_node_features(feature_list::Vector{<:NodeFeature}, graph, S, d_S)
    features = [node_feature(graph, S, d_S) for node_feature in feature_list]
    vcat(features...)
end

# struct EgoNetNodeFeature <: NodeFeature end




# function create_sample(graph::SimpleGraph{Int},
#                        S::Union{Set{Int}, Vector{Int}},
#                        lookahead_func::LookaheadSearchFunction
#                        )::GNNGraph
#     # node features
#     degrees = degree(graph)
#     d_S = thesis.LocalSearch.calculate_d_S(graph, S)
#     node_features = Float32.(vcat(degrees', d_S'))

#     # use lookahead function to obtain best neighboring solutions
#     obj_val, solutions = lookahead_func(graph, S, d_S)
#     targets = fill(0.0f0, nv(graph))

#     # compute target node labels
#     for v in S
#         targets[v] = 1.0
#     end

#     for (in_nodes, out_nodes) in solutions
#         # nodes that are in S and not in every best neighboring solution are scored lower
#         # so they are considered for swaps
#         for u in in_nodes
#             targets[u] = 0.0
#         end
#         # mark nodes that are attractive for swaps with 1
#         for v in out_nodes
#             targets[v] = 1.0
#         end
#     end

#     # create GNNGraph
#     gnn_graph = GNNGraph(graph,
#                          ndata=(; x = node_features, y = targets)
#                          )
#     add_self_loops(gnn_graph)
#     return gnn_graph
# end

# function create_dataset(density_str::String, lookahead_func::LookaheadSearchFunction)
#     println("Create dataset for $density_str with $(string(lookahead_func))")
#     gnn = ResGatedGraphConvGNN(2, [32, 32, 32])
#     training_samples = []
#     graph_to_S = Dict()
#     for i = 1:100
#         println("Create training samples for graph $i")
#         BSON.@load "training_data/$density_str/graph_$(lpad(string(i), 3, '0'))_samples.bson" data
#         for sample in data.samples
#             gnn_graph = create_sample(data.graph, sample, lookahead_func)
#             push!(training_samples, (gnn_graph, data.graph, sample))
#         end
#     end

#     return training_samples
# end

# function train_gnn(training_data, num_epochs, it)
#     gnn = ResGatedGraphConvGNN(2, [32, 32, 32])

#     training_data = MLUtils.shuffleobs(training_data)
#     train_graphs, test_graphs = MLUtils.splitobs(training_data, at=0.8)
#     train_graphs = map(x -> x[1], train_graphs)
#     train_loader = Flux.DataLoader(train_graphs, batchsize=32, shuffle=true, collate=true)
#     # test_loader = Flux.DataLoader(test_graphs, batchsize=32, shuffle=false, collate=true)
#     loss(g::GNNGraph) = Flux.logitbinarycrossentropy( vec(gnn(g, g.ndata.x)), g.ndata.y) # logitbinarycrossentropy
#     # loss(g::GNNGraph) = Flux.mse( vec(gnn(g, g.ndata.x)), g.ndata.y)
#     loss(loader) = mean(loss(g |> device) for g in loader)

#     ps = Flux.params(gnn.model)
#     loss_before = 0
#     loss_after = 0
#     for epoch in 1:num_epochs
#         epoch_start = time()
#         for g in first(train_loader, 4)
#             g = g |> device
#             loss_before = loss(g)
#             gs = gradient(ps) do 
#                 loss(g)
#             end
#             Flux.Optimise.update!(gnn.opt, ps, gs)
#             loss_after = loss(g)
#         end

#         @info (; it, epoch,  time=time() - epoch_start, loss_before, loss_after)
#     end
#     return gnn, test_graphs
# end

# function evaluate_gnn(gnn, test_graphs, lookahead_func, logger)
#     correct_greedy = 0
#     correct_neighborhood = 0
#     local_optima = 0
#     for (gnn_graph, graph, S) in test_graphs
#         gnn_graph = gnn_graph |> device
#         scores = vec(gnn(gnn_graph, gnn_graph.ndata.x)) |> Flux.cpu
#         S = collect(S)
#         V_S = filter(v -> v ∉ S, vertices(graph))

#         scores_S = [scores[i] for i in S]
#         scores_V_S = [scores[i] for i in V_S]

#         v_greedy = V_S[argmax(scores_V_S)]
#         u_greedy = S[argmin(scores_S)]

#         d_S = thesis.LocalSearch.calculate_d_S(graph, S)

#         S′ = Set(S)
#         if d_S[v_greedy] - d_S[u_greedy] - Int(has_edge(graph, u_greedy, v_greedy)) > 0
#             push!(S′, v_greedy)
#             delete!(S′, u_greedy)
#         else
#             u_greedy = 0
#             v_greedy = 0
#         end
        
#         obj_gnn = ne(induced_subgraph(graph, collect(S′))[1])
#         with_logger(logger) do 
#             @info("GNN predicts swap ($u_greedy,$v_greedy) with objective value $(obj_gnn) (greedily)")
#         end
        

#         best_S = [S[i] for i in partialsortperm(scores_S, 1:min(10, length(S)))]
#         best_V_S = [V_S[i] for i in partialsortperm(scores_V_S, 1:min(10, length(V_S)); rev=true)]

#         Δuv_best, u_best, v_best = 0, 0, 0
#         for u in best_S, v in best_V_S
#             Δuv = d_S[v] - d_S[u] - Int(has_edge(graph, u, v))
#             if Δuv > Δuv_best
#                 Δuv_best, u_best, v_best = Δuv, u, v
#             end
#         end


#         S′ = Set(S)

#         if Δuv_best > 0
#             push!(S′, v_best)
#             delete!(S′, u_best)
#         end

#         obj_gnn_neighborhood = ne(induced_subgraph(graph, collect(S′))[1])
        
#         with_logger(logger) do 
#             @info("GNN predicts swap ($u_best,$v_best) with objective value $(obj_gnn_neighborhood) (restricted neighborhood)")
#         end

#         obj_val, solutions = lookahead_func(graph, Set(S))
#         if isempty(solutions)
#             local_optima += 1
#         end
#         with_logger(logger) do 
#             @info("Actual Solutions: $(solutions), objective value $(obj_val)")
#         end
#         if obj_val == obj_gnn
#             correct_greedy += 1
#         end
#         if obj_val == obj_gnn_neighborhood
#             correct_neighborhood += 1
#         end
#     end
#     acc_greedy = correct_greedy / length(test_graphs)
#     acc_greedy_adjusted = (correct_greedy-local_optima) / (length(test_graphs)-local_optima)
#     acc_restricted = correct_neighborhood / length(test_graphs)
#     acc_restricted_adjusted = (correct_neighborhood-local_optima) / (length(test_graphs)-local_optima)

#     with_logger(logger) do
#         @info("GNN picked one of the best swaps $correct_greedy times (greedily), accuracy: $(acc_greedy)")
#         @info("GNN picked one of the best swaps $correct_neighborhood times (restricted neighborhood)), accuracy: $(acc_restricted)")
#         @info("Local Optima with respect to Ω_1 neighborhood: $local_optima")
#         @info("GNN picked one of the best swaps $(correct_greedy-local_optima) times (greedily)"*
#             " $(acc_greedy_adjusted)")
#         @info("GNN picked one of the best swaps $(correct_neighborhood-local_optima) times (restricted neighborhood))"*
#             " $(acc_restricted_adjusted)")
#     end
#     return (;acc_greedy, acc_greedy_adjusted, acc_restricted, acc_restricted_adjusted)
# end

# function evaluation(training_data, iterations, num_epochs, to_file=false)
#     io = to_file ? open("logfile.txt", "w+") : Base.stdout
#     logger = SimpleLogger(io)
#     results = []
#     for i = 1:iterations
#         gnn, test_graphs = train_gnn(training_data, num_epochs, i)
#         push!(results, evaluate_gnn(gnn, test_graphs, Ω_1_LookaheadSearchFunction(), logger))
#         flush(io)
#     end
#     to_file && close(io)
#     acc_greedy = mean(elem.acc_greedy for elem in results)
#     acc_greedy_adjusted = mean(elem.acc_greedy_adjusted for elem in results)
#     acc_restricted = mean(elem.acc_restricted for elem in results)
#     acc_restricted_adjusted = mean(elem.acc_restricted_adjusted for elem in results)
#     return (;acc_greedy, acc_greedy_adjusted, acc_restricted, acc_restricted_adjusted)
# end

# training_data = create_dataset("high_density", Ω_1_LookaheadSearchFunction())
# evaluation(training_data, 20, 200)

# BSON.@save "gnn_medium_density_Ω_1_Lookahead.bson" gnn

end # module