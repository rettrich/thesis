module GNNs

using Flux, Graphs, GraphNeuralNetworks, CUDA
# using BSON
# using thesis
# using Statistics
# using Printf
# using MLUtils
# using Logging

export GNNModel, ResGatedGraphConvGNN, compute_node_features, device,
    NodeFeature, d_S_NodeFeature, DegreeNodeFeature, get_feature_list

device = CUDA.functional() ? Flux.gpu : Flux.cpu
# device = Flux.cpu

ENV["JULIA_DEBUG"] = "thesis"


abstract type GNNModel end

Flux.params(gnn::GNNModel) = Flux.params(gnn.model)

(gnn::GNNModel)(gnn_graph::GNNGraph, inputs::AbstractMatrix) = gnn.model(gnn_graph, inputs)

abstract type NodeFeature end

"""
    (::NodeFeature)(graph::SimpleGraph, S::Union{Vector{Int}, Set{Int}, Nothing} = nothing)

Compute some node feature of a `graph` and optional candidate solution `S` and return it as a vector 
of length of `vertices(graph)`. 

"""
(::NodeFeature)(graph::SimpleGraph, S = nothing, d_S = nothing)::Vector{Float32} = error("NodeFeature: Abstract functor called")

get_feature_list(gnn::GNNModel) = gnn.node_features

AddResidual(l) = Parallel(+, Base.identity, l) # residual connection

struct ResGatedGraphConvGNN <: GNNModel
    num_layers::Int # number of layers
    d_in::Int # dimension of node feature vectors
    dims::Vector{Int} # output dimensions of GCN layers (dims[i] is output dim of layer i)
    model::GNNChain
    opt
    node_features::Vector{<:NodeFeature}

    function ResGatedGraphConvGNN(d_in::Int, dims::Vector{Int}; 
                                  opt=Adam(0.001, (0.9, 0.999)), 
                                  node_features::Vector{<:NodeFeature}=[DegreeNodeFeature(), d_S_NodeFeature()]
                                  )
        @assert length(dims) >= 1
        inner_layers_gcn = (AddResidual(ResGatedGraphConv(dims[i] => dims[i+1], relu)) for i in 1:(length(dims)-1))
        inner_layers_batch_norm = (BatchNorm(dims[i+1]) for i in 1:(length(dims)-1))
        inner_layers = collect(Iterators.flatten(zip(inner_layers_gcn, inner_layers_batch_norm)))

        model = GNNChain(
            ResGatedGraphConv(d_in => dims[1], relu),
            BatchNorm(dims[1]),
            inner_layers...,
            Dense(dims[end] => 1, sigmoid),
        ) |> device
        new(length(dims), d_in, dims, model, opt, node_features)
    end
end

Flux.params(gnn::ResGatedGraphConvGNN) = Flux.params(gnn.model)

(gnn::ResGatedGraphConvGNN)(gnn_graph::GNNGraph, inputs::AbstractMatrix) = gnn.model(gnn_graph, inputs)

function compute_node_features(graph::SimpleGraph, d_S::Vector{Int})
    degrees = degree(graph)
    node_features = Float32.(vcat(degrees', d_S'))
    return node_features
end

function compute_node_features(feature_list::Vector{<:NodeFeature}, graph, S, d_S)
    features = [node_feature(graph, S, d_S)' for node_feature in feature_list]
    vcat(features...)
end

struct DegreeNodeFeature <: NodeFeature end

(::DegreeNodeFeature)(graph::SimpleGraph, S = nothing, d_S = nothing) = Float32.(degree(graph))

struct d_S_NodeFeature <: NodeFeature end

(::d_S_NodeFeature)(graph::SimpleGraph, S = nothing, d_S = nothing) = Float32.(d_S)


# struct EgoNetNodeFeature <: NodeFeature end

# struct PageRankNodeFeature <: NodeFeature end

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