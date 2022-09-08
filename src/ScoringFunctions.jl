
using thesis.GNNs: GNNModel, compute_node_features, device
using Flux: gpu, cpu
using GraphNeuralNetworks

"""
    ScoringFunction

Abstract type for scoring functions used in the local search algorithm in order to easily change scoring functions. 
It provides a simple interface: 
1. At the start of the local search procedure for candidate solution `S`, `update!(sf, graph, S)` must be called.
2. Every time a pair of vertices `u,v` is swapped during local search, `update!(sf, u, v)` must be called.
3. `get_restricted_neighborhood(sf, S, V_S)` returns the sets `X ⊆ S`, `Y ⊆ V∖S`, which is the restricted neighborhood 
    according to the current scores of the scoring function. 

"""
abstract type ScoringFunction end

update!(sf::ScoringFunction, S::Union{Vector{Int}, Set{Int}}) = error("ScoringFunction: Abstract update!(sf, graph, S) called")
update!(sf::ScoringFunction, u::Int, v::Int) = error("ScoringFunction: Abstract update!(sf, u, v) called")
get_restricted_neighborhood(sf::ScoringFunction, S::Union{Vector{Int}, Set{Int}}, 
                            V_S::Union{Vector{Int}, Set{Int}}
                            )::@NamedTuple{X::Union{Vector{Int}, Set{Int}}, Y::Union{Vector{Int}, Set{Int}}} = 
    error("ScoringFunction: Abstract get_restricted_neighborhood(sf) called")

mutable struct d_S_ScoringFunction <: ScoringFunction
    graph::Union{Nothing, SimpleGraph}
    d_S::Vector{Int}

    function d_S_ScoringFunction()
        new(nothing, [])
    end
end

function update!(sf::d_S_ScoringFunction, graph::SimpleGraph, S::Union{Vector{Int}, Set{Int}})::Vector{Int}
    sf.graph = graph
    sf.d_S = fill(0, nv(graph))
    sf.d_S = calculate_d_S(sf.graph, S)
    return sf.d_S
end

function update!(sf::d_S_ScoringFunction, u::Int, v::Int)
    for w in neighbors(sf.graph, u)
        sf.d_S[w] -= 1
    end
    for w in neighbors(sf.graph, v)
        sf.d_S[w] += 1
    end
    return sf.d_S
end

function get_restricted_neighborhood(sf::d_S_ScoringFunction, S::Union{Vector{Int}, Set{Int}}, 
                                     V_S::Union{Vector{Int}, Set{Int}}
                                     )::@NamedTuple{X::Union{Vector{Int}, Set{Int}}, Y::Union{Vector{Int}, Set{Int}}}
    d_S = sf.d_S
    d_min = minimum([d_S[i] for i in S])
    d_max = maximum([d_S[i] for i in V_S])
    X = filter(u -> (d_S[u] <= d_min+1), S)
    Y = filter(v -> (d_S[v] >= d_max-1), V_S)
    return (;X, Y)
end

mutable struct GNN_ScoringFunction <: ScoringFunction
    graph::Union{Nothing, SimpleGraph}
    gnn_graph::Union{Nothing, GNNGraph}
    gnn::GNNModel
    d_S::Vector{Int}
    scores::Vector{Float32}
    neighborhood_size::Int

    function GNN_ScoringFunction(gnn, neighborhood_size)
        new(nothing, nothing, gnn, [], [], neighborhood_size)
    end
end

function update!(sf::GNN_ScoringFunction, graph::SimpleGraph, S::Union{Vector{Int}, Set{Int}})
    if graph != sf.graph
        sf.graph = graph
        sf.gnn_graph = GNNGraph(sf.graph) |> device
    end
    sf.d_S = calculate_d_S(sf.graph, S)
    node_features = compute_node_features(sf.graph, sf.d_S)
    sf.scores = vec(sf.gnn(sf.gnn_graph, node_features |> device) |> cpu)
    return sf.scores
end

function update!(sf::GNN_ScoringFunction, u::Int, v::Int)
    for w in neighbors(sf.graph, u)
        sf.d_S[w] -= 1
    end
    for w in neighbors(sf.graph, v)
        sf.d_S[w] += 1
    end
    node_features = compute_node_features(sf.graph, sf.d_S)
    sf.scores = vec(sf.gnn(sf.gnn_graph, node_features |> device) |> cpu)
    return sf.scores
end

function get_restricted_neighborhood(sf::GNN_ScoringFunction, S::Union{Vector{Int}, Set{Int}}, 
                                     V_S::Union{Vector{Int}, Set{Int}}
                                     )::@NamedTuple{X::Union{Vector{Int}, Set{Int}}, Y::Union{Vector{Int}, Set{Int}}}
    if typeof(S) <: Set # need fixed order of elements
        S = collect(S)
        V_S = collect(V_S)
    end
    scores_S = [sf.scores[i] for i in S]
    scores_V_S = [sf.scores[i] for i in V_S]
    X = S[partialsortperm(scores_S, 1:min(sf.neighborhood_size, length(S)))]
    Y = V_S[partialsortperm(scores_V_S, 1:min(sf.neighborhood_size, length(V_S)); rev=true)]

    return (;X, Y)
end
