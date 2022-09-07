
using thesis.GNNs
using GraphNeuralNetworks

"""
    ScoringFunction

Abstract type for scoring functions used in the local search algorithm in order to easily change scoring functions. 
It provides a simple interface: 
1. Every time a new graph is being operated on, `update!(sf, graph)` must be called. 
2. Every time a new candidate solution is constructed by a construction heuristic, `update!(sf, S)` must be called.
3. Every time a pair of vertices `u,v` is swapped during local search, `update!(sf, u, v)` must be called.
4. `get_restricted_neighborhood(sf, S, V_S)` returns the sets `X ⊆ S`, `Y ⊆ V∖S`, which is the restricted neighborhood 
    according to the current scores of the scoring function. 

"""
abstract type ScoringFunction end

update!(sf::ScoringFunction, graph::SimpleGraph) = error("update!(sf, graph) called on abstract ScoringFunction")
update!(sf::ScoringFunction, S::Set{Int}) = error("update!(sf, S) called on abstract ScoringFunction")
update!(sf::ScoringFunction, u::Int, v::Int) = error("update!(sf, u, v) called on abstract ScoringFunction")
get_restricted_neighborhood(sf::ScoringFunction, S::Set{Int}, V_S::Set{Int})::@NamedTuple{X::Vector{Int}, Y::Vector{Int}} = 
    error("get_restricted_neighborhood(sf) called on abstract ScoringFunction")

struct d_S_ScoringFunction <: ScoringFunction
    graph::SimpleGraph
    d_S::Vector{Int}
end

function update!(sf::d_S_ScoringFunction, graph::SimpleGraph)
    sf.graph = graph
    sf.d_S = fill(0, nv(graph))
    return nothing
end

function update!(sf::d_S_ScoringFunction, S::Set{Int})::Vector{Int}
    sf.d_S = thesis.LocalSearch.calculate_d_S(sf.graph, S)
    return sf.d_S
end

function update!(sf::d_S_ScoringFunction, u::Int, v::Int)
    for w in neighbors(sf.graph, u)
        sf.d_S[w] -= 1
    end
    for v in neighbors(sf.graph, v)
        sf.d_S[w] += 1
    end
    return sf.d_S
end

function get_restricted_neighborhood(sf::d_S_ScoringFunction, S::Set{Int}, V_S::Set{Int})
    d_S = sf.d_S
    d_min = minimum([d_S[i] for i in S])
    d_max = maximum([d_S[i] for i in V_S])
    X = filter(u -> (d_S[u] <= d_min+1), S)
    Y = filter(v -> (d_S[v] >= d_max-1), V_S)
    return X, Y
end

struct GNN_ScoringFunction <: ScoringFunction
    graph::SimpleGraph
    gnn_graph::GNNGraph
    gnn::GNNModel
    d_S::Vector{Int}
    scores::Vector{Float32}
    neighborhood_size::Int
end

function update!(sf::GNN_ScoringFunction, graph::SimpleGraph)
    sf.graph = graph
end

function update!(sf::GNN_ScoringFunction, S::Set{Int})
    sf.gnn_graph = GNNGraph(sf.graph)
    
    sf.d_S = thesis.calculate_d_S(sf.graph, S)
    node_features = thesis.Training.compute_node_features(sf.graph, S, d_S)
    sf.scores = sf.gnn(gnn_graph, node_features)
end

function update!(sf::GNN_ScoringFunction, u::Int, v::Int)
    for w in neighbors(sf.graph, u)
        sf.d_S[w] -= 1
    end
    for v in neighbors(sf.graph, v)
        sf.d_S[w] += 1
    end
    node_features = thesis.Training.compute_node_features(sf.graph, S, d_S)
    sf.scores = sf.gnn(gnn_graph, node_features)
end

function get_restricted_neighborhood(sf::GNN_ScoringFunction, S::Set{Int}, V_S::Set{Int})
    S = collect(S)
    V_S = collect(V_S)
    scores_S = [sf.scores[i] for i in S]
    scores_V_S = [sf.scores[i] for i in V_S]
    X = S[partialsortperm(scores_S, 1:min(sf.neighborhood_size, length(S)))]
    Y = V_S[partialsortperm(scores_V_S, 1:min(sf.neighborhood_size, length(V_S)); rev=true)]

    return X, Y
end
