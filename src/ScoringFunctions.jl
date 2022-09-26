
using thesis.GNNs: GNNModel, SimpleGNN, Encoder_Decoder_GNNModel, compute_node_features, device, get_feature_list
using Flux: gpu, cpu, NNlib.gather
using GraphNeuralNetworks
using DataStructures: PriorityQueue, enqueue!, dequeue!
using StatsBase
using Statistics

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

update!(sf::ScoringFunction, graph::SimpleGraph, S::Set{Int}) = error("ScoringFunction: Abstract update!(sf, graph, S) called")
update!(sf::ScoringFunction, u::Int, v::Int) = error("ScoringFunction: Abstract update!(sf, u, v) called")
get_restricted_neighborhood(sf::ScoringFunction, S::Set{Int}, V_S::Set{Int}
                           )::@NamedTuple{X::Union{Vector{Int}, Set{Int}}, Y::Union{Vector{Int}, Set{Int}}} = 
    error("ScoringFunction: Abstract get_restricted_neighborhood(sf) called")

mutable struct d_S_ScoringFunction <: ScoringFunction
    graph::Union{Nothing, SimpleGraph}
    d_S::Vector{Int}

    function d_S_ScoringFunction()
        new(nothing, [])
    end
end

function update!(sf::d_S_ScoringFunction, graph::SimpleGraph, S::Set{Int})::Vector{Int}
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

function get_restricted_neighborhood(sf::d_S_ScoringFunction, S::Set{Int}, V_S::Set{Int}
                                     )::@NamedTuple{X::Union{Vector{Int}, Set{Int}}, Y::Union{Vector{Int}, Set{Int}}}
    d_S = sf.d_S
    d_min = minimum([d_S[i] for i in S])
    d_max = maximum([d_S[i] for i in V_S])
    X = filter(u -> (d_S[u] <= d_min+1), S)
    Y = filter(v -> (d_S[v] >= d_max-1), V_S)
    return (;X, Y)
end

mutable struct Random_ScoringFunction <: ScoringFunction 
    neighborhood_size::Int
    d_S_sf::d_S_ScoringFunction
    d_S::Vector{Int}

    function Random_ScoringFunction(neighborhood_size::Int)
        new(neighborhood_size, d_S_ScoringFunction(), [])
    end
end

function update!(sf::Random_ScoringFunction, graph::SimpleGraph, S::Set{Int}) 
    d_S = update!(sf.d_S_sf, graph, S)
    sf.d_S = d_S
    return sf.d_S
end

function update!(sf::Random_ScoringFunction, u::Int, v::Int) 
    d_S = update!(sf.d_S_sf, u, v)
    sf.d_S = d_S
    return sf.d_S
end

function get_restricted_neighborhood(sf::Random_ScoringFunction, S::Set{Int}, V_S::Set{Int}
                                     )::@NamedTuple{X::Union{Vector{Int}, Set{Int}}, Y::Union{Vector{Int}, Set{Int}}}

    if sf.neighborhood_size < length(S)
        # X = S[partialsortperm(scores_S, 1:sf.neighborhood_size)]
        X = sample(collect(S), sf.neighborhood_size; replace=false)
    else
        X = S
    end

    if sf.neighborhood_size < length(V_S)
        Y = sample(collect(V_S), sf.neighborhood_size; replace=false)
    else
        Y = V_S
    end

    return (; X, Y)
end

abstract type GNN_ScoringFunction <: ScoringFunction end

mutable struct SimpleGNN_ScoringFunction <: GNN_ScoringFunction
    graph::Union{Nothing, SimpleGraph}
    gnn_graph::Union{Nothing, GNNGraph}
    gnn::GNNModel
    d_S::Vector{Int}
    scores::Vector{Float32}
    neighborhood_size::Int

    function SimpleGNN_ScoringFunction(gnn::GNNModel, neighborhood_size::Int)
        new(nothing, nothing, gnn, [], [], neighborhood_size)
    end
end

function update!(sf::SimpleGNN_ScoringFunction, graph::SimpleGraph, S::Set{Int})
    if graph !== sf.graph
        sf.graph = graph
        sf.gnn_graph = GNNGraph(sf.graph) |> device
    end
    sf.d_S = calculate_d_S(sf.graph, S)
    node_features = compute_node_features(get_feature_list(sf.gnn), sf.graph, S, sf.d_S)
    sf.scores = vec(sf.gnn(sf.gnn_graph, node_features |> device) |> cpu)
    return sf.scores
end

function update!(sf::SimpleGNN_ScoringFunction, u::Int, v::Int)
    for w in neighbors(sf.graph, u)
        sf.d_S[w] -= 1
    end
    for w in neighbors(sf.graph, v)
        sf.d_S[w] += 1
    end
    node_features = compute_node_features(get_feature_list(sf.gnn), sf.graph, nothing, sf.d_S)
    sf.scores = vec(sf.gnn(sf.gnn_graph, node_features |> device) |> cpu)
    return sf.scores
end

mutable struct Encoder_Decoder_ScoringFunction <: GNN_ScoringFunction
    graph::Union{Nothing, SimpleGraph}
    gnn_graph::Union{Nothing, GNNGraph}
    gnn::Encoder_Decoder_GNNModel
    embeddings::Union{Nothing, AbstractMatrix{Float32}}
    d_S::Vector{Int}
    S::Set{Int}
    scores::Vector{Float32}
    neighborhood_size::Int

    function Encoder_Decoder_ScoringFunction(gnn::Encoder_Decoder_GNNModel, neighborhood_size::Int)
        new(nothing, nothing, gnn, nothing, [], Set(), [], neighborhood_size)
    end
end

# apply encoder each time the graph changes, otherwise just compute the context from the embedding
function update!(sf::Encoder_Decoder_ScoringFunction, graph::SimpleGraph, S::Set{Int})
    sf.d_S = calculate_d_S(graph, S)
    sf.S = copy(S)

    if graph !== sf.graph
        sf.graph = graph
        node_features = compute_node_features(get_feature_list(sf.gnn), sf.graph, S, sf.d_S)
        sf.gnn_graph = GNNGraph(sf.graph, ndata=(;x=node_features)) |> device
        # compute node embeddings
        sf.embeddings = sf.gnn.encoder(sf.gnn_graph, sf.gnn_graph.ndata.x)
    end
    
    # compute context from node embeddings: context is the index wise mean of all node embeddings of nodes in S
    embeddings_with_context = vcat(sf.embeddings, compute_context(sf.embeddings, sf.S, nv(graph)))

    # compute scores by applying decoder on embeddings + context
    sf.scores = vec(sf.gnn.decoder(embeddings_with_context) |> cpu)

    return sf.scores
end

function compute_context(embeddings::AbstractMatrix{Float32}, S::Set{Int}, n::Int; offset=0)
    S = [v+offset for v in S] # if embedding is of batched graph, add offset to vertex number for correct matrix columns
    repeat(mean(gather(embeddings, S), dims=2), 1, n)
end

# only apply decoder, compute context from node embeddings
function update!(sf::Encoder_Decoder_ScoringFunction, u::Int, v::Int)
    for w in neighbors(sf.graph, u)
        sf.d_S[w] -= 1
    end
    for w in neighbors(sf.graph, v)
        sf.d_S[w] += 1
    end
    delete!(sf.S, u)
    push!(sf.S, v)

    embeddings_with_context = vcat(sf.embeddings, compute_context(sf.embeddings, sf.S, nv(sf.graph)))

    # compute scores by applying decoder on embeddings + context
    sf.scores = vec(sf.gnn.decoder(embeddings_with_context) |> cpu)
    
    return sf.scores
end

# this method is the same for SimpleGNN_ScoringFunction and Encoder_Decoder_GNN_ScoringFunction
function get_restricted_neighborhood(sf::GNN_ScoringFunction, S::Set{Int}, V_S::Set{Int}
                                     )::@NamedTuple{X::Union{Vector{Int}, Set{Int}}, Y::Union{Vector{Int}, Set{Int}}}
    if typeof(S) <: Set # need fixed order of elements
        S = collect(S)
        V_S = collect(V_S)
    end

    scores_S = [sf.scores[i] for i in S]
    scores_V_S = [sf.scores[i] for i in V_S]

    if sf.neighborhood_size < length(S)
        X = get_k_order_statistic(S, scores_S, sf.neighborhood_size; order=Base.Order.ReverseOrdering())
    else
        X = S
    end

    if sf.neighborhood_size < length(V_S)
        Y = get_k_order_statistic(V_S, scores_V_S, sf.neighborhood_size)
    else
        Y = V_S
    end

    return (;X, Y)
end


"""
    get k maximum (ForwardOrdering) / minimum (ReverseOrdering) valued keys from (key, value) pairs in zip(keys, values) 
"""
function get_k_order_statistic(keys::Vector{K}, values::Vector{V}, k::Int; order=Base.Order.ForwardOrdering()) where {K, V}
    pq = PriorityQueue{K, V}(order)
    compare = (a::V, b::V) -> (Base.lt(Base.Order.ReverseOrdering(order), a, b))
    for i=1:k
        enqueue!(pq, keys[i], values[i])
    end

    for i=(k+1):length(keys)
        key, val = peek(pq)
        if compare(values[i], val)
            dequeue!(pq)
            enqueue!(pq, keys[i], values[i])
        end
    end

    result = K[]
    for i=1:k
        key, val = peek(pq)
        dequeue!(pq)
        push!(result, key)
    end
    result
end
