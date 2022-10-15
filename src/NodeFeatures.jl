abstract type NodeFeature end

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

(::DegreeNodeFeature)(graph::SimpleGraph, S = nothing, d_S = nothing) = Float32.(degree(graph)./nv(graph))'

Base.length(::DegreeNodeFeature) = 1

struct d_S_NodeFeature <: NodeFeature 
    add_mean::Bool
    add_std::Bool

    function d_S_NodeFeature(; add_mean=true, add_std=true)
        new(add_mean, add_std)
    end
end

function (d_S_nf::d_S_NodeFeature)(graph::SimpleGraph, S = nothing, d_S = nothing)  
    features = d_S'
    if d_S_nf.add_mean
        features = vcat(features, repeat([mean(d_S)], 1, length(d_S)))
    end
    if d_S_nf.add_std
        features = vcat(features, repeat([std(d_S)], 1, length(d_S)))
    end
    Float32.(features)
end

Base.length(d_S_nf::d_S_NodeFeature) = 1 + Int(d_S_nf.add_mean) + Int(d_S_nf.add_std)

abstract type RepresentationLearningNodeFeature <: NodeFeature end

struct DeepWalkNodeFeature <: RepresentationLearningNodeFeature
    rws::RandomWalkSimulator
    walks_per_node::Int
    embedding_size::Int

    function DeepWalkNodeFeature(; walk_length=50, window_size=3, walks_per_node=30, embedding_size=64)
        rws = RandomWalkSimulator(walk_length, window_size)
        new(rws, walks_per_node, embedding_size)
    end
end

struct Node2VecNodeFeature <: RepresentationLearningNodeFeature
    rws::SecondOrderRandomWalkSimulator
    walks_per_node::Int
    embedding_size::Int
    
    function Node2VecNodeFeature(p=1, q=1; walk_length=50, window_size=3, walks_per_node=30, embedding_size=64)
        rws = SecondOrderRandomWalkSimulator(walk_length, window_size, p, q)
        new(rws, walks_per_node, embedding_size)
    end
end

function (nf::RepresentationLearningNodeFeature)(graph::SimpleGraph, S = nothing, d_S = nothing)
    learn_embeddings(nf.rws, graph; nf.embedding_size, nf.walks_per_node)
end

Base.length(x::RepresentationLearningNodeFeature) = x.embedding_size

# EgoNet features of the `d`-hop neighborhood of a vertex (all vertices reachable from a node by paths of length <= d)
# Features: 
# - Size of egonet (nodes, edges)
# - Number of edges to outside
struct EgoNetNodeFeature <: NodeFeature
    d::Int
    normalize::Bool

    function EgoNetNodeFeature(d::Int = 1; normalize=true)
        new(d, normalize)
    end
end

function (enf::EgoNetNodeFeature)(graph::SimpleGraph, S = nothing, d_S = nothing)
    features = []
    n = nv(graph)
    m = ne(graph)

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

        if enf.normalize
            num_v /= n
            num_e /= m
            outgoing_edges /= m
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