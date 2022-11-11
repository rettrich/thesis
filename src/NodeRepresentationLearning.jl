module NodeRepresentationLearning

using StatsBase
using Graphs, SimpleWeightedGraphs
using Flux
using Random
using DynamicAxisWarping, Distances, DelimitedFiles
using Word2Vec
using ArgParse
using MHLib

export WalkSimulator, RandomWalkSimulator, SecondOrderRandomWalkSimulator, Struct2VecWalkSimulator,
    learn_embeddings, learn_embeddings_word2vec

# Thanks @ Dan Saattrup Nielsen for his explanation of deepwalk: https://saattrupdan.github.io/2020-08-24-deepwalk/
# Deepwalk implementation is inspired by his explanation and code 

const settings_cfg = ArgParseSettings()
@add_arg_table! settings_cfg begin
    "--tmpdir"
        help = "Temporary directory for word2vec output"
        arg_type = String
        default = "/tmp"
end

abstract type WalkSimulator end

struct RandomWalkSimulator <: WalkSimulator
    walk_length::Int
    window_size::Int
end

struct SecondOrderRandomWalkSimulator <: WalkSimulator
    walk_length::Int
    window_size::Int
    p::Real
    q::Real
    transition_probs::Dict{Tuple{Int, Int}, Vector{Float32}}

    function SecondOrderRandomWalkSimulator(walk_length::Int, window_size::Int, p::Real, q::Real)
        new(walk_length, window_size, p, q, Dict())
    end
end

mutable struct Struct2VecWalkSimulator <: WalkSimulator
    walk_length::Int
    window_size::Int
    k′::Int
    q::Real
    multi_layer_graph::Vector{SimpleWeightedGraph}
    layer_transition_probs

    function Struct2VecWalkSimulator(walk_length, window_size, k′, q)
        new(walk_length, window_size, k′, q, [], [])
    end
end

struct Struct2VecDist <: SemiMetric end
struct Struct2VecDistTuple <: SemiMetric end

(::Struct2VecDist)(a,b) = max(a,b)/min(a,b) + 1
(::Struct2VecDistTuple)(a,b) = Struct2VecDist()(a[1], b[1]) * max(a[2], b[2])

function init_struct2vec(rws::Struct2VecWalkSimulator, graph::AbstractGraph)
    # construct multilayer graph
    degrees = degree(graph)
    vertex_and_degree = [(degrees[i], i) for i in vertices(graph)]
    sort!(vertex_and_degree)

    num_neighbors = round(Int, log(nv(graph))) # optimization 2 from paper: only use 2 * log n most similar nodes as neighbors in multilayer graph
    neighbors_of_vertices = []

    mlg_edges = Set{Tuple{Int,Int}}() # set of edges of the multilayergraph as tuples (u,v) where u < v
    f_values = Dict{Tuple{Int,Int}, Float64}() # initialize f values for computation of edge weights
    
    for v in vertices(graph)
        idx = searchsortedfirst(vertex_and_degree, (degrees[v] , v))
        for i = max(1, idx-num_neighbors):min(nv(graph), idx+num_neighbors)
            (i == idx) && continue
            u = vertex_and_degree[i][2]
            curr_u, curr_v = min(u,v), max(u,v) # sort edge s.t. u < v 
            push!(mlg_edges, (curr_u, curr_v)) # add edge to list 
            f_values[(curr_u, curr_v)] = 0
        end
    end

    # optimization 3 from paper: only compute k′ layers
    k′ = min(rws.k′, diameter(graph))
    
    multi_layer_graph = SimpleWeightedGraph[]

    # for each layer, store transition probabilities in upwards and downwards direction 
    layer_transition_probs = [] 
    
    # f_values always contains f_k(u,v) for the previous layer

    for k in 1:(k′+1) # adjust for 1-indexed data structures
        current_graph = SimpleWeightedGraph(nv(graph))
        transition_probs = @NamedTuple{up::Float64, down::Float64}[]

        total_weight = 0 # sum of all weights of this layer

        for (u,v) in mlg_edges
            if k > 1 && !has_edge(multi_layer_graph[k-1], u, v)
                continue
            end

            if !has_edge(current_graph, u, v)
                g_value = g_dist(graph, u, v, k-1, degrees)

                # if one of the vertices does not have any vertices in the ring at distance k, 
                # the value is undefined and the edge disappears
                if g_value < 0 continue end 

                # update f_value for this layer
                f_values[(u,v)] = f_values[(u, v)] + g_value 
                
                # add edge to graph of current layer
                w = ℯ^(-f_values[(u,v)])
                total_weight += w
                SimpleWeightedGraphs.add_edge!(current_graph, u, v, w)
            end
        end

        if ne(current_graph) == 0 break end # if no edges are present, then stop constructing further layers
        
        # avg edge weight in the current layer
        avg_weight = total_weight / ne(current_graph)
        
        for u in vertices(graph)
            # compute transition probabilities
            N_u = neighbors(current_graph, u)
            # number of edges incident to u that have weight larger than avg edge weight of all edges in graph
            Γₖ_u = sum([ (get_weight(current_graph, u, v) > avg_weight) ? 1 : 0 for v in N_u])
            up = (k == k′+1) ? 0 : log(Γₖ_u + ℯ)
            down = (k == 1) ? 0 : 1
            push!(transition_probs, (;up, down))
        end
        push!(multi_layer_graph, current_graph)
        push!(layer_transition_probs, transition_probs)
    end

    return multi_layer_graph, layer_transition_probs
end

# function g from paper, computes dynamic time warp value between ordered degree series of k-hop rings of u, v
function g_dist(graph, u, v, k, degrees)
    a = sort([degrees[i] for i in k_hop_ring(graph, u, k)]; rev=true)
    b = sort([degrees[i] for i in k_hop_ring(graph, v, k)]; rev=true)

    if isempty(a) || isempty(b) return -1 end

    # TODO / Maybe: Implement optimization 1 from paper
    return dtw(a, b, Struct2VecDist())[1]
end

function k_hop_ring(graph, v, k)
    k == 0 && return [v]
    
    return setdiff(neighborhood(graph, v, k), neighborhood(graph, v, k-1))
end

init_walk_simulator(::WalkSimulator, graph::AbstractGraph) = nothing
init_walk_simulator(rws::SecondOrderRandomWalkSimulator, graph::AbstractGraph) = empty!(rws.transition_probs)
function init_walk_simulator(rws::Struct2VecWalkSimulator, graph::AbstractGraph)
    rws.multi_layer_graph, rws.layer_transition_probs = init_struct2vec(rws, graph)
end

function skip_gram_model(vocabulary_size, embedding_size)::Chain
    model = Chain(
        Dense(vocabulary_size => embedding_size),
        Dense(embedding_size => vocabulary_size),
    )
    return model
end

function simulate_random_walk(ws::WalkSimulator, graph::AbstractGraph, v_start::Int)
    walk = random_walk(ws, graph, v_start)
    nodes, context = get_context(walk, ws.window_size)
    return nodes, context
end

function skip_gram_loss(model, nodes::Vector{Int}, context::Vector{Int}, labels::Vector{Int})
    walk = Flux.onehotbatch(nodes, labels)
    context = Flux.onehotbatch(context, labels)
    Flux.logitcrossentropy(model(walk), context)
end

"""
    random_walk(rws::RandomWalkSimulator, graph::AbstractGraph, v_start::Int)

Implementation of a basic random walk, where at each point in the walk, a random neighbor is 
sampled with uniform probability. Corresponds to the random walks used in DeepWalk. 

"""
function random_walk(rws::RandomWalkSimulator, graph::AbstractGraph, v_start::Int)::Vector{Int}
    walk = [v_start]
    curr = v_start
    for _=2:rws.walk_length
        curr = sample(neighbors(graph, curr))
        push!(walk, curr)
    end
    return walk
end

"""
    random_walk(rws::SecondOrderRandomWalkSimulator, graph::AbstractGraph, v_start::Int)

Implementation of a second order random walk for Node2Vec with hyperparameters p and q

"""
function random_walk(rws::SecondOrderRandomWalkSimulator, graph::AbstractGraph, v_start::Int)::Vector{Int}
    walk = [v_start]
    curr = sample(neighbors(graph, v_start))
    prev = v_start
    p = rws.p
    q = rws.q

    for _=3:rws.walk_length
        # current edge is always prev -> curr
        neighboring_nodes = neighbors(graph, curr)

        # if this edge has not been traversed yet, compute unnormalized probabilities
        if !haskey(rws.transition_probs, (prev, curr))
            weights = Float32[]
            for x in neighboring_nodes
                if prev == x
                    push!(weights, 1/p)
                elseif has_edge(graph, prev, x)
                    push!(weights, 1)
                else
                    push!(weights, 1/q)
                end
            end
            rws.transition_probs[(prev, curr)] = weights
        end

        weights = rws.transition_probs[(prev, curr)]
            
        prev = curr
        curr = sample(neighboring_nodes, Weights(weights))
        push!(walk, curr)
    end
    return walk
end

function random_walk(rws::Struct2VecWalkSimulator, graph::AbstractGraph, v_start::Int)::Vector{Int}
    multi_layer_graph = rws.multi_layer_graph
    layer_transition_probs = rws.layer_transition_probs
    q = rws.q

    layer = 0
    max_layer = length(multi_layer_graph)

    walk = [v_start]
    curr = v_start

    for _=2:rws.walk_length
        # with probability 1-q change layers
        if rand() > q
            p_up, p_down = layer_transition_probs[layer+1][curr].up, layer_transition_probs[layer+1][curr].down
            p = p_up / (p_up + p_down)
            if rand() < p
                if layer+1 < max_layer && !isempty(neighbors(multi_layer_graph[layer+2], curr))
                    layer += 1
                end
            else
                layer = max(0, layer-1)
            end
        end

        N_v = neighbors(multi_layer_graph[layer+1], curr)
        curr = sample(N_v, Weights([get_weight(multi_layer_graph[layer+1], curr, v) for v in N_v]))
        push!(walk, curr)
    end
    return walk
end

function get_context(walk::Vector{Int}, window_size::Int)
    nodes = Int[]
    context = Int[]
    for i in eachindex(walk)
        left = max(1, i-window_size)
        right = min(length(walk), i+window_size)
        for w in walk[left : right]
            if w == walk[i]
                continue
            end
            push!(nodes, walk[i])
            push!(context, w) 
        end
    end
    return nodes, context
end

function learn_embeddings(ws::WalkSimulator, graph::AbstractGraph; embedding_size::Int=64, walks_per_node::Int=1)::AbstractMatrix{Float32}
    init_walk_simulator(ws, graph)
    model = skip_gram_model(nv(graph), embedding_size)
    loss = (nodes, context, labels) -> skip_gram_loss(model, nodes, context, labels)
    labels = collect(vertices(graph))
    opt = Adam(0.001, (0.9, 0.999))
    ps = Flux.params(model)

    for _ in 1:walks_per_node
        for v_start in shuffle(vertices(graph))
            nodes, context = simulate_random_walk(ws, graph, v_start)
            
            gs = gradient(ps) do 
                loss(nodes, context, labels) 
            end
            
            Flux.Optimise.update!(opt, ps, gs)
        end
    end
    
    return model[1](Flux.onehotbatch(labels, labels))
end

# word2vec only works on linux/mac but not on windows 
function learn_embeddings_word2vec(ws::WalkSimulator, graph::AbstractGraph; embedding_size::Int=64, walks_per_node::Int=1)
    init_walk_simulator(ws, graph)
    walks = []
    for _ in 1:walks_per_node
        for v_start in shuffle(vertices(graph))
            walk = random_walk(ws, graph, v_start)
            push!(walks, walk)
        end
    end
    str_walks = map(x -> string.(x), walks)
    walks_file = tempname(settings[:tmpdir]; cleanup=false)
    vecs_file = tempname(settings[:tmpdir]; cleanup=false)
    writedlm(walks_file, str_walks)
    # suppress word2vec logs
    redirect_stdout(()->word2vec(walks_file, vecs_file; verbose=false, debug=0, size=embedding_size, window=ws.window_size), open("/dev/null", "w"))
    model = wordvectors(vecs_file)
    rm(walks_file)
    rm(vecs_file)
    return reduce(hcat, [get_vector(model, string(v)) for v in vertices(graph)])
end

end # module
