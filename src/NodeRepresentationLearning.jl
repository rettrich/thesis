module NodeRepresentationLearning

using StatsBase
using Graphs
using Flux
using Random

export WalkSimulator, RandomWalkSimulator, SecondOrderRandomWalkSimulator,
    learn_embeddings

# Thanks @ Dan Saattrup Nielsen for his explanation of deepwalk: https://saattrupdan.github.io/2020-08-24-deepwalk/
# Deepwalk implementation is inspired by his explanation and code 

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

end # module

