module LocalSearch

using StatsBase
using Graphs
using thesis.LookaheadSearch

export local_search_with_EX, construction_heuristic, beam_search_construction, 
    GuidanceFunction, GreedyCompletionHeuristic, SumOfNeighborsHeuristic, 
    calculate_d_S, calculate_num_edges, run

include("ConstructionHeuristics.jl")

"""
    BeamSearchConstructionSettings

Settings for beam search construction heuristic (Algorithm 5.3 in thesis)

- `β`: Beamwidth
- `expansion_limit`: Every node in the beam search tree is expanded into at most `expansion_limit` nodes

"""
struct BeamSearchConstructionSettings
    β::Int # beamwidth
    expansion_limit::Int
    guidance_function::GuidanceFunction
end

"""
   ExplorationConstructionSettings
   
Settings for construction heuristic that aims to explore the search space (Algorithm 5.4)
"""
struct ExplorationConstructionSettings
    p::Real
    α::Real
end

struct LocalSearchSettings
    initial_construction_settings::BeamSearchConstructionSettings
    construction_heuristic_settings::ExplorationConstructionSettings
    timelimit::Int
    max_iter::Int
    next_improvement::Bool
    stm::ShortTermMemory
end

"""
    ShortTermMemory

A short term memory mechanism (e.g. tabu list) to prevent cycling through the 
same solutions during a local search procedure. 
Each implementation has to implement the methods reset!, remove_blocked!, move!

"""
abstract type ShortTermMemory end

"""
    reset!(short_term_memory)

Resets the short term memory. Memory about blocked vertices is reset

- `short_term_memory`: Short term memory to be reset

"""
reset!(stm::ShortTermMemory) = 
    error("ShortTermMemory: Abstract reset! called")

"""
    get_blocked!(short_term_memory, vertex_list)

Returns all currently blocked vertices as a vector.

- `short_term_memory`: The short term memory data structure from which information 
    about blocked vertices is retrieved

"""
get_blocked(stm::ShortTermMemory) = 
    error("ShortTermMemory: Abstract remove_blocked! called")

"""
    move!(short_term_memory, u, v)

This method should be called when a vertex `u` ∈ S is swapped with a vertex `v` ∈ V ∖ S. 
The `short_term_memory` then stores information about this swap and blocks 
vertices `u` or `v` according to the concrete implementation. 
"""
move!(stm::ShortTermMemory, u::Int, v::Int) = 
    error("ShortTermMemory: Abstract move! called")

"""
    TabuList

Keeps a tabu list of vertices that are blocked from being swapped. 

- `g`: Input graph
- `block_length`: Each time a vertex is added to the `tabu_list`, it is blocked 
    for `block_length` iterations
- `tabu_list`: `tabu_list[i]` corresponds to vertex `i`. Vertex `i` is blocked until iteration 
    `tabu_list[i]`
. `iteration`: Iteration counter, is increased each time `move!` is called on a `TabuList` instance
"""
mutable struct TabuList <: ShortTermMemory 
    g::SimpleGraph
    block_length::Int
    tabu_list::Vector{Int}
    iteration::Int

    function TabuList(n::Int, block_length::Int)
        new(n, block_length, fill(0, n), 0)
    end
end

function reset!(stm::TabuList)
    stm.tabu_list = fill(0, stm.n)
    stm.iteration = 0
end

function get_blocked(stm::TabuList)
    filter(v -> stm.tabu_list[v] > stm.iteration , vertices(stm.g))
end

function move!(stm::TabuList, u::Int, v::Int)
    stm.tabu_list[u] = stm.iteration + stm.block_length
    stm.tabu_list[v] = stm.iteration + stm.block_length
    stm.iteration += 1
end

"""
    ConfigurationChecking

ConfigurationChecking according to BoundedCC as described in Chen et al. 2021, 
NuQClq: An Effective Local Search Algorithm for Maximum Quasi-Clique Problem. 
Initially, each for each vertex v set `conf_change[v]` and `threshold[v]` to 1. 
Each time a vertex v is added to the solution, increase `conf_change[u]` by 1 for all neighbors of v, 
and increase `threshold[v]` by 1. If `threshold[v]` > `ub_threshold`, then `threshold[v]` is reset to 1.
Each time a vertex v is moved outside of the solution, `conf_change[v]` is reset to 0. 
A vertex v is blocked if `conf_change[v]` < `threshold[v]`. 

- `g`: Input graph
- `ub_threshold`: Upper bound for threshold. When `threshold[v]` > `ub_threshold`, set `threshold[1]` to 1. 
- `threshold`: Contains threshold values for all vertices in `g`. 
- `conf_change`: Contains current configuration for all vertices in `g`.

"""
mutable struct ConfigurationChecking <: ShortTermMemory 
    g::SimpleGraph
    ub_threshold::Int
    threshold::Vector{Int}
    conf_change::Vector{Int}

    function TabuList(g::SimpleGraph, ub_threshold::Int)
        new(g, ub_threshold, fill(1, nv(g)), fill(1, nv(g)))
    end
end

function reset!(stm::ConfigurationChecking)
    stm.threshold = fill(1, n)
    stm.conf_change = fill(1, n)
end

function get_blocked(stm::ConfigurationChecking)
    filter(v -> stm.conf_change[v] < stm.threshold[v], vertices(stm.g))
end

function move!(stm::ConfigurationChecking, u::Int, v::Int)
    # u is moved from S outside of solution
    stm.conf_change[u] = 0

    # v is moved from V ∖ S into S
    for w in neighbors(stm.g, v)
        conf_change[w] += 1
    end
    stm.threshold[v] += 1
    stm.threshold[v] > stm.ub_threshold && (stm.threshold = 1)
end


function run(settings::LocalSearchSettings, g::SimpleGraph, γ::Real)
    # initial construction
    S′ = beam_search_construction(g, γ, 
                                 settings.initial_construction_settings.guidance_function; 
                                 settings.initial_construction_settings.β,
                                 settings.initial_construction_settings.expansion_limit)
    k = length(S′)+1
    freq = fill(0, nv(g))

    while time() < settings.timelimit
        S = construction_heuristic(g, k, freq; 
                                   settings.construction_heuristic_settings.p,
                                   settings.construction_heuristic_settings.α)
        S = local_search_procedure(g, S, γ, freq, settings.stm)

        if is_feasible_MQC(g, S, γ)
            S = extend_solution(g, S, γ)
            S′ = S
            k = length(S′)+1
            freq = fill(0, nv(g))
        end
    end
    return S′
end

# TODO: Extend to bfs variant
function extend_solution(g::SimpleGraph, S::Vector{Int}, γ::Real)::Vector{Int}
    S = copy(S)
    V_S = Set(setdiff(vertices(g), S))
    d_S = calculate_d_S(g, S)
    num_edges = calculate_num_edges(g, S)
    k = length(S)

    while true
        min_edges_needed = ceil(Int, γ * k * (k+1) / 2)
        for v in V_S
            if num_edges + d_S[v] >= min_edges_needed
                delete!(V_S, v)
                push!(S, v)
                num_edges = num_edges + d_S[v]
                for u in neighbors(g, v)
                    d_S[u] += 1
                end
                k += 1
                break
            end
        end
        break
    end
    return S
end

function local_search_procedure(g::SimpleGraph, S::Vector{Int}, γ::Real, freq::Vector{Int}, 
                                short_term_memory::ShortTermMemory; timelimit::Int, max_iter::Int, first_improvement::Bool)::Vector{Int}
    k = length(S)
    best_obj = calculate_num_edges(g, S)
    d_S = calculate_d_S(g, S)
    S = Set(S)
    S′ = copy(S)
    V_S = Set(filter(v -> v ∉ S, vertices(g)))
    current_obj = best_obj
    min_edges_needed = γ * k * (k-1) / 2

    iter_since_last_improvement = 0
    timelimit = time() + timelimit
    
    reset!(short_term_memory)

    while time() < timelimit && iter_since_last_improvement < max_iter
        blocked = get_blocked(short_term_memory)
        X_unblocked = filter(u -> u ∉ blocked, S)
        Y_unblocked = filter(v -> v ∉ blocked, V_S)
        d_min = minimum([d_S[i] for i in X_unblocked])
        d_max = maximum([d_S[i] for i in Y_unblocked])
        X_restricted = filter(u -> (d_S[u] <= d_min+1), X_unblocked)
        Y_restricted = filter(v -> (d_S[v] >= d_max-1), Y_unblocked)

        if !isempty(X_restricted) && !isempty(Y_restricted)
            u, v, Δuv = search_neighborhood(g, d_S, X_restricted, Y_restricted; first_improvement)
            # if move is not improving, maybe use arbitrary move
        else 
            u = sample(first_non_empty(X_restricted, X_unblocked, S))
            v = sample(first_non_empty(Y_restricted, Y_unblocked, V_S))
            Δuv = gain(g, d_S, u, v)
        end

        # update S, V∖S
        push!(S, v)
        delete!(S, u)
        push!(V_S, u)
        delete!(V_S, v)

        # update short term memory
        move!(short_term_memory, u, v)
        
        # update long term memory
        freq[u] += 1
        freq[v] += 1
        
        # update current candidate solution and corresponding data
        current_obj += Δuv
        update_d_S!(g, d_S, u, v)
        
        if current_obj > best_obj
            S′ = S
            best_obj = current_obj
            iter_since_last_improvement = 0
        else
            iter_since_last_improvement += 1
        end

        if best_obj >= min_edges_needed
            return collect(S′)
        end 
    end
    return collect(S′)
end

function gain(g, d_S, u, v) 
    return d_S[v] - d_S[u] - Int(has_edge(g, u ,v))
end

function update_d_S!(g, d_S, u, v)
    for w in neighbors(g, u)
        d_S[w] -= 1
    end
    for w in neighbors(g, v)
        d_S[w] += 1
    end
end

function search_neighborhood(g, d_S, X, Y; next_improvement=true)
    best = 0, 0, -Inf
    for u ∈ X, v ∈ Y
        Δuv = gain(g, d_S, u, v)
        if Δuv > best[3]
            best = u, v, Δuv
        end
        if next_improvement && Δuv > 0
            return best...
        end
    end
    return best...
end

function first_non_empty(itrs...)
    for i in itrs
        if !isempty(i)
            return i
        end
    end
    error("all empty")
end


function local_search_with_EX(g, γ; d=10, α=10, β=100)
    restarts = 0
    best = Set(), -Inf
    k = 3

    while restarts < 5
        candidate_solution = grasp(g, k)
        
        improvement = true

        current = ne(induced_subgraph(g, collect(candidate_solution))[1])
        max_γ = k*(k-1)/2

        while improvement
            bs_obj, bs_sol = beam_search(g, candidate_solution, min(k, d); α, β)

            if bs_obj > current
                current = bs_obj
                candidate_solution = bs_sol
            else
                improvement = false
            end

            if current / max_γ >= γ
                best = candidate_solution, current
                break
            end
        end

        if current / max_γ < γ
            restarts += 1
            @debug "Restarting"
        else
            @debug "Found solution of size $k"
            k += 1
            restarts = 0
        end
    end
    return best
end

"""
    calculate_d_S(g, candidate_solution)

Returns a vector d_S in the size of the vertex set of `vertices(g)`, where `d_S[i]` denotes the number of 
adjacent vertices in `S` for vertex `i` in `g`. 
    
- `g`: Input Graph
- `S`: Vector of vertices in `g`
"""
function calculate_d_S(g::SimpleGraph, S::Vector{Int})
    d_S = Int[0 for _ in 1:nv(g)]
    for u in S
        for v in neighbors(g, u)
            d_S[v] += 1
        end
    end
    return d_S
end

function is_feasible_MQC(g, S, γ)
    return density(induced_subgraph(g, S)[1]) >= γ 
end

end