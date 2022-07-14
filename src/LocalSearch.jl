module LocalSearch

using StatsBase
using Graphs
using thesis.LookaheadSearch

export local_search_with_EX, construction_heuristic, beam_search_construction, 
    GuidanceFunction, GreedyCompletionHeuristic, SumOfNeighborsHeuristic, 
    calculate_d_S, calculate_num_edges, run

include("ConstructionHeuristics.jl")
include("ShortTermMemory.jl")

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
    run_MQCP(settings, g, γ)

Runs the local search algorithm for the MQCP for the instance defined by graph `g` and target 
density `γ` with settings defined in `settings` 
"""
function run_MQCP(g::SimpleGraph, γ::Real; settings::LocalSearchSettings)
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
        S, freq = local_search_procedure(g, S, γ, freq, settings.stm;
                                   settings.timelimit, settings.max_iter, settings.first_improvement)

        if is_feasible_MQC(g, S, γ)
            S = extend_solution(g, S, γ)
            S′ = S
            k = length(S′)+1
            freq = fill(0, nv(g))
        end
    end
    return S′
end

# TODO: Extend to bfs variant?
"""
    extend_solution(g, S, γ)

Greedily extends solution `S` by adding the vertex outside `S` with maximum number of neighbors in `S` 
if this produces a valid γ-quasi clique. 

- `g`: Input graph
- `S`: Feasible γ-quasi clique to extend
- `γ`: Target density

"""
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

"""
    local_search_procedure(g, S, γ, freq, short_term_memory; timelimit, max_iter, first_improvement)

Local search procedure for MQCP. Corresponds to Algorithm 5.5 in thesis. 
Returns the best found solution and the updated frequency list that tracks how many times each vertex was moved. 

- `g`: Input graph
- `S`: Candidate solution
- `γ`: Target density
- `freq`: Frequency list, keeps track of how many times each vertex is moved
- `short_term_memory`: Short term memory mechanism that blocks vertices from being moved in and out 
    of the solution too often for a short time. 
- `timelimit`: Cutoff time for search
- `max_iter`: Stop search after `max_iter` iterations of no improvement
- `first_improvement`: Strategy used for searching the neighborhood: If `first_improvement` is `true`, then 
        the first improving neighboring solution will be selected, otherwise the neighborhood is always searched to 
        completion and best improvement is used. 

"""
function local_search_procedure(g::SimpleGraph, S::Vector{Int}, γ::Real, freq::Vector{Int}, 
                                short_term_memory::ShortTermMemory; 
                                timelimit::Int, max_iter::Int, first_improvement::Bool)::Tuple{Vector{Int}, Vector{Int}}
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
            # TODO: if move is not improving, maybe use arbitrary move?
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
    return collect(S′), freq
end

function gain(g::SimpleGraph, d_S::Vector{Int}, u::Int, v::Int) 
    return d_S[v] - d_S[u] - Int(has_edge(g, u ,v))
end

function update_d_S!(g::SimpleGraph, d_S::Vector{Int}, u::Int, v::Int)
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
            return best
        end
    end
    return best
end

"""
    first_non_empty(itrs...)

Returns first non-empty iterator in `itrs` or an error, if all are empty. 

"""
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