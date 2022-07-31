module LocalSearch

using StatsBase
using Graphs
# using thesis.LookaheadSearch

export local_search_with_EX, construction_heuristic, lower_bound_heuristic, 
    LocalSearchSettings, ConstructionHeuristicSettings,
    GuidanceFunction, GreedyCompletionHeuristic, GreedyCompletionHeuristicPQVariant, SumOfNeighborsHeuristic, 
    ConfigurationChecking, TabuList,
    run_MQCP

include("ConstructionHeuristics.jl")
include("ShortTermMemory.jl")

"""
   ExplorationConstructionSettings
   
Settings for construction heuristics used in the local search algorithm (Algorithms 5.3 and 5.4 in thesis)

- `p`: Controls the balance between GRASP construction and preferring vertices with low frequency values. 
    `p`=0 ignores frequency values, while `p`=1 only uses frequency values. 
- `α`: GRASP parameter; α=0 performs a greedy construction, α=1 performs a randomized construction
- `β`: Beamwidth
- `expansion_limit`: Every node in the beam search tree is expanded into at most `expansion_limit` nodes
- `guidance_function`: Guidance function used to evaluate nodes in the beam search
"""
struct ConstructionHeuristicSettings
    p::Real
    α::Real
    β::Int # beamwidth
    expansion_limit::Int
    guidance_function::GuidanceFunction
end

"""
    LocalSearchSettings

Settings for the local search algorithm. 

- `g`: Input graph
- `construction_heuristic_settings`: Settings for construction heuristics used in the local search algorithm
- `stm`: Short term memory mechanism used in local search (TabuList, ConfigurationChecking)
- `timelimit`: Timelimit for search in seconds
- `max_iter`: Search is restarted after `max_iter` iterations without finding an improved solution
- `next_improvement`: If true: Next improvement is used as strategy when searching neighborhoods. Otherwise, 
    best improvement is used. 
"""
struct LocalSearchSettings
    g::SimpleGraph
    construction_heuristic_settings::ConstructionHeuristicSettings
    stm::ShortTermMemory
    timelimit::Float64
    max_iter::Int
    next_improvement::Bool

    # default settings
    function LocalSearchSettings(g::SimpleGraph; 
                                 construction_heuristic_settings::ConstructionHeuristicSettings = ConstructionHeuristicSettings(0.2, 0.2, 1, 50, GreedyCompletionHeuristic()), 
                                 short_term_memory::ShortTermMemory = ConfigurationChecking(g, 7),
                                 timelimit::Float64 = 600.0,
                                 max_iter::Int = 4000,
                                 next_improvement::Bool = true
                                 )
        new(g, construction_heuristic_settings, short_term_memory, timelimit, max_iter, next_improvement)
    end
end

"""
    run_MQCP(g, γ; settings)

Runs the local search algorithm for the MQCP for the instance defined by graph `g` and target 
density `γ` with settings defined in `settings` 
"""
function run_MQCP(g::SimpleGraph, γ::Real; settings::LocalSearchSettings)
    # initial construction
    @debug "Constructing initial solution..."
    S′ = lower_bound_heuristic(g, γ, 
                                 settings.construction_heuristic_settings.guidance_function; 
                                 settings.construction_heuristic_settings.β,
                                 settings.construction_heuristic_settings.expansion_limit)
    @debug "Found initial solution of size $(length(S′))"
    k = length(S′)+1
    freq = fill(0, nv(g))
    timelimit = time() + settings.timelimit

    while time() < timelimit
        @debug "Generate candidate solution for size $k"
        S = construction_heuristic(g, k, freq; 
                                   settings.construction_heuristic_settings.p,
                                   settings.construction_heuristic_settings.α)
        @debug "Starting local search with candidate solution of density $(density_of_subgraph(g, S))"
        S, freq = local_search_procedure(g, S, γ, freq, settings.stm;
                                   timelimit, settings.max_iter, settings.next_improvement)
        @debug "Found solution with density $(density_of_subgraph(g, S))"

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
                                timelimit::Float64, max_iter::Int, next_improvement::Bool)::Tuple{Vector{Int}, Vector{Int}}
    k = length(S)
    best_obj = calculate_num_edges(g, S)
    d_S = calculate_d_S(g, S)
    S = Set(S)
    S′ = copy(S)
    V_S = Set(filter(v -> v ∉ S, vertices(g)))
    current_obj = best_obj
    min_edges_needed = γ * k * (k-1) / 2

    iter_since_last_improvement = 0

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
            u, v, Δuv = search_neighborhood(g, d_S, X_restricted, Y_restricted; next_improvement)
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
            return collect(S′), freq
        end 
    end
    return collect(S′), freq
end

"""
    gain(g, d_S, u, v)

Calculates Δuv, which is the gain of edges in the candidate solution S when vertex u ∈ S is swapped 
with vertex v ∈ V ∖ S.

- `g`: Input graph
- `d_S`: d_S[w] Contains number of neighboring vertices of w in S for each vertex in `g`
- `u`: Vertex in S
- `v`: Vertex in V ∖ S 
"""
function gain(g::SimpleGraph, d_S::Vector{Int}, u::Int, v::Int) 
    return d_S[v] - d_S[u] - Int(has_edge(g, u ,v))
end

"""
    update_d_S!(g, d_S, u, v)

Update d_S efficiently after swapping vertex u ∈ S with vertex v ∈ V ∖ S.

- `g`: Input graph
- `d_S`: d_S[w] Contains number of neighboring vertices of w in S for each vertex in `g`
- `u`: Vertex in S
- `v`: Vertex in V ∖ S 
"""
function update_d_S!(g::SimpleGraph, d_S::Vector{Int}, u::Int, v::Int)
    for w in neighbors(g, u)
        d_S[w] -= 1
    end
    for w in neighbors(g, v)
        d_S[w] += 1
    end
end

"""
    search_neighborhood(g, d_S, X, Y; next_improvement)

Search the neighborhood relative to candidate solution S defined by swapping vertices in X ⊆ S with vertices in Y ⊆ V ∖ S. 
If `next_improvement` is true, the first improving move is returned. Otherwise, the whole neighborhood is searched 
and the best move is returned. If no improving move can be found, the best non-improving move is returned. 
Returns a triple u, v, Δuv, where u ∈ X, v ∈ Y, and Δuv is the gain of edges in the candidate solution S.  

- `g`: Input graph
- `d_S`: d_S[w] Contains number of neighboring vertices of w in S for each vertex in `g`
- `X`: (Restricted) candidate list X ⊆ S
- `Y`: (Restricted) candidate list Y ⊆ V ∖ S
- `next_improvement`: Determines whether the neighborhood is searched with next improvement or best improvement strategy. 

"""
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
    return density_of_subgraph(g, S) >= γ 
end

function density_of_subgraph(g, S)
    density(induced_subgraph(g, S)[1])
end

# # used for testing lookahead search, maybe remove later
# function local_search_with_EX(g, γ; d=10, α=10, β=100)
#     restarts = 0
#     best = Set(), -Inf
#     k = 3

#     while restarts < 5
#         candidate_solution = grasp(g, k)
        
#         improvement = true

#         current = ne(induced_subgraph(g, collect(candidate_solution))[1])
#         max_γ = k*(k-1)/2

#         while improvement
#             bs_obj, bs_sol = beam_search(g, candidate_solution, min(k, d); α, β)

#             if bs_obj > current
#                 current = bs_obj
#                 candidate_solution = bs_sol
#             else
#                 improvement = false
#             end

#             if current / max_γ >= γ
#                 best = candidate_solution, current
#                 break
#             end
#         end

#         if current / max_γ < γ
#             restarts += 1
#             @debug "Restarting"
#         else
#             @debug "Found solution of size $k"
#             k += 1
#             restarts = 0
#         end
#     end
#     return best
# end

end