module LocalSearch

using StatsBase
using Graphs
# using thesis.LookaheadSearch

export calculate_d_S, calculate_num_edges,
    LocalSearchBasedMH,
    LowerBoundHeuristic, BeamSearch_LowerBoundHeuristic,
    ConstructionHeuristic, Freq_GRASP_ConstructionHeuristic,
    ConfigurationChecking, TabuList,
    LocalSearchProcedure, MQCP_LocalSearchProcedure,
    GuidanceFunction, GreedyCompletionHeuristic, GreedyCompletionHeuristicPQVariant, SumOfNeighborsHeuristic,
    SwapHistory, sample_candidate_solutions,
    ScoringFunction, d_S_ScoringFunction, GNN_ScoringFunction,
    SolutionExtender, MQCP_GreedySolutionExtender,
    FeasibilityChecker, MQCP_FeasibilityChecker,
    run_lsbmh

include("LowerBoundHeuristic.jl")
include("ConstructionHeuristic.jl")
include("ShortTermMemory.jl")
include("SwapHistory.jl")
include("ScoringFunctions.jl")
include("LocalSearchProcedure.jl")
include("SolutionExtender.jl")
include("FeasibilityChecker.jl")

"""
    LocalSearchBasedMH

Type for the Local Search Based Metaheuristic. Consists of these key components that depend on the specific problem 
(MQCP, MDCP, MDP): 
- a `LowerBoundHeuristic` used to quickly obtain a feasible solution
- a `ConstructionHeuristic` that produces a candidate solution as a starting point for local search
- a `LocalSearchProcedure` that performs a (swap-based) local search, and includes a `ShortTermMemory` and `ScoringFunction`
- a `FeasibilityChecker` that checks the (problem specific) feasibility of a candidate solution 
- a `SolutionExtender` that tries to extend an already feasible solution into one of greater cardinality

Furthermore, the following settings are customisable: 
- `timelimit`: Timelimit for search in seconds
- `max_iter`: Search is restarted after `max_iter` iterations without finding an improved solution
- `next_improvement`: If true: Next improvement is used as strategy when searching neighborhoods. Otherwise, 
    best improvement is used. 
- `record_swap_history`: If true: Records all encountered candidate solutions as a `SwapHistory` in order to 
    process them later (e.g. create training samples). 

"""
struct LocalSearchBasedMH
    # key components
    lower_bound_heuristic::LowerBoundHeuristic
    construction_heuristic::ConstructionHeuristic
    local_search_procedure::LocalSearchProcedure
    feasibility_checker::FeasibilityChecker
    solution_extender::SolutionExtender

    # settings
    timelimit::Float64
    max_iter::Int
    next_improvement::Bool
    record_swap_history::Bool

    function LocalSearchBasedMH(lower_bound_heuristic::LowerBoundHeuristic, 
                                construction_heuristic::ConstructionHeuristic,
                                local_search_procedure::LocalSearchProcedure,
                                feasibility_checker::FeasibilityChecker,
                                solution_extender::SolutionExtender;
                                timelimit::Float64 = 600.0, # 10 minutes
                                max_iter::Int = 4000, 
                                next_improvement::Bool = false,
                                record_swap_history::Bool = false,
                                )
        new(lower_bound_heuristic, construction_heuristic, local_search_procedure, feasibility_checker, solution_extender, 
            timelimit, max_iter, next_improvement, record_swap_history)
    end
end

# """
#     LocalSearchSettings

# Settings for the local search algorithm. 

# - `g`: Input graph
# - `construction_heuristic_settings`: Settings for construction heuristics used in the local search algorithm
# - `stm`: Short term memory mechanism used in local search (TabuList, ConfigurationChecking)
# - `timelimit`: Timelimit for search in seconds
# - `max_iter`: Search is restarted after `max_iter` iterations without finding an improved solution
# - `next_improvement`: If true: Next improvement is used as strategy when searching neighborhoods. Otherwise, 
#     best improvement is used. 
# """
# struct LocalSearchSettings
#     g::SimpleGraph
#     construction_heuristic_settings::ConstructionHeuristicSettings
#     stm::ShortTermMemory
#     timelimit::Float64
#     max_iter::Int
#     next_improvement::Bool
#     record_swap_history::Bool

#     # default settings
#     function LocalSearchSettings(g::SimpleGraph; 
#                                  construction_heuristic_settings::ConstructionHeuristicSettings = ConstructionHeuristicSettings(0.2, 0.2, 1, 50, GreedyCompletionHeuristic()), 
#                                  short_term_memory::ShortTermMemory = ConfigurationChecking(g, 7),
#                                  timelimit::Float64 = 600.0,
#                                  max_iter::Int = 4000,
#                                  next_improvement::Bool = true,
#                                  record_swap_history::Bool = false
#                                  )
#         new(g, construction_heuristic_settings, short_term_memory, timelimit, 
#             max_iter, next_improvement, record_swap_history)
#     end
# end

function run_lsbmh(local_search::LocalSearchBasedMH, graph::SimpleGraph)::@NamedTuple{solution::Vector{Int}, swap_history::Union{Nothing, SwapHistory}}
    # initial construction
    @debug "Constructing initial solution..."
    S′ = local_search.lower_bound_heuristic(graph)

    @debug "Found initial solution of size $(length(S′))"
    k = length(S′)+1
    freq = fill(0, nv(graph))
    timelimit = time() + local_search.timelimit

    swap_history::Union{Nothing, SwapHistory} = local_search.record_swap_history ? SwapHistory(graph) : nothing

    while time() < timelimit
        @debug "Generate candidate solution for size $k"
        S = local_search.construction_heuristic(graph, k, freq)

        if !isnothing(swap_history)
            push!(swap_history, Set(S))
        end

        @debug "Starting local search with candidate solution of density $(density_of_subgraph(graph, S))"
        S, freq, swap_history = local_search.local_search_procedure(graph, S, freq;
                    timelimit, local_search.max_iter, local_search.next_improvement, swap_history)
        
        @debug "Found solution with density $(density_of_subgraph(graph, S))"

        if local_search.feasibility_checker(graph, S)
            S = extend(local_search.solution_extender, graph, S)
            S′ = S
            k = length(S′)+1
            freq = fill(0, nv(graph))
        end
    end
    return (;solution=S′, swap_history)
end

# """
#     run_MQCP(g, γ; settings)

# Runs the local search algorithm for the MQCP for the instance defined by graph `g` and target 
# density `γ` with settings defined in `settings` 
# """
# function run_MQCP(g::SimpleGraph, γ::Real; settings::LocalSearchSettings)
#     # initial construction
#     @debug "Constructing initial solution..."
#     S′ = lower_bound_heuristic(g, γ, 
#                                  settings.construction_heuristic_settings.guidance_function; 
#                                  settings.construction_heuristic_settings.β,
#                                  settings.construction_heuristic_settings.expansion_limit)
#     @debug "Found initial solution of size $(length(S′))"
#     k = length(S′)+1
#     freq = fill(0, nv(g))
#     timelimit = time() + settings.timelimit

#     swap_history::Union{Nothing, SwapHistory} = settings.record_swap_history ? SwapHistory(g) : nothing

#     while time() < timelimit
#         @debug "Generate candidate solution for size $k"
#         S = construction_heuristic(g, k, freq; 
#                                    settings.construction_heuristic_settings.p,
#                                    settings.construction_heuristic_settings.α)

#         if !isnothing(swap_history)
#             push!(swap_history, Set(S))
#         end

#         @debug "Starting local search with candidate solution of density $(density_of_subgraph(g, S))"
#         S, freq, swap_history = local_search_procedure(g, S, γ, freq, settings.stm;
#                                    timelimit, settings.max_iter, settings.next_improvement,
#                                    swap_history)
#         @debug "Found solution with density $(density_of_subgraph(g, S))"

#         if is_feasible_MQC(g, S, γ)
#             S = extend_solution(g, S, γ)
#             S′ = S
#             k = length(S′)+1
#             freq = fill(0, nv(g))
#         end
#     end
#     return S′, swap_history
# end

# # TODO: Extend to bfs variant?
# """
#     extend_solution(g, S, γ)

# Greedily extends solution `S` by adding the vertex outside `S` with maximum number of neighbors in `S` 
# if this produces a valid γ-quasi clique. 

# - `g`: Input graph
# - `S`: Feasible γ-quasi clique to extend
# - `γ`: Target density

# """
# function extend_solution(graph::SimpleGraph, S::Vector{Int}, γ::Real)::Vector{Int}
#     S = copy(S)
#     V_S = Set(setdiff(vertices(graph), S))
#     d_S = calculate_d_S(graph, S)
#     num_edges = calculate_num_edges(graph, S)
#     k = length(S)

#     while true
#         min_edges_needed = ceil(Int, γ * k * (k+1) / 2)
#         for v in V_S
#             if num_edges + d_S[v] >= min_edges_needed
#                 delete!(V_S, v)
#                 push!(S, v)
#                 num_edges = num_edges + d_S[v]
#                 for u in neighbors(graph, v)
#                     d_S[u] += 1
#                 end
#                 k += 1
#                 break
#             end
#         end
#         break
#     end
#     return S
# end

# """
#     local_search_procedure(g, S, γ, freq, short_term_memory; timelimit, max_iter, first_improvement)

# Local search procedure for MQCP. Corresponds to Algorithm 5.5 in thesis. 
# Returns the best found solution, the updated frequency list that tracks how many times each vertex was moved, 
# and optionally the swap history. 

# - `graph`: Input graph
# - `S`: Candidate solution
# - `γ`: Target density
# - `freq`: Frequency list, keeps track of how many times each vertex is moved
# - `short_term_memory`: Short term memory mechanism that blocks vertices from being moved in and out 
#     of the solution too often for a short time. 
# - `timelimit`: Cutoff time for search
# - `max_iter`: Stop search after `max_iter` iterations of no improvement
# - `first_improvement`: Strategy used for searching the neighborhood: If `first_improvement` is `true`, then 
#         the first improving neighboring solution will be selected, otherwise the neighborhood is always searched to 
#         completion and best improvement is used. 
# - `swap_history`: Swap history will be recorded, if a `SwapHistory` instance is passed.

# """
# function local_search_procedure(graph::SimpleGraph, S::Vector{Int}, γ::Real, freq::Vector{Int}, 
#                                 short_term_memory::ShortTermMemory; 
#                                 timelimit::Float64, max_iter::Int, next_improvement::Bool,
#                                 swap_history::Union{Nothing, SwapHistory})::Tuple{Vector{Int}, Vector{Int}, Union{Nothing, SwapHistory}}
#     k = length(S)
#     best_obj = calculate_num_edges(graph, S)
#     d_S = calculate_d_S(graph, S)
#     S = Set(S)
#     S′ = copy(S)
#     V_S = Set(filter(v -> v ∉ S, vertices(graph)))
#     current_obj = best_obj
#     min_edges_needed = γ * k * (k-1) / 2

#     iter_since_last_improvement = 0

#     reset!(short_term_memory)
    
#     while time() < timelimit && iter_since_last_improvement < max_iter
#         blocked = get_blocked(short_term_memory)
#         X_unblocked = filter(u -> u ∉ blocked, S)
#         Y_unblocked = filter(v -> v ∉ blocked, V_S)
#         d_min = minimum([d_S[i] for i in X_unblocked])
#         d_max = maximum([d_S[i] for i in Y_unblocked])
#         X_restricted = filter(u -> (d_S[u] <= d_min+1), X_unblocked)
#         Y_restricted = filter(v -> (d_S[v] >= d_max-1), Y_unblocked)

#         if !isempty(X_restricted) && !isempty(Y_restricted)
#             u, v, Δuv = search_neighborhood(graph, d_S, X_restricted, Y_restricted; next_improvement)
#             # TODO: if move is not improving, maybe use arbitrary move?
#         else 
#             u = sample(first_non_empty(X_restricted, X_unblocked, S))
#             v = sample(first_non_empty(Y_restricted, Y_unblocked, V_S))
#             Δuv = gain(graph, d_S, u, v)
#         end

#         # update S, V∖S
#         push!(S, v)
#         delete!(S, u)
#         push!(V_S, u)
#         delete!(V_S, v)

#         # update short term memory
#         move!(short_term_memory, u, v)
        
#         # update long term memory
#         freq[u] += 1
#         freq[v] += 1

#         # update swap SwapHistory
#         if !isnothing(swap_history)
#             push!(swap_history, u, v)
#         end
        
#         # update current candidate solution and corresponding data
#         current_obj += Δuv
#         update_d_S!(graph, d_S, u, v)
        
#         if current_obj > best_obj
#             S′ = S
#             best_obj = current_obj
#             iter_since_last_improvement = 0
#         else
#             iter_since_last_improvement += 1
#         end

#         if best_obj >= min_edges_needed
#             return collect(S′), freq, swap_history
#         end 
#     end
#     return collect(S′), freq, swap_history
# end

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
    calculate_d_S(g, candidate_solution)

Returns a vector d_S in the size of the vertex set of `vertices(g)`, where `d_S[i]` denotes the number of 
adjacent vertices in `S` for vertex `i` in `g`. 
    
- `g`: Input Graph
- `S`: Vector of vertices in `g`
"""
function calculate_d_S(g::SimpleGraph, S::Union{Vector{Int}, Set{Int}})
    d_S = Int[0 for _ in 1:nv(g)]
    for u in S
        for v in neighbors(g, u)
            d_S[v] += 1
        end
    end
    return d_S
end



end