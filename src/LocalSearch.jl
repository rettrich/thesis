module LocalSearch

using StatsBase
using Graphs
using Combinatorics
using Flux

export calculate_d_S, calculate_num_edges,
    LocalSearchBasedMH,
    LowerBoundHeuristic, BeamSearch_LowerBoundHeuristic, SingleVertex_LowerBoundHeuristic,
    ConstructionHeuristic, Freq_GRASP_ConstructionHeuristic,
    ConfigurationChecking, TabuList,
    LocalSearchProcedure, MQCP_LocalSearchProcedure,
    GuidanceFunction, GreedyCompletionHeuristic, GreedyCompletionHeuristicPQVariant, 
    SumOfNeighborsHeuristic, FeasibleNeighborsHeuristic, MDCP_FeasibleNeighborsHeuristic, RandomHeuristic, GreedySearchHeuristic, 
    SwapHistory, sample_candidate_solutions,
    ScoringFunction, d_S_ScoringFunction, SimpleGNN_ScoringFunction, Random_ScoringFunction, Encoder_Decoder_ScoringFunction, 
    SolutionExtender, MQCP_GreedySolutionExtender, MDCP_GreedySolutionExtender, 
    FeasibilityChecker, MQCP_FeasibilityChecker, MDCP_FeasibilityChecker, 
    NeighborhoodSearch, VariableNeighborhoodDescent, VariableNeighborhoodDescent_SparseEvaluation, Ω_1_NeighborhoodSearch, 
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
- a `NeighborhoodSearch` that searches the neighborhoods of candidate solutions during the local search procedure (e.g. VariableNeighborhoodDescent)

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
    neighborhood_search::NeighborhoodSearch

    # settings
    timelimit::Float64
    max_iter::Int
    next_improvement::Bool
    record_swap_history::Bool
    max_restarts::Float32
    sparse_evaluation::Bool
    score_based_sampling::Bool
    is_mdcp::Bool

    function LocalSearchBasedMH(lower_bound_heuristic::LowerBoundHeuristic, 
                                construction_heuristic::ConstructionHeuristic,
                                local_search_procedure::LocalSearchProcedure,
                                feasibility_checker::FeasibilityChecker,
                                solution_extender::SolutionExtender; 
                                neighborhood_search = Ω_1_NeighborhoodSearch(),
                                timelimit::Float64 = 600.0, # 10 minutes
                                max_iter::Int = 1000, 
                                next_improvement::Bool = true,
                                record_swap_history::Bool = false,
                                max_restarts = Inf,
                                sparse_evaluation::Bool = false,
                                score_based_sampling::Bool = false,
                                is_mdcp::Bool = false,
                                )
        new(lower_bound_heuristic, construction_heuristic, local_search_procedure, feasibility_checker, solution_extender, neighborhood_search,
            timelimit, max_iter, next_improvement, record_swap_history, max_restarts, sparse_evaluation, score_based_sampling, is_mdcp)
    end
end

function run_lsbmh(local_search::LocalSearchBasedMH, graph::SimpleGraph)::@NamedTuple{solution::Vector{Int}, swap_history::Union{Nothing, SwapHistory}}
    # initial construction
    @debug "Constructing initial solution..."
    S′ = local_search.lower_bound_heuristic(graph)

    @debug "Found initial solution of size $(length(S′))"
    k = length(S′)+1
    freq = fill(0, nv(graph))
    timelimit = time() + local_search.timelimit

    swap_history::Union{Nothing, SwapHistory} = local_search.record_swap_history ? SwapHistory(graph) : nothing

    restarts = 0

    while time() < timelimit && restarts <= local_search.max_restarts
        @debug "Generate candidate solution for size $k, attempt: $restarts"
        S = local_search.construction_heuristic(graph, k, freq)

        if !isnothing(swap_history)
            push!(swap_history, Set(S))
        end

        @debug "Starting local search with candidate solution of density $(density_of_subgraph(graph, S))"
        S, freq, swap_history = local_search.local_search_procedure(graph, S, freq;
                    local_search.neighborhood_search, timelimit, local_search.max_iter, 
                    local_search.next_improvement, swap_history, local_search.sparse_evaluation,
                    local_search.score_based_sampling, local_search.is_mdcp)
        
        @debug "Found solution with density $(density_of_subgraph(graph, S))"

        restarts += 1

        if local_search.feasibility_checker(graph, S)
            S = extend(local_search.solution_extender, graph, S)
            S′ = S
            k = length(S′)+1
            freq = fill(0, nv(graph))
            restarts = 0
        end
    end
    return (;solution=S′, swap_history)
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