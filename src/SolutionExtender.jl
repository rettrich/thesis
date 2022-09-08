"""
    SolutionExtender

When applied to a graph and a feasible solution `S`, the SolutionExtender tries to find a solution 
`S′` of maximum cardinality such that `S ⊆ S′`. 
"""
abstract type SolutionExtender end

extend(::SolutionExtender, graph::SimpleGraph, S::Vector{Int}) = 
    error("SolutionExtender: Abstract extend called")

struct MQCP_GreedySolutionExtender <: SolutionExtender
    γ::Real
end

# TODO: Add bfs variant? (exhaustive search instead of greedy search)
#       or BS variant?
"""
    extend(solution_extender, graph, S)

Greedily extends solution `S` by adding the vertex outside `S` with maximum number of neighbors in `S` 
if this produces a valid γ-quasi clique. 

- `solution_extender`: `MQCP_SolutionExtender` instance that stores problem specific data
- `graph`: Input graph
- `S`: Feasible γ-quasi clique to extend

"""
function extend(solution_extender::MQCP_GreedySolutionExtender, graph::SimpleGraph, S::Vector{Int})
    γ = solution_extender.γ
    S = copy(S)
    V_S = Set(setdiff(vertices(graph), S))
    d_S = calculate_d_S(graph, S)
    num_edges = calculate_num_edges(graph, S)
    k = length(S)

    while true
        min_edges_needed = ceil(Int, γ * k * (k+1) / 2)
        for v in V_S
            if num_edges + d_S[v] >= min_edges_needed
                delete!(V_S, v)
                push!(S, v)
                num_edges = num_edges + d_S[v]
                for u in neighbors(graph, v)
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