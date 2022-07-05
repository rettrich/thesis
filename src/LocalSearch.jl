module LocalSearch

using StatsBase
using Graphs
using thesis.LookaheadSearch

export local_search_with_EX

include("ConstructionHeuristics.jl")


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

function calculate_d_S(g, candidate_solution)
    d_S = Int[0 for _ in 1:nv(g)]
    for u in candidate_solution
        for v in neighbors(g, u)
            d_S[v] += 1
        end
    end
    return d_S
end

end