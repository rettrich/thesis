module LocalSearch

using StatsBase
using Graphs
using thesis.LookaheadSearch

export local_search_with_EX

#############################################################################################
############################   Construction Heuristics   ####################################
#############################################################################################

function grasp(g, k; α=0.25)
    V = Set(vertices(g))
    min_val = Δ(g) - α*(Δ(g) - δ(g))
    restricted_candidate_set = filter(v -> degree(g,v) >= min_val, V)

    candidate_solution = Set([sample(collect(restricted_candidate_set))])

    while length(candidate_solution) < k
        d_S = calculate_d_S(g, candidate_solution)
        V_S = setdiff(V, candidate_solution)
        d_S_V_S = [d_S[i] for i in V_S]
        max_d_S_V_S = maximum(d_S_V_S)
        min_val = max_d_S_V_S - α*(max_d_S_V_S - minimum(d_S_V_S))
        restricted_candidate_set = filter(v -> d_S[v] >= min_val, V_S)
        new_vertex = sample(collect(restricted_candidate_set))
        push!(candidate_solution, new_vertex)
    end

    return candidate_solution
end

#############################################################################################
############################    Local Search Variants    ####################################
#############################################################################################


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