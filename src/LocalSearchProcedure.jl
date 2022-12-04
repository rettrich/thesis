abstract type LocalSearchProcedure end

struct MQCP_LocalSearchProcedure <: LocalSearchProcedure
    γ::Real
    short_term_memory::ShortTermMemory
    scoring_function::ScoringFunction

    function MQCP_LocalSearchProcedure(γ::Real, short_term_memory::ShortTermMemory, scoring_function::ScoringFunction)
        new(γ, short_term_memory, scoring_function)
    end
end

abstract type NeighborhoodSearch end

"""
    VariableNeighborhoodDescent

Searches the neighborhoods Ω_1, Ω_2, ..., Ω_d in order and returns the first improving move. 

"""
struct VariableNeighborhoodDescent <: NeighborhoodSearch
    d::Int # search neighborhoods Ω_1, ... , Ω_d
end

struct Ω_1_NeighborhoodSearch <: NeighborhoodSearch end

"""
    VariableNeighborhoodDescent_SparseEvaluation

Searches the neighborhoods Ω_1, Ω_2, ..., Ω_d in order until an improving move is found. 
The move is then applied, the swapped vertices are removed from the neighborhood, and VND is performed again, 
without re-evaluating the scores for vertices by a GNN. This process is repeated until no improving move can be found.   

"""
struct VariableNeighborhoodDescent_SparseEvaluation <: NeighborhoodSearch
    d::Int # search neighborhoods Ω_1, ... , Ω_d
end

"""

    (local_search_procedure::MQCP_LocalSearchProcedure)(g, S, freq, short_term_memory; timelimit, max_iter, first_improvement)

Local search procedure for MQCP. Corresponds to Algorithm 5.5 in thesis. 
Returns the best found solution, the updated frequency list that tracks how many times each vertex was moved, 
and optionally the swap history. 

- `graph`: Input graph
- `S`: Candidate solution
- `freq`: Frequency list, keeps track of how many times each vertex is moved
- `short_term_memory`: Short term memory mechanism that blocks vertices from being moved in and out 
    of the solution too often for a short time. 
- `timelimit`: Cutoff time for search
- `max_iter`: Stop search after `max_iter` iterations of no improvement
- `first_improvement`: Strategy used for searching the neighborhood: If `first_improvement` is `true`, then 
        the first improving neighboring solution will be selected, otherwise the neighborhood is always searched to 
        completion and best improvement is used. 
- `swap_history`: Swap history will be recorded, if a `SwapHistory` instance is passed.
- `neighborhood_search`: Type of neighborhood search, e.g. Variable Neighborhood Descent 
- `sparse_evaluation`: Only re-evaluate by NN, if no improving swap could be found. 
- `score_based_sampling`: If no improving move can be found, sample a random move based on scores given by scoring function

"""
function (local_search_procedure::MQCP_LocalSearchProcedure)(
          graph::SimpleGraph, S::Union{Vector{Int}, Set{Int}}, freq::Vector{Int};
          timelimit::Float64, max_iter::Int, next_improvement::Bool,
          swap_history::Union{Nothing, SwapHistory}, neighborhood_search::NeighborhoodSearch,
          sparse_evaluation::Bool, score_based_sampling::Bool,
          is_mdcp::Bool=false,
          )::@NamedTuple{S::Vector{Int}, freq::Vector{Int}, swap_history::Union{Nothing, SwapHistory}}

    γ = local_search_procedure.γ
    short_term_memory = local_search_procedure.short_term_memory
    scoring_function = local_search_procedure.scoring_function
    
    k = length(S)
    best_obj = calculate_num_edges(graph, S)
    S = Set(S)
    S′ = copy(S)
    V_S = Set(filter(v -> v ∉ S, vertices(graph)))
    current_obj = best_obj
    if is_mdcp
        min_edges_needed = k * (k-1) / 2 - γ
    else
        min_edges_needed = γ * k * (k-1) / 2
    end

    reset!(short_term_memory, graph)
    update!(scoring_function, graph, S) 
    d_S = scoring_function.d_S

    if typeof(scoring_function) <: GNN_ScoringFunction && !isnothing(swap_history)
        # save feature matrix in swap_history
        swap_history.node_features = scoring_function.gnn_graph.ndata.x
    end

    iter_since_last_improvement = 0

    while time() < timelimit && iter_since_last_improvement < max_iter
        aspiration_val = best_obj - current_obj
        blocked = get_blocked(short_term_memory)
        X, Y = get_restricted_neighborhood(scoring_function, S, V_S)

        swap, Δ = search_neighborhood(neighborhood_search, graph, d_S, X, Y, blocked, aspiration_val; next_improvement)

        # no improving move could be found
        if Δ < 0 && Δ > -Inf
            # use move according to a scoring criterion
            if score_based_sampling
                X, Y = collect(X), collect(Y)
                scores = get_scores(scoring_function)
                X_scores = [(1 - scores[i]) for i in X]
                Y_scores = [scores[i] for i in Y]
                u = sample(X, Weights(softmax(X_scores)))
                v = sample(Y, Weights(softmax(Y_scores)))
            end
            # else just use the move with best Δ
        # there is no unblocked move
        elseif Δ == -Inf
            vec_S = collect(S)
            unblocked_S = filter(u -> u ∉ blocked , vec_S) 
            
            # try to use random unblocked vertex in open neighborhood of S, then 
            # random vertex in neighborhood  
            N_G_S = open_set_neighborhood(graph, vec_S)
            unblocked_N_G_S = filter(v -> v ∉ blocked, N_G_S)
            
            u = sample(first_non_empty(unblocked_S, vec_S))
            v = sample(first_non_empty(unblocked_N_G_S, setdiff(N_G_S, V_S), collect(V_S)))
            Δ = gain(graph, d_S, u, v)
            inside, outside = [u], [v]
            swap = (; inside, outside)
        end

        swaps_to_perform = Iterators.Stateful(zip(swap.inside, swap.outside))
        for (u, v) in swaps_to_perform
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

            # update swap SwapHistory
            if !isnothing(swap_history)
                push!(swap_history, u, v)
            end
            # update current candidate solution and corresponding data
            Δuv = gain(graph, d_S, u, v)
            @assert Δuv > -Inf
            current_obj += Δuv

            #update scoring function
            # in last iteration, evaluate scoring function (gnn) again
            # also, if sparse_evaluation is true, only evaluate if gain of swap was not positive
            evaluate = isnothing(peek(swaps_to_perform)) && (!sparse_evaluation || (Δ ≤ 0) )
            update!(scoring_function, u, v; evaluate)
            d_S = scoring_function.d_S
        end
        
        if current_obj > best_obj
            S′ = copy(S)
            best_obj = current_obj
            iter_since_last_improvement = 0
        else
            iter_since_last_improvement += 1
        end

        if best_obj >= min_edges_needed
            return (;S=collect(S′), freq, swap_history)
        end 
    end
    return (;S=collect(S′), freq, swap_history)
end

function search_neighborhood(::Ω_1_NeighborhoodSearch, 
                    graph::SimpleGraph, d_S, X, Y, blocked::Set{Int}, aspiration_val; 
                    next_improvement=true)
    inside, outside, Δ = search_neighborhood_Ω_1(graph, d_S, X, Y, blocked, aspiration_val; next_improvement)
    swap = (;inside, outside)
    return swap, Δ
end
"""
    search_neighborhood(vnd::VariableNeighborhoodDescent, graph::SimpleGraph, d_S, X, Y, blocked::Set{Int}, aspiration_val; next_improvement=true)

Searches the neighborhoods Ω_1, Ω_2, ..., Ω_d in order and returns the first improving move if `next_improvement` is true, 
otherwise returns the best unblocked move. Return value is (swaps, Δ_best), where swaps = (; inside, outside), inside and outside are vectors containing 
vertices in S and in V∖S, respectively, and Δ_best is the gain of edges relative to candidate solution S.  
In case all moves are blocked, and no blocked move is better than `aspiration_val`, (nothing, -Inf) is returned. 

"""
function search_neighborhood(vnd::VariableNeighborhoodDescent, 
                             graph::SimpleGraph, d_S, X, Y, blocked::Set{Int}, aspiration_val; 
                             next_improvement=true)
    d = vnd.d
    X, Y = collect(X), collect(Y)
    d_min_vals = partialsort([d_S[i] for i in X], 1:min(d, length(X)))
    d_max_vals = partialsort([d_S[i] for i in Y], 1:min(d, length(Y)); rev=true)
    Δ_best = -Inf
    swap = nothing
    d = vnd.d

    for i = 1:d
        if i > length(X) || i > length(Y)
            break
        end
        lower_bound = sum(d_max_vals[1:min(i, length(d_max_vals))]) - sum(d_min_vals[1:min(i, length(d_max_vals))]) - i^2
        upper_bound = lower_bound + i^2 + 2*binomial(i, 2)
        if upper_bound < 0
            continue # S is local optimum with respect to Ω_i, therefore skip searching the neighborhood
        end
        if i == 1
            inside, outside, Δ = search_neighborhood_Ω_1(graph, d_S, X, Y, blocked, aspiration_val; next_improvement)
        else
            inside, outside, Δ = search_neighborhood_Ω_d(graph, d_S, X, Y, blocked,  aspiration_val, lower_bound, i; next_improvement)
        end
        if Δ > Δ_best
            swap = (; inside, outside)
            Δ_best = Δ
        end
        if next_improvement && Δ_best > 0
            return swap, Δ_best
        end
    end
    
    if Δ_best > 0 && length(swap.inside) > 1
        @debug "Found improving move in Ω_i for i > 1"
    end
        
    return swap, Δ_best
end

"""
    search_neighborhood(vnd_sparse::VariableNeighborhoodDescent_SparseEvaluation, graph::SimpleGraph, d_S, X, Y, blocked::Set{Int}, aspiration_val; next_improvement=true)

Works similar to `VariableNeighborhoodDescent`, but if a move with positive gain is found by variable neighborhood descent, the neighborhood is searched again 
(without considering already swapped moves) until a local optimum is reached. All moves with a positive gain are then combined in the return value. 
If there is no move with positive gain, this method works exactly as the regular VariableNeighborhoodDescent. 

"""
function search_neighborhood(vnd_sparse::VariableNeighborhoodDescent_SparseEvaluation, 
                             graph::SimpleGraph, d_S, X, Y, blocked::Set{Int}, aspiration_val; 
                             next_improvement=true)
    inside, outside = [], []
    d_S = copy(d_S)
    vnd = VariableNeighborhoodDescent(vnd_sparse.d)
    Δ_total = 0
    first_iter = true
    while true
        swap, Δ_best = search_neighborhood(vnd, graph, d_S, X, Y, blocked, aspiration_val; next_improvement)

        if Δ_best < 0
            # no improving solution was found: 
            # if first iteration, then return best found swap. 
            # otherwise, a swap with positive gain has already been found
            if first_iter
                append!(inside, swap.inside)
                append!(outside, swap.outside)
                Δ_total += Δ_best
            end
            break
        else
            # add swaps to current swap list and update total gain Δ_total
            append!(inside, swap.inside)
            append!(outside, swap.outside)
            Δ_total += Δ_best
            
            # remove vertices that are already swapped and update d_S for further calculations
            X = setdiff(X, swap.inside)
            Y = setdiff(Y, swap.outside)
            blocked = setdiff(blocked, swap.inside, swap.outside)
            for (u,v) in zip(swap.inside, swap.outside)
                update_d_S!(graph, d_S, u, v)
            end
            first_iter = false
        end
    end
    swap = (; inside, outside)
    return swap, Δ_total
end

"""
    search_neighborhood_Ω_1(g, d_S, X, Y, blocked; next_improvement)

Search the neighborhood Ω_1 relative to candidate solution S defined by swapping a single vertex in X ⊆ S with a single vertex in Y ⊆ V ∖ S. 
If `next_improvement` is true, the first improving move is returned. Otherwise, the whole neighborhood is searched 
and the best move is returned. If no improving move can be found, the best non-improving move is returned. 
Vertices in `blocked` are blocked and will only be returned, if they are better than the current 
best solution (aspiration criterion). 
Returns a triple [u], [v], Δuv, where u ∈ X, v ∈ Y, and Δuv is the gain of edges in the candidate solution S. 

- `graph`: Input graph
- `d_S`: d_S[w] Contains number of neighboring vertices of w in S for each vertex in `g`, used to quickly compute the gain
- `X`: (Restricted) candidate list X ⊆ S
- `Y`: (Restricted) candidate list Y ⊆ V ∖ S
- `blocked`: A list of vertices that are blocked from being swapped, except they appear in a swap that fulfills the aspiration criterion
- `aspiration_val`: If a swap including a blocked vertex has a gain higher than `aspiration_val`, then it can be returned as the result 
    despite being blocked. 
- `next_improvement`: Determines whether the neighborhood is searched with next improvement or best improvement strategy. 

"""
function search_neighborhood_Ω_1(graph::SimpleGraph, d_S, X, Y, blocked::Set{Int}, aspiration_val; next_improvement=true)
    best = [], [], -Inf
    for u ∈ X, v ∈ Y
        Δuv = gain(graph, d_S, u, v)
        if Δuv <= aspiration_val && (u ∈ blocked || v ∈ blocked)
            continue
        end
        if Δuv > best[3]
            best = [u], [v], Δuv
        end
        if next_improvement && Δuv > 0
            return best
        end
    end
    return best
end

"""
    search_neighborhood_Ω_d(graph::SimpleGraph, d_S, X, Y, blocked::Set{Int}, aspiration_val::Int, initial_lower_bound, d::Int; next_improvement=true)

Search the neighborhood Ω_d relative to candidate solution S defined by swapping up to `d` vertices in X ⊆ S with `d` vertices in Y ⊆ V ∖ S. 
If `next_improvement` is true, the first improving move is returned. Otherwise, the whole neighborhood is searched 
and the best move is returned. If no improving move can be found, the best non-improving move is returned. 
Vertices in `blocked` are blocked and will only be returned, if they are better than the current 
best solution (aspiration criterion). 
Returns a triple [u], [v], Δuv, where u ∈ X, v ∈ Y, and Δuv is the gain of edges in the candidate solution S. 

- `graph`: Input graph
- `d_S`: d_S[w] Contains number of neighboring vertices of w in S for each vertex in `g`, used to quickly compute the gain
- `X`: (Restricted) candidate list X ⊆ S
- `Y`: (Restricted) candidate list Y ⊆ V ∖ S
- `blocked`: A list of vertices that are blocked from being swapped, except they appear in a swap that fulfills the aspiration criterion
- `aspiration_val`: If a swap including a blocked vertex has a gain higher than `aspiration_val`, then it can be returned as the result 
    despite being blocked. 
- `initial_lower_bound`: Provide lower bound for best found solution so far. 
- `d`: Maximum swap distance from candidate solution S
- `next_improvement`: Determines whether the neighborhood is searched with next improvement or best improvement strategy. 

"""
function search_neighborhood_Ω_d(graph::SimpleGraph, d_S, X, Y, blocked::Set{Int}, aspiration_val::Int, initial_lower_bound, d::Int; next_improvement=true)
    best = [], [], -Inf
    lower_bound = initial_lower_bound

    for inside in combinations(X, d), outside in combinations(Y, d)
        sum_outside = sum(d_S[outside[i]] for i in 1:d)
        sum_inside = sum(d_S[inside[i]] for i in 1:d)
        if sum_outside - sum_inside + 2*binomial(d, 2) < lower_bound
            continue
        end
        Δ = gain_S(graph, inside, outside, sum_outside, sum_inside)
        if Δ <= aspiration_val && (any([x ∈ blocked for x in inside]) || any([x ∈ blocked for x in outside]))
            continue
        end
        if Δ > max(best[3], lower_bound)
            best = inside, outside, Δ
            lower_bound = Δ
        end
        if next_improvement && Δ > 0
            return best
        end
    end
    return best
end

function gain_S(graph, X_i, Y_i, sum_outside, sum_inside)
    return sum_outside - sum_inside + edges_between(graph, X_i) + edges_between(graph, Y_i) - edges_between(graph, X_i, Y_i)
end

function edges_between(graph::SimpleGraph, nodes::Vector{Int})::Int
    count = 0
    for (u ,v) in combinations(nodes, 2)
        count += Int(has_edge(graph, u, v))
    end
    return count
end

function edges_between(graph::SimpleGraph, nodes_1::Vector{Int}, nodes_2::Vector{Int})::Int
    count = 0
    for u in nodes_1, v in nodes_2
        count += Int(has_edge(graph, u, v))
    end
    return count
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