abstract type LocalSearchProcedure end

struct MQCP_LocalSearchProcedure <: LocalSearchProcedure
    γ::Real
    short_term_memory::ShortTermMemory
    scoring_function::ScoringFunction

    function MQCP_LocalSearchProcedure(γ::Real, short_term_memory::ShortTermMemory, scoring_function::ScoringFunction)
        new(γ, short_term_memory, scoring_function)
    end
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

"""
function (local_search_procedure::MQCP_LocalSearchProcedure)(
          graph::SimpleGraph, S::Union{Vector{Int}, Set{Int}}, freq::Vector{Int};
          timelimit::Float64, max_iter::Int, next_improvement::Bool,
          swap_history::Union{Nothing, SwapHistory}
          )::@NamedTuple{S::Vector{Int}, freq::Vector{Int}, swap_history::Union{Nothing, SwapHistory}}

    γ = local_search_procedure.γ
    short_term_memory = local_search_procedure.short_term_memory
    scoring_function = local_search_procedure.scoring_function

    reset!(short_term_memory, graph)
    update!(scoring_function, graph, S)
    
    k = length(S)
    best_obj = calculate_num_edges(graph, S)
    d_S = scoring_function.d_S
    S = Set(S)
    S′ = copy(S)
    V_S = Set(filter(v -> v ∉ S, vertices(graph)))
    current_obj = best_obj
    min_edges_needed = γ * k * (k-1) / 2

    iter_since_last_improvement = 0

    while time() < timelimit && iter_since_last_improvement < max_iter
        aspiration_val = best_obj - current_obj
        blocked = get_blocked(short_term_memory)
        X, Y = get_restricted_neighborhood(scoring_function, S, V_S)

        u, v, Δuv = search_neighborhood(graph, d_S, X, Y, blocked, aspiration_val; next_improvement)

        # if all moves are blocked and no move fulfills the aspiration criterion, use random move
        if Δuv == -Inf
            # does this happen?
            vec_S = collect(S)
            unblocked_S = filter(u -> u ∉ blocked , vec_S) 
            
            # try to use random unblocked vertex in open neighborhood of S, then 
            # random vertex in neighborhood  
            N_G_S = open_set_neighborhood(graph, vec_S)
            unblocked_N_G_S = filter(v -> v ∉ blocked, N_G_S)
            
            u = sample(first_non_empty(unblocked_S, vec_S))
            v = sample(first_non_empty(unblocked_N_G_S, setdiff(N_G_S, V_S), collect(V_S)))
            # TODO: use freq to sample vertex that was not moved as often?
            Δuv = gain(graph, d_S, u, v)
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

        # update swap SwapHistory
        if !isnothing(swap_history)
            push!(swap_history, u, v)
        end
        
        # update current candidate solution and corresponding data
        @assert Δuv > -Inf
        current_obj += Δuv

        #update scoring function
        update!(scoring_function, u, v)
        d_S = scoring_function.d_S
        
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

"""
    search_neighborhood(g, d_S, X, Y, blocked; next_improvement)

Search the neighborhood relative to candidate solution S defined by swapping vertices in X ⊆ S with vertices in Y ⊆ V ∖ S. 
If `next_improvement` is true, the first improving move is returned. Otherwise, the whole neighborhood is searched 
and the best move is returned. If no improving move can be found, the best non-improving move is returned. 
Vertices in `blocked` are blocked and will only be returned, if they are better than the current 
best solution (aspiration criterion). 
Returns a triple u, v, Δuv, where u ∈ X, v ∈ Y, and Δuv is the gain of edges in the candidate solution S. 

- `graph`: Input graph
- `d_S`: d_S[w] Contains number of neighboring vertices of w in S for each vertex in `g`, used to quickly compute the gain
- `X`: (Restricted) candidate list X ⊆ S
- `Y`: (Restricted) candidate list Y ⊆ V ∖ S
- `blocked`: A list of vertices that are blocked from being swapped, except they appear in a swap that fulfills the aspiration criterion
- `aspiration_val`: If a swap including a blocked vertex has a gain higher than `aspiration_val`, then it can be returned as the result 
    despite being blocked. 
- `next_improvement`: Determines whether the neighborhood is searched with next improvement or best improvement strategy. 

"""
function search_neighborhood(graph::SimpleGraph, d_S, X, Y, blocked::Set{Int}, aspiration_val; next_improvement=true)
    best = 0, 0, -Inf
    for u ∈ X, v ∈ Y
        Δuv = gain(graph, d_S, u, v)
        if Δuv <= aspiration_val && (u ∈ blocked || v ∈ blocked)
            continue
        end
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