"""
    SwapHistory

An instance of `SwapHistory` is initialized with a graph. It is meant to record all candidate solutions encountered 
during the execution of the local search algorithm. This is done by storing an initial solution returned by the 
construction heuristic as a set in `initial_solution_vec`, and then storing all swaps `(u,v)` as tuples in order in 
`swap_list_vec`, where `u` is a vertex in S, and `v` is a vertex in V∖S. Encountered candidate solutions can then 
be reconstructed by doing the swaps in swap_list_vec in order.

"""
struct SwapHistory
    graph::SimpleGraph
    initial_solution_vec::Vector{Set{Int}}
    swap_list_vec::Vector{Vector{Tuple{Int, Int}}}

    function SwapHistory(graph::SimpleGraph)
        new(graph, [], [])
    end

end

"""
    length(sh::SwapHistory)

Returns the total number of encountered candidate solutions. 

"""
function Base.length(sh::SwapHistory)::Int 
    len = 0
    for i in 1:length(sh.initial_solution_vec)
        len += 1
        len += length(sh.swap_list_vec[i])
    end
    return len
end

function Base.push!(sh::SwapHistory, initial_solution::Set{Int}) 
    push!(sh.initial_solution_vec, initial_solution)
    if length(sh.swap_list_vec) < length(sh.initial_solution_vec)
        push!(sh.swap_list_vec, [])
    end
end

# push a swap to the history belonging to the latest initial solution
function Base.push!(sh::SwapHistory, u::Int, v::Int)
    idx = length(sh.initial_solution_vec)
    push!(sh.swap_list_vec[idx], (u,v))
end

"""
    get(sh::SwapHistory, i::Int)

Returns the `i`-th candidate solution recorded in this `SwapHistory`. 

"""
function get_candidate_solution(sh::SwapHistory, i::Int)::Set{Int}
    if i > length(sh)
        throw(ArgumentError("Index $i too large: SwapHistory contains only $(length(sh)) elements"))
    end

    sh_idx = 1

    # find initial solution that belongs to this index
    while length(sh.swap_list_vec[sh_idx])+1 < i
        i -= length(sh.swap_list_vec[sh_idx])+1
        sh_idx += 1
    end

    # construct candidate solution from initial solution by performing swaps
    S = copy(sh.initial_solution_vec[sh_idx])
    
    # initial solution has index 1, remaining indices are in sh.swap_list_vec[sh_idx]
    # therefore sh.swap_list_vec[sh_idx] contains length(sh.swap_list_vec[sh_idx])+1 
    # candidate solutions
    i -= 1
    for idx in 1:i
        # a swap is a tuple (u,v), where u is in S, and v is in V∖S
        # to recreate the swap, remove u from S and add v to S
        u, v = sh.swap_list_vec[sh_idx][idx]
        push!(S, v)
        delete!(S, u)
    end
    return S
end

"""
    sample_candidate_solutions(sh, n)

Samples `n` candidate solutions uniformly without replacement from `sh` and 
returns it as a `NamedTuple{graph::SimpleGraph, samples::Vector{Set{Int}}}`. 

"""
function sample_candidate_solutions(sh::SwapHistory, n::Int)
    res = @NamedTuple{graph::SimpleGraph, samples::Vector{Set{Int}}}((sh.graph, []))
    sample_indices = sort(sample(1:length(sh), n; replace=false))

    for i in sample_indices
        S = get_candidate_solution(sh, i)
        push!(res.samples, S)
    end
    return res
end