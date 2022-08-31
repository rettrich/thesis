module LookaheadSearch

# using thesis.MPModels
using Graphs
using StatsBase

export LookaheadSearchFunction, Ω_1_LookaheadSearchFunction

abstract type LookaheadSearchFunction end

"""
    (::LookaheadSearchFunction)(graph::SimpleGraph, S::Set{Int}, num_solutions::Int)

Returns the best neighboring solution(s) `S′` of `S` in `graph` depending on the
concrete type of `LookaheadSearchFunction` along with its objective value
(number of edges in G[S′]). If there is more than one best neighboring solution,
up to `num_solutions` equal solutions are chosen randomly among the best solutions.
Neighboring solutions are returned as a vector of tuples `(in_nodes, out_nodes)`
in order to be space efficient; a neighboring solution can be reconstructed by
swapping the nodes in `in_nodes ∈ S` with the nodes in `out_nodes ∈ V ∖ S`.

"""
(::LookaheadSearchFunction)(graph::SimpleGraph{Int}, S::Set{Int}, d_S::Union{Vector{Int}, Nothing}=nothing, num_solutions=typemax(Int)::Int)::Tuple{Int, Vector{Tuple}} =
    error("Abstract (::LookaheadSearchFunction)(graph::SimpleGraph, S::Set{Int}) called")

"""
    Ω_1_LookaheadSearchFunction

Searches the Ω_1 neighborhood exhaustively and returns the neighboring solution(s) with
the highest objective value.

"""
struct Ω_1_LookaheadSearchFunction <: LookaheadSearchFunction end

Base.show(io::IO, ::Ω_1_LookaheadSearchFunction) = print(io, "Ω_1_LookaheadSearchFunction")

function (::Ω_1_LookaheadSearchFunction)(graph::SimpleGraph{Int}, S::Set{Int}, 
                                         d_S::Union{Vector{Int}, Nothing}=nothing, 
                                         num_solutions=typemax(Int)::Int
                                         )::Tuple{Int, Vector{Tuple}}
    d_S = isnothing(d_S) ? calculate_d_S(graph, S) : d_S
    obj_val = calculate_obj(graph, S, d_S)

    V_S = filter(v -> v ∉ S, vertices(graph))

    d_min = minimum([d_S[i] for i in S])
    d_max = maximum([d_S[i] for i in V_S])
    X = filter(u -> (d_S[u] <= d_min+1), S)
    Y = filter(v -> (d_S[v] >= d_max-1), V_S)

    Δuv_best = 0

    solutions = []

    # search restricted neighborhood for best solutions
    for u ∈ X, v ∈ Y
        Δuv = d_S[v] - d_S[u] - Int(has_edge(graph, u, v))
        if Δuv >= Δuv_best
            if Δuv > Δuv_best
                Δuv_best = Δuv
                solutions = [] # discard worse solutions
            end
            push!(solutions, ([u], [v])) # store solution as pair of nodes that need to be swapped, (u ∈ S, v ∈ V∖S)
        end
    end

    if Δuv_best == 0
        return obj_val, [] # S is local optimum, therefore do not return any neighboring solutions
    else
        if num_solutions < length(solutions)
            solutions = sample(1:length(solutions), num_solutions; replace=false)
        end
        return obj_val + Δuv_best, solutions
    end
end

# """
#     Ω_d_LookaheadSearchFunction

# Searches the Ω_d neighborhood exhaustively and returns the neighboring solution with
# the highest objective value, ties are broken randomly. For Ω_1 neighborhood, use
# `Ω_1_LookaheadSearchFunction`, as it is optimized for that specific neighborhood.

# """
# struct Ω_d_LookaheadSearchFunction <: LookaheadSearchFunction
#     d::Int

#     function Ω_d_LookaheadSearchFunction(d::Int)
#         new(d)
#     end
# end

# # TODO: revisit recursive search?
# function (::Ω_d_LookaheadSearchFunction)(graph::SimpleGraph, S::Set{Int})::Tuple{Int, Vector{Tuple{Vector{Int}, Vector{Int}}}}
#     error("Not implemented")
# end

# """
#     BeamSearch_LookaheadSearchFunction

# Interface for beam search lookahead search

# """
# struct BeamSearch_LookaheadSearchFunction <: LookaheadSearchFunction
#     d::Int
#     α::Int
#     β::Int

#     function BeamSearch_LookaheadSearchFunction(d::Int, α::Int, β::Int)
#         new(d, α, β)
#     end
# end

# (bs::BeamSearch_LookaheadSearchFunction)(graph::SimpleGraph, S::Set{Int}) =
#     beam_search(graph, S, bs.d; bs.α, bs.β)

##################################    Beam Search    ###############################################

"""
    Node

Node in the beam search tree of look-ahead search

- `obj_val`: Objective value of the solution that corresponds to this node.
- `in`: Set of nodes that must be added to the original candidate solution to obtain the solution corresponding to this node
- `out`: Set of nodes that must be removed from the original candidate solution to obtain the solution corresponding to this node
"""
struct Node
    obj_val::Int
    in::Set{Int} # set of nodes added to candidate solution
    out::Set{Int} # set of nodes removed from candidate solution
end

function Base.isless(a::Node, b::Node)
    return a.obj_val < b.obj_val
end

function Base.isequal(a::Node, b::Node)
    return issetequal(a.in, b.in) && issetequal(a.out, b.out)
end

Base.hash(a::Node, h::UInt) = hash(a.in, hash(a.out, hash(:Node, h)))

"""
    beam_search(g, candidate_solution, d; α, β)

Performs a beam search to find a neighboring solution that (approximately) maximizes the objective value.
`d` is the depth of the beam search (number of node swaps).

- `g`: `SimpleGraph`, input graph
- `candidate_solution`: Current candidate solution from which the search is started
- `d`: Number of swaps that are considered - depth of the beam search tree
- `α`: A node is expanded into at most `α`² child nodes: The worst α nodes in the candidate_solution
        and the best α nodes outside the candidate solution are considered for swapping
- `β`: Beam width. On each level of the beam search tree, only the best β nodes are kept.

"""
function beam_search(g::SimpleGraph, candidate_solution::Set{Int}, d::Int; α::Int=10, β::Int=10)
    candidate_solution = copy(candidate_solution) # set of nodes in candidate solution
    V_S = Set(setdiff(vertices(g), candidate_solution)) # remaining vertices in V(G)
    d_S = calculate_d_S(g, candidate_solution) # number of neighbors in candidate solution for each node
    obj = calculate_obj(g, candidate_solution, d_S) # objective value of candidate solution

    # to prevent duplicate solutions during search, store each node (defined by node.in and node.out) in hash map
    visited_solutions = Set{Node}()

    root = Node(obj, Set(), Set()) # root of BS tree
    max_node = root
    beam = [root]

    while !isempty(beam) && d > 0
        d -= 1
        children = Node[]
        for node in beam
            children = vcat(children, expand(g, candidate_solution, V_S, node, visited_solutions; α))
        end

        beam = partialsort(children, 1:min(β, length(children)), rev=true)

        if !isempty(beam)
            max_beam = beam[1]
            if max_beam > max_node
                max_node = max_beam
            end
        end
    end

    @debug "visited $(length(visited_solutions)) solutions"

    # obtain solution from node
    return get_solution(candidate_solution, max_node)
end

function get_solution(candidate_solution::Set{Int}, node::Node)
    for (u,v) in zip(node.in, node.out)
        push!(candidate_solution, u)
        delete!(candidate_solution, v)
    end
    return node.obj_val, candidate_solution
end

function update_candidate_solution!(candidate_solution::Set{Int}, V_S::Set{Int}, node::Node; rev::Bool=false)
    if !rev
        swap_history = zip(node.in, node.out)
    else
        swap_history = zip(node.out, node.in)
    end

    for (u,v) in swap_history
        push!(candidate_solution, u)
        push!(V_S, v)
        delete!(V_S, u)
        delete!(candidate_solution, v)
    end
end

"""
    expand(g, candidate_solution, V_S, node, visited_solutions; α)

Expand a node in the beam search into its children solutions that can be obtained by swapping
a node in the solution that corresponds to the node with a node outside of the solution.

- `g`: Graph instance
- `candidate_solution`: Candidate solution of root in BS tree
- `V_S`: Nodes outside the candidate solution of root in BS tree
- `node`: Current node, contains information to modify candidate solution to the solution
    corresponding to the current `node`
- `visited_solutions`: Hash table (Set) containing all visited nodes so no duplicates are considered
- `α`: Considere only up to α² swaps: Swap α worst nodes in current candidate solution with α
    best nodes outside
"""
function expand(g::AbstractGraph, candidate_solution::Set{Int},
                V_S::Set{Int}, node::Node, visited_solutions::Set{Node};
                α::Int=10) :: Vector{Node}
    # step into candidate solution
    update_candidate_solution!(candidate_solution, V_S, node)
    d_S = calculate_d_S(g, candidate_solution)

    restricted_V_S = filter(v -> v ∉ node.out, V_S)
    restricted_candidate_solution = filter(v -> v ∉ node.in, candidate_solution)

    d_S_V_S = [((i in restricted_V_S) ? d_S[i] : -1) for i=vertices(g)]
    d_S_candidate_solution = [((i in restricted_candidate_solution) ? d_S[i] : nv(g)) for i=vertices(g)]

    candidates_V_S = partialsortperm(d_S_V_S, 1:min(α, length(restricted_V_S)); rev=true)
    candidates_c_s = partialsortperm(d_S_candidate_solution, 1:min(α, length(restricted_candidate_solution)))

    expanded_children = Node[]

    for u in candidates_c_s, v in candidates_V_S
        gain = d_S[v] - d_S[u] - has_edge(g, u, v)
        in_set = Set([copy(node.in)..., v])
        out_set = Set([copy(node.out)..., u])
        child_node = Node(node.obj_val + gain, in_set, out_set)
        if child_node ∉ visited_solutions
            push!(visited_solutions, child_node)
            push!(expanded_children, child_node)
        end
    end

    update_candidate_solution!(candidate_solution, V_S, node; rev=true)
    return expanded_children
end

####################################################################################################

function calc_UB(d_S, d, k)
    UB = 0
    # max_d_v = maximum([d_S[j] for j in V_S])
    # min_d_u = minimum([d_S[j] for j in candidate_solution])
    max_total = maximum(d_S)
    min_total = minimum(d_S)
    for i=0:(d-1)
        # UB += min(max_d_v + i, k-1) - max(min_d_u - i, 0)
        UB += min(max_total + i, k-1) - max(min_total - i, 0)
    end
    return UB
end

#TODO: bug in branch and bound? UB might not be correct
function recursive_search(g, candidate_solution, d; swap_history=[], V_S=[], d_S = nothing, current_obj=0, best=([], 0)) :: Tuple{Vector{Tuple{Int, Int}}, Int}
    # intial call only
    if d_S === nothing
        d_S = calculate_d_S(g, candidate_solution)
        current_obj = calculate_obj(g, candidate_solution, d_S)
        best = Tuple{Int, Int}[], current_obj
        V_S = filter((v) -> (v ∉ candidate_solution), vertices(g))
    end

    # prune search
    UB = calc_UB(d_S, d, length(candidate_solution))
    if current_obj + UB <= best[2]
        return best
    end

    max_d_S = maximum(d_S[i] for i in V_S)
    min_d_S = minimum(d_S[i] for i in candidate_solution)

    for i in 1:eachindex(V_S)
        if d_S[V_S[i]] < max_d_S - 1
            continue
        end
        for j in 1:eachindex(candidate_solution)
            if d_S[candidate_solution[j]] > min_d_S + 1
                continue
            end
            # swap u, v
            u = candidate_solution[j]
            v = V_S[i]
            if (u,v) in swap_history || (v,u) in swap_history
                continue
            end

            candidate_solution[j] = v
            V_S[i] = u
            push!(swap_history, (u, v))

            # delta evaluate d_S and obj
            update_d_S!(g, u, v, d_S)
            new_obj = current_obj + d_S[v] - d_S[u] - Int(has_edge(g, u, v))

            if new_obj > best[2]
                best = copy(swap_history), new_obj
            end

            # TODO: call only for best ones
            # call recursive method
            if d > 1
                result = recursive_search(g, candidate_solution, d-1; swap_history, V_S, d_S, current_obj=new_obj, best)
                if result[2] > best[2]
                    best = copy(result[1]), result[2]
                end
            end
            # reverse operations
            candidate_solution[j] = u
            V_S[i] = v
            pop!(swap_history)
            update_d_S!(g, v, u, d_S)
        end
    end

    return best
end

function get_solution(candidate_solution, swap_history)
    S = Set(candidate_solution)

    for (u,v) in swap_history
        delete!(S, u)
        push!(S, v)
    end
    sort(collect(S))
end

function calculate_d_S(g::SimpleGraph, candidate_solution)
    d_S = Int[0 for _ in 1:nv(g)]
    for u in candidate_solution
        for v in neighbors(g, u)
            d_S[v] += 1
        end
    end
    return d_S
end

function calculate_obj(g, candidate_solution, d_S)
    num_edges = 0
    for u in candidate_solution
        num_edges += d_S[u]
    end
    num_edges = Int(num_edges / 2)
    return num_edges
end

function update_d_S!(g, u, v, d_S)
    for w in neighbors(g, u)
        d_S[w] -= 1
    end
    for w in neighbors(g, v)
        d_S[w] += 1
    end
end

# function num_to_edge(num::Int, n::Int)
#     i = 1
#     j = 1
#     k = n-1
#     while num > k
#         num -= k
#         k -= 1
#         i += 1
#     end
#     j = i + num
#     return i, j
# end

# function edge_to_num(e::Tuple{Int, Int}, n::Int)
#     u,v = e

#     u == 1 && return v-1

#     return (u-1)*(n) - sum(1:(u-1)) + v-u
# end

end