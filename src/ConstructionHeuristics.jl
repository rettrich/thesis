using DataStructures

"""
    Node

A node in the beam search tree for `lower_bound_heuristic`.

- `S`: A feasible γ-clique representing this node
- `d_S`: Vector in the size of vertex set of the graph, d_S[v] holds number of neighbors of v in S
- `num_edges`: Number of edges in subgraph induced by S
- `h_val`: Heuristic value determined by guidance function 
"""
mutable struct Node
    S::Set{Int} # 
    d_S::Vector{Int} # d_S[v] holds number of neighbors of v in S 
    num_edges::Int # number of edges in subgraph induced by S 
    h_val::Real # heuristic value determined by guidance function
end

"""
    GuidanceFunction

A type for guidance functions for the beam search used in `lower_bound_heuristic`

"""
abstract type GuidanceFunction end

"""
    (::GuidanceFunction)(g::SimpleGraph, node::Node, γ::Real)

This function must be implemented by every subtype of `GuidanceFunction`

- `g`: Input graph
- `node`: `Node` in the beam search tree
- `γ`: Target density

"""
(::GuidanceFunction)(g::SimpleGraph, node::Node, γ::Real)::Real = error("Abstract implementation called")

"""
    GreedyCompletionHeuristic

Returns the size of a solution that is completed greedily by always picking the node outside the 
current solution that maximizes density.  
"""
struct GreedyCompletionHeuristic <: GuidanceFunction end

function (::GreedyCompletionHeuristic)(g::SimpleGraph, node::Node, γ::Real)::Real
    S′ = copy(node.S)
    k = length(S′)
    d_S = copy(node.d_S)
    num_edges = node.num_edges

    # guaranteed complexity of O(n) per iteration, number of iterations dependent on solution size
    while true
        k = length(S′)
        k == nv(g) && break
        min_edges_needed = γ * (k*(k+1)/2) # edges needed for feasibility in clique of size k+1

        V_S = filter(v -> v ∉ S′, vertices(g))
        v_max = V_S[argmax([d_S[i] for i in V_S])] # vertex in V ∖ S that maximizes d_S

        if d_S[v_max] + num_edges < min_edges_needed
            break
        else
            push!(S′, v_max)
            num_edges = num_edges + d_S[v_max]
            for u in neighbors(g, v_max)
                d_S[u] += 1
            end
        end
    end

    #approx_density = 0.999 * density(induced_subgraph(g, S′)[1])

    return length(S′)# + approx_density
end

"""
    GreedyCompletionHeuristicPQVariant

Same as `GreedyCompletionHeuristic`, but uses a priority queue to efficiently identify maximum element, 
with worse worst time complexity, but maybe better practical performance. 

TODO/Update: performs worse, as an improvement remove construction of PriorityQueue, and use a field Node.pq, 
as construction of the PQ is too slow -> store pq with node and only update each time a vertex is added

"""
struct GreedyCompletionHeuristicPQVariant <: GuidanceFunction end

function (::GreedyCompletionHeuristicPQVariant)(g::SimpleGraph, node::Node, γ::Real)::Real
    S′ = copy(node.S)
    k = length(S′)
    d_S = copy(node.d_S)
    V_S = Set(filter(v -> v ∉ S′, vertices(g)))

    # make PQ from keys in V_S and values in d_S
    # construction takes O(n log n) time
    pq = PriorityQueue{Int, Int}(Base.Order.Reverse)
    for v in V_S
        pq[v] = d_S[v]
    end

    num_edges = node.num_edges

    while true
        k = length(S′)
        k == nv(g) && break
        min_edges_needed = γ * (k*(k+1)/2) # edges needed for feasibility in clique of size k+1

        v_max, d_max = dequeue_pair!(pq)

        if d_max + num_edges < min_edges_needed
            break
        else
            push!(S′, v_max)
            delete!(V_S, v_max)
            num_edges = num_edges + d_max

            # this step has a worst case complexity of O(n log n), 
            # which is worse than the other variant (GreedyCompletionHeuristic). 
            # in practice hopefully must neighbors of v_max are in S and not in V_S, 
            # therefore the average case could (should) perform better than the other variant.
            for u in neighbors(g, v_max)
                d_S[u] += 1
                if u ∈ V_S
                    pq[u] = d_S[u] # update priority in pq
                end
            end
        end
    end


    return length(S′)
end

"""
    SumOfNeighborsHeuristic

Evaluates a node by the `d` highest values of vertices in V ∖ S in d_S 
(d_S[i] holds the number of neighbors in candidate solution S for vertex i). 
Additionally takes into account the density of a candidate solution.

- `d`: Number of nodes to be considered
- `weight_num_edges`: ∈[0,1] Density is weighted by this amount, while the sum of neighbors is 
    weighted by 1-`weight_num_edges`
"""
struct SumOfNeighborsHeuristic <: GuidanceFunction 
    d::Int
    weight_num_edges::Real
end

function (heu::SumOfNeighborsHeuristic)(g::SimpleGraph, node::Node, γ::Real)::Real
    V_S = filter(v -> v ∉ node.S, vertices(g))
    d_S_filtered = [node.d_S[i] for i in V_S]

    num_edges = node.num_edges
    sum_of_neighbors = sum(partialsort(d_S_filtered, 1:min(heu.d, length(d_S_filtered)); rev=true))

    return heu.weight_num_edges * num_edges + (1 - heu.weight_num_edges) * sum_of_neighbors
end

"""
    lower_bound_heuristic(g, γ, guidance_function; β, expansion_limit)

Beam search construction that returns a feasible `γ`-quasi clique in `g`. 
Each node of the beam search tree is expanded into at most `expansion_limit` nodes, 
and the beamwidth is defined by `β`

- `g`: Input graph
- `γ`: Target density
- `guidance_function`: Guidance function used to evaluate nodes in the beam search tree
- `β`: Beam width, at most β nodes on each level of the beam search tree are pursued further
- `expansion_limit`: A node in the beam search tree is expanded into at most `expansion_limit` successor nodes

"""
function lower_bound_heuristic(g::SimpleGraph, γ::Real, guidance_function::GuidanceFunction; β=10, expansion_limit=Inf)
    root = Node(Set(), fill(0, nv(g)), 0, 0, )
    max_node = root
    beam = [root]
    level::Int = 0

    while !isempty(beam)
        level = level + 1
        @debug "level", level
        
        children = []
        visited_solutions = Set{Set{Int}}()

        # ignore expansion limit only for first node
        for node in beam
            if level == 1
                children = vcat(children, expand(g, node, γ, visited_solutions))
            else
                children = vcat(children, expand(g, node, γ, visited_solutions; expansion_limit))
            end
        end
        if !isempty(children)
            max_node = sample(children)
        end

        filter!(node -> !is_terminal(g, γ, node), children)

        for node in children
            node.h_val = guidance_function(g, node, γ)
        end

        beam = partialsort(children, 1:min(β, length(children)); by=(node -> node.h_val), rev=true)
    end

    return collect(max_node.S)
end

# expand node into feasible successors
function expand(g::SimpleGraph, node::Node, γ::Real, visited_solutions::Set{Set{Int}}; expansion_limit=Inf)::Vector{Node}
    d_S = node.d_S
    k = length(node.S)
    min_edges_needed = Int(ceil(γ * (k*(k+1)/2))) - node.num_edges
    d_S_sorted_perm = filter(v -> v ∉ node.S, sortperm(d_S; rev=true))

    children = Vector{Node}()
    
    for v in d_S_sorted_perm
        if d_S[v] >= min_edges_needed
            S′ = Set([node.S..., v])
            if S′ ∉ visited_solutions
                push!(visited_solutions, S′)
                push!(children, Node(S′, update_d_S(g, d_S, v), node.num_edges + d_S[v], 0))
                if length(children) > expansion_limit
                    break
                end
            end
        else
            break
        end
    end
    return children    
end

function update_d_S(g, d_S, v; rev=false)
    d_S = copy(d_S)
    sign = rev ? -1 : 1
    for u in neighbors(g, v)
        d_S[u] = d_S[u] + sign
    end
    return d_S
end


"""
    is_terminal(g, γ, S)

Returns true if candidate solution `S` can be extended to a γ-quasi clique of size |`S`|+1, and 
false otherwise.

- `g`: Input Graph
- `γ`: Target density that defines feasibility of a quasi-clique
- `node`: Node in the beam search tree

"""
function is_terminal(g::SimpleGraph, γ::Real, node::Node)
    V_S = setdiff(vertices(g), node.S)
    d_S_filtered = [node.d_S[i] for i in V_S]
    k = length(node.S)
    edges_needed = γ*(k*(k+1)/2) # minimum number of edges needed in solution of size |S|+1 

    # if clique cannot be extended by a single node, it is terminal -> return true
    if maximum(d_S_filtered) + node.num_edges < edges_needed
        return true
    else
        return false
    end
end

"""
    construction_heuristic(g, k, freq; α, p)

Corresponds to Algorithm 5.4 in thesis. Returns a vector of distinct vertices in `g` of size `k` that is build 
incrementally. In each iteration, with probability `p` a vertex with low frequency value is added, 
and with probability 1-p a vertex is added in a GRASP-like manner.

- `g`: Input Graph
- `k`: Target size of the returned vector of vertices
- `freq`: Frequency list with length |V|. Vertices with low frequency are preferred during construction.
- `α`: GRASP parameter; α=0 performs a greedy construction, α=1 performs a randomized construction
- `p`: Controls the balance between GRASP construction and preferring vertices with low frequency values. 
    `p`=0 ignores frequency values, while `p`=1 only uses frequency values.

"""
function construction_heuristic(g::SimpleGraph, k::Int, freq=[0 for i=1:nv(g)]::Vector{Int}; α=0.2::Real, p=0.2::Real)
    freq_sorted = sortperm(freq)
    init_vertex = freq_sorted[1]
    S = [init_vertex]
    d_S = calculate_d_S(g, S)

    while length(S) < k
        if rand() < p
            N_G_S = open_set_neighborhood(g, S)
            if !isempty(N_G_S)
                u = filter(v -> v ∈ N_G_S, freq_sorted)[1]
            else
                V_S = setdiff(vertices(g), S)
                u = filter(v -> v ∈ V_S, freq_sorted)[1]
            end
        else
            V_S = setdiff(vertices(g), S) # V ∖ candidate_solution
            d_S_V_S = [d_S[i] for i in V_S] # only d_S values for V_S
            d_max = maximum(d_S_V_S)
            d_min = minimum(d_S_V_S)
            min_val = d_max - α*(d_max - d_min)
            restricted_candidate_list = filter(v -> d_S[v] >= min_val, V_S)
            u = sample(restricted_candidate_list)
        end
        push!(S, u)
        for v in neighbors(g, u)
            d_S[v] += 1
        end
    end

    return S
end

"""
    open_set_neighborhood(g, vertex_list)

Return the union of neighborhoods of vertices in `vertex_list` excluding vertices in `vertex_list`

- `g`: Input Graph
- `vertex_list`: Vector of vertices from `g`

"""
open_set_neighborhood(g::SimpleGraph, vertex_list::Vector{Int})::Vector{Int} = 
    setdiff(reduce(vcat, neighbors(g, v) for v in vertex_list), vertex_list)


open_set_neighborhood(g::SimpleGraph, vertex_set::Set{Int})::Vector{Int} = open_set_neighborhood(g, collect(vertex_set))

"""
    calculate_num_edges(g, candidate_solution)

Returns the number of edges in the subgraph induced by candidate solution `S` in `g`

- `g`: Input Graph
- `S`: List of nodes representing the candidate solution

"""
calculate_num_edges(g::SimpleGraph, S::Vector{Int})::Int = ne(induced_subgraph(g, S)[1])
