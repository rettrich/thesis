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

struct GreedySearchHeuristic <: GuidanceFunction end

function (::GreedySearchHeuristic)(g::SimpleGraph, node::Node, γ::Real)::Real
    return node.num_edges
end

# returns random value for each node
struct RandomHeuristic <: GuidanceFunction end

function (::RandomHeuristic)(g::SimpleGraph, node::Node, γ::Real)::Real
    return rand()
end

struct FeasibleNeighborsHeuristic <: GuidanceFunction 
    variant_a::Bool
end

function (noeh::FeasibleNeighborsHeuristic)(g::SimpleGraph, node::Node, γ::Real)::Real
    k = length(node.S)
    min_edges_needed = ceil(Int, γ * (k*(k+1)/2)) - node.num_edges # edges needed for feasibility in clique of size k+1
    d_S = [node.d_S[v] for v in vertices(g) if v ∉ node.S] # d_S values for vertices outside S
    result = 0
    for val in d_S
        if val >= min_edges_needed
            result += 1
            add = noeh.variant_a ? ((val - min_edges_needed)/(k*(k+1)/2)) : (val - min_edges_needed)
            result += add
        end
    end 
    return result
end

struct MDCP_FeasibleNeighborsHeuristic <: GuidanceFunction 
    variant_a::Bool
end

function (noeh::MDCP_FeasibleNeighborsHeuristic)(g::SimpleGraph, node::Node, γ::Real)::Real
    k = length(node.S)
    min_edges_needed = (k*(k+1)/2) - γ - node.num_edges # edges needed for feasibility in clique of size k+1
    d_S = [node.d_S[v] for v in vertices(g) if v ∉ node.S] # d_S values for vertices outside S
    result = 0
    for val in d_S
        if val >= min_edges_needed
            result += 1
            add = noeh.variant_a ? ((val - min_edges_needed)/(k*(k+1)/2)) : (val - min_edges_needed)
            result += add
        end
    end 
    return result
end

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
    LowerBoundHeuristic

Abstract type for LowerBoundHeuristic in the Local Search based Metaheuristic
Used to find a feasible solution quickly that can be used as a lower bound. 

"""
abstract type LowerBoundHeuristic end

(::LowerBoundHeuristic)(graph::SimpleGraph) = 
    error("Abstract method (::LowerBoundHeuristic)(graph::SimpleGraph) called")

"""
    SingleVertex_LowerBoundHeuristic

Return a single vertex (with index 1) as the lower bound solution or an empty solution 
if the graph is empty
"""
struct SingleVertex_LowerBoundHeuristic <: LowerBoundHeuristic end

(::SingleVertex_LowerBoundHeuristic)(graph::SimpleGraph) = (nv(graph) > 0 ? [1] : [])

"""
    BeamSearch_LowerBoundHeuristic

Type for a beam search heuristic for the MQCP that returns a feasible `γ`-quasi clique 
when applied to a graph. Each node of the beam search tree is expanded into at most 
`expansion_limit` nodes, and the beamwidth of the search is defined by `β`.

- `guidance_func`: Guidance function used to evaluate nodes in the beam search
- `β`: Beam width
- `γ`: Target density for MQCP
- `expansion_limit`: Limit expansion of nodes into up to `expansion_limit` child nodes for performance

"""
struct BeamSearch_LowerBoundHeuristic <: LowerBoundHeuristic
    guidance_func::GuidanceFunction
    β::Int
    γ::Real
    expansion_limit::Int
    is_mdcp::Bool

    """
    
        BeamSearch_LowerBoundHeuristic
    
    - `guidance_func`: GuidanceFunction used to evaluate nodes in the beam search
    - `β`: Beam width
    - `γ`: Target density for MQCP
    - `expansion_limit`: Limit expansion of nodes into up to `expansion_limit` child nodes for performance

    """
    function BeamSearch_LowerBoundHeuristic(guidance_func::GuidanceFunction; β::Int=5, γ::Real, expansion_limit::Int=50, is_mdcp::Bool=false)
        new(guidance_func, β, γ, expansion_limit, is_mdcp)
    end
end

"""
    (bs_lbh::BeamSearch_LowerBoundHeuristic)(graph::SimpleGraph)

Beam search heuristic that returns a feasible `γ`-quasi clique in `graph`. 
Each node of the beam search tree is expanded into at most `expansion_limit` nodes, 
and the beamwidth is defined by `β`. Corresponds to Algorithm 5.3 in thesis.

- `graph`: Input graph

"""
function (bs_lbh::BeamSearch_LowerBoundHeuristic)(graph::SimpleGraph)
    β = bs_lbh.β
    expansion_limit = bs_lbh.expansion_limit
    γ = bs_lbh.γ
    guidance_function = bs_lbh.guidance_func

    root = Node(Set(), fill(0, nv(graph)), 0, 0, )
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
                children = vcat(children, expand(graph, node, γ, visited_solutions; bs_lbh.is_mdcp))
            else
                children = vcat(children, expand(graph, node, γ, visited_solutions; expansion_limit, bs_lbh.is_mdcp))
            end
        end
        if !isempty(children)
            max_node = sample(children)
        end

        filter!(node -> !is_terminal(graph, γ, node; bs_lbh.is_mdcp), children)

        for node in children
            node.h_val = guidance_function(graph, node, γ)
        end

        beam = partialsort(children, 1:min(β, length(children)); by=(node -> node.h_val), rev=true)
    end

    return collect(max_node.S)
end

# expand node into feasible successors
function expand(graph::SimpleGraph, node::Node, γ::Real, visited_solutions::Set{Set{Int}}; expansion_limit=Inf, is_mdcp=false)::Vector{Node}
    d_S = node.d_S
    k = length(node.S)
    if is_mdcp
        min_edges_needed = Int((k*(k+1)/2)-γ) - node.num_edges
    else
        min_edges_needed = Int(ceil(γ * (k*(k+1)/2))) - node.num_edges
    end
    d_S_sorted_perm = filter(v -> v ∉ node.S, sortperm(d_S; rev=true))

    children = Vector{Node}()
    
    for v in d_S_sorted_perm
        if d_S[v] >= min_edges_needed
            S′ = Set([node.S..., v])
            if S′ ∉ visited_solutions
                push!(visited_solutions, S′)
                push!(children, Node(S′, update_d_S(graph, d_S, v), node.num_edges + d_S[v], 0))
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

function update_d_S(graph, d_S, v; rev=false)
    d_S = copy(d_S)
    sign = rev ? -1 : 1
    for u in neighbors(graph, v)
        d_S[u] = d_S[u] + sign
    end
    return d_S
end


"""
    is_terminal(graph, γ, S)

Returns true if candidate solution `S` can be extended to a γ-quasi clique of size |`S`|+1, and 
false otherwise.

- `graph`: Input Graph
- `γ`: Target density that defines feasibility of a quasi-clique
- `node`: Node in the beam search tree

"""
function is_terminal(graph::SimpleGraph, γ::Real, node::Node; is_mdcp::Bool=false)
    V_S = setdiff(vertices(graph), node.S)
    d_S_filtered = [node.d_S[i] for i in V_S]
    k = length(node.S)
    if is_mdcp
        edges_needed = k*(k+1)/2 - γ
    else
        edges_needed = γ*(k*(k+1)/2) # minimum number of edges needed in solution of size |S|+1 
    end

    # if clique cannot be extended by a single node, it is terminal -> return true
    if maximum(d_S_filtered) + node.num_edges < edges_needed
        return true
    else
        return false
    end
end