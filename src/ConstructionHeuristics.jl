using StatsBase

abstract type GuidanceFunction end

(::GuidanceFunction)(g::SimpleGraph, S::Vector{Int}, γ::Real)::Real = error("Abstract implementation called")

struct GreedyCompletionHeuristic <: GuidanceFunction end

function (::GreedyCompletionHeuristic)(g::SimpleGraph, S::Vector{Int}, γ::Real)::Real
    S′ = copy(S)
    k = length(S′)
    d_S = calculate_d_S(g, S′)
    num_edges = calculate_num_edges(g, S′)

    while true
        k = length(S′)
        k == nv(g) && break
        min_edges_needed = γ * (k*(k+1)/2) # edges needed for feasibility in clique of size k+1

        d_S_sorted_perm = filter(v -> v ∉ S′, sortperm(d_S; rev=true))

        if d_S[d_S_sorted_perm[1]] + num_edges < min_edges_needed
            break
        else
            push!(S′, d_S_sorted_perm[1])
            num_edges = num_edges + d_S[d_S_sorted_perm[1]]
            for v in neighbors(g, d_S_sorted_perm[1])
                d_S[v] += 1
            end
        end
    end

    #approx_density = 0.999 * density(induced_subgraph(g, S′)[1])

    return length(S′)# + approx_density
end

struct SumOfNeighborsHeuristic <: GuidanceFunction 
    d::Int
    weight_num_edges::Real
end

function (heu::SumOfNeighborsHeuristic)(g::SimpleGraph, S::Vector{Int}, γ::Real)::Real
    V_S = setdiff(vertices(g), S)
    d_S = calculate_d_S(g, S)
    d_S_filtered = [d_S[i] for i in V_S]

    num_edges = calculate_num_edges(g, S)
    sum_of_neighbors = sum(partialsort(d_S_filtered, 1:min(heu.d, length(d_S_filtered)); rev=true))

    return round(Int, heu.weight_num_edges * num_edges + (1 - heu.weight_num_edges) * sum_of_neighbors)
end

mutable struct Node
    num_edges::Int # number of edges in subgraph induced by S 
    g_val::Real # heuristic value determined by guidance function
    S::Set{Int}
end

function beam_search_construction(g::SimpleGraph, γ::Real, guidance_function::GuidanceFunction; β=100, expansion_limit=Inf)
    root = Node(0, 0, Set())
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

        filter!(node -> !is_terminal(g, γ, collect(node.S)), children) # might not be necessary

        for node in children
            node.g_val = guidance_function(g, collect(node.S), γ)
        end

        beam = partialsort(children, 1:min(β, length(children)); by=(node -> node.g_val), rev=true)
    end

    return collect(max_node.S)
end

# expand node into feasible successors
function expand(g::SimpleGraph, node::Node, γ::Real, visited_solutions::Set{Set{Int}}; expansion_limit=Inf)::Vector{Node}
    d_S = calculate_d_S(g, collect(node.S))
    k = length(node.S)
    min_edges_needed = Int(ceil(γ * (k*(k+1)/2))) - node.num_edges
    d_S_sorted_perm = filter(v -> v ∉ node.S, sortperm(d_S; rev=true))

    children = Vector{Node}()
    
    for v in d_S_sorted_perm
        if d_S[v] >= min_edges_needed
            S′ = Set([node.S..., v])
            if S′ ∉ visited_solutions
                push!(visited_solutions, S′)
                push!(children, Node(node.num_edges + d_S[v], 0, S′))
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


"""
    is_terminal(g, γ, S)

Returns true if candidate solution `S` can be extended to a γ-quasi clique of size |`S`|+1, and 
false otherwise.

- `g`: Input Graph
- `γ`: Target density that defines feasibility of a quasi-clique
- `S`: List of distinct nodes in `g`, candidate solution

"""
function is_terminal(g::SimpleGraph, γ::Real, S::Vector{Int})
    V_S = setdiff(vertices(g), S)
    d_S = calculate_d_S(g, S)
    num_edges = calculate_num_edges(g, S)
    d_S_filtered = [d_S[i] for i in V_S]
    k = length(S)
    edges_needed = γ*(k*(k+1)/2) # minimum number of edges needed in solution of size |S|+1 

    # if clique cannot be extended by a single node, it is terminal -> return true
    if maximum(d_S_filtered) + num_edges < edges_needed 
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

    while length(S) < k
        if rand() < p
            N_G_S = open_set_neighborhood(g, S)
            if !isempty(open_set_neighborhood)
                u = filter(v -> v ∈ N_G_S, freq_sorted)[1]
            else
                V_S = setdiff(vertices(g), S)
                u = filter(v -> v ∈ V_S, freq_sorted)[1]
            end
        else
            d_S = calculate_d_S(g, S)
            V_S = setdiff(vertices(g), S) # V ∖ candidate_solution
            d_S_V_S = [d_S[i] for i in V_S] # only d_S values for V_S
            d_max = maximum(d_S_V_S)
            d_min = minimum(d_S_V_S)
            min_val = d_max - α*(d_max - d_min)
            restricted_candidate_list = filter(v -> d_S[v] >= min_val, V_S)
            u = sample(restricted_candidate_list)
        end
        push!(S, u)
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
