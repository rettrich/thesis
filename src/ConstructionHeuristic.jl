
"""
    ConstructionHeuristic

Base type for all Construction heuristics used in the `LocalSearchBasedMH`.
Returns an initial candidate solution of given size when applied to a graph and 
a frequency list, which may in general not be feasible. This candidate solution 
is then used as a starting point for the local search procedure. 

"""
abstract type ConstructionHeuristic end

"""
    (::ConstructionHeuristic)(graph::SimpleGraph, k::Int, freq::Vector{Int})

Returns an initial candidate solution of given size when applied to a graph and 
a frequency list, which may in general not be feasible. This candidate solution 
is then used as a starting point for the local search procedure. 

- `graph`: Input graph
- `k`: Size of the constructed candidate solution
- `freq`: Frequency list, records how many times each vertex was swapped during local search. 

"""
(::ConstructionHeuristic)(graph::SimpleGraph, k::Int, freq::Vector{Int}) = 
    error("Abstract (::ConstructionHeuristic)(graph, k, freq) called")

"""
    Freq_GRASP_ConstructionHeuristic

Type for construction heuristic that corresponds to Algorithm 5.4 in thesis. 
In each iteration, with probability `p` a vertex with low frequency value is added, 
and with probability 1-p a vertex is added in a GRASP-like manner.

"""
struct Freq_GRASP_ConstructionHeuristic <: ConstructionHeuristic
    α::Real
    p::Real

    function Freq_GRASP_ConstructionHeuristic(α::Real=0.2, p::Real=0.2)
        new(α, p)
    end
end

"""
    construction_heuristic(g, k, freq; α, p)

Corresponds to Algorithm 5.4 in thesis. Returns a vector of distinct vertices in `g` of size `k` that is build 
incrementally. In each iteration, with probability `p` a vertex with low frequency value is added, 
and with probability 1-p a vertex is added in a GRASP-like manner.

- `graph`: Input Graph
- `k`: Target size of the returned vector of vertices
- `freq`: Frequency list with length |V|. Vertices with low frequency are preferred during construction.

"""
function (construction_heuristic::Freq_GRASP_ConstructionHeuristic)(graph::SimpleGraph, k::Int, freq::Vector{Int})
    α = construction_heuristic.α
    p = construction_heuristic.p

    freq_sorted = sortperm(freq)
    init_vertex = freq_sorted[1]
    S = [init_vertex]
    d_S = calculate_d_S(graph, S)

    while length(S) < k
        if rand() < p
            N_G_S = open_set_neighborhood(graph, S)
            if !isempty(N_G_S)
                u = filter(v -> v ∈ N_G_S, freq_sorted)[1]
            else
                V_S = setdiff(vertices(graph), S)
                u = filter(v -> v ∈ V_S, freq_sorted)[1]
            end
        else
            V_S = setdiff(vertices(graph), S) # V ∖ candidate_solution
            d_S_V_S = [d_S[i] for i in V_S] # only d_S values for V_S
            d_max = maximum(d_S_V_S)
            d_min = minimum(d_S_V_S)
            min_val = d_max - α*(d_max - d_min)
            restricted_candidate_list = filter(v -> d_S[v] >= min_val, V_S)
            u = sample(restricted_candidate_list)
        end
        push!(S, u)
        for v in neighbors(graph, u)
            d_S[v] += 1
        end
    end

    return S
end

# """
#     construction_heuristic(g, k, freq; α, p)

# Corresponds to Algorithm 5.4 in thesis. Returns a vector of distinct vertices in `g` of size `k` that is build 
# incrementally. In each iteration, with probability `p` a vertex with low frequency value is added, 
# and with probability 1-p a vertex is added in a GRASP-like manner.

# - `g`: Input Graph
# - `k`: Target size of the returned vector of vertices
# - `freq`: Frequency list with length |V|. Vertices with low frequency are preferred during construction.
# - `α`: GRASP parameter; α=0 performs a greedy construction, α=1 performs a randomized construction
# - `p`: Controls the balance between GRASP construction and preferring vertices with low frequency values. 
#     `p`=0 ignores frequency values, while `p`=1 only uses frequency values.

# """
# function construction_heuristic(graph::SimpleGraph, k::Int, freq=[0 for i=1:nv(graph)]::Vector{Int}; α=0.2::Real, p=0.2::Real)
#     freq_sorted = sortperm(freq)
#     init_vertex = freq_sorted[1]
#     S = [init_vertex]
#     d_S = calculate_d_S(graph, S)

#     while length(S) < k
#         if rand() < p
#             N_G_S = open_set_neighborhood(graph, S)
#             if !isempty(N_G_S)
#                 u = filter(v -> v ∈ N_G_S, freq_sorted)[1]
#             else
#                 V_S = setdiff(vertices(graph), S)
#                 u = filter(v -> v ∈ V_S, freq_sorted)[1]
#             end
#         else
#             V_S = setdiff(vertices(graph), S) # V ∖ candidate_solution
#             d_S_V_S = [d_S[i] for i in V_S] # only d_S values for V_S
#             d_max = maximum(d_S_V_S)
#             d_min = minimum(d_S_V_S)
#             min_val = d_max - α*(d_max - d_min)
#             restricted_candidate_list = filter(v -> d_S[v] >= min_val, V_S)
#             u = sample(restricted_candidate_list)
#         end
#         push!(S, u)
#         for v in neighbors(graph, u)
#             d_S[v] += 1
#         end
#     end

#     return S
# end

"""
    open_set_neighborhood(graph, vertex_list)

Return the union of neighborhoods of vertices in `vertex_list` excluding vertices in `vertex_list`

- `graph`: Input Graph
- `vertex_list`: Vector of vertices from `g`

"""
open_set_neighborhood(graph::SimpleGraph, vertex_list::Vector{Int})::Vector{Int} = 
    setdiff(reduce(vcat, neighbors(graph, v) for v in vertex_list), vertex_list)


open_set_neighborhood(graph::SimpleGraph, vertex_set::Set{Int})::Vector{Int} = open_set_neighborhood(graph, collect(vertex_set))

"""
    calculate_num_edges(graph, candidate_solution)

Returns the number of edges in the subgraph induced by candidate solution `S` in `g`

- `graph`: Input Graph
- `S`: List of nodes representing the candidate solution

"""
calculate_num_edges(graph::SimpleGraph, S::Vector{Int})::Int = ne(induced_subgraph(graph, S)[1])
