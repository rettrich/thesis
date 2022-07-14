"""
    ShortTermMemory

A short term memory mechanism (e.g. tabu list) to prevent cycling through the 
same solutions during a local search procedure. 
Each subtype has to implement the methods reset!, get_blocked, move!

"""
abstract type ShortTermMemory end

"""
    reset!(short_term_memory)

Resets the short term memory. Memory about blocked vertices is reset

- `short_term_memory`: Short term memory to be reset

"""
reset!(stm::ShortTermMemory) = 
    error("ShortTermMemory: Abstract reset! called")

"""
    get_blocked!(short_term_memory, vertex_list)

Returns all currently blocked vertices as a vector.

- `short_term_memory`: The short term memory data structure from which information 
    about blocked vertices is retrieved

"""
get_blocked(stm::ShortTermMemory)::Vector{Int} = 
    error("ShortTermMemory: Abstract remove_blocked! called")

"""
    move!(short_term_memory, u, v)

This method should be called when a vertex `u` ∈ S is swapped with a vertex `v` ∈ V ∖ S. 
The `short_term_memory` then stores information about this swap and blocks 
vertices `u` or `v` according to the concrete implementation. 
"""
move!(stm::ShortTermMemory, u::Int, v::Int) = 
    error("ShortTermMemory: Abstract move! called")

"""
    TabuList

Keeps a tabu list of vertices that are blocked from being swapped. 

- `g`: Input graph
- `tabu_tenure`: Each time a vertex is added to the `tabu_list`, it is blocked 
    for `tabu_tenure` iterations
- `tabu_list`: `tabu_list[i]` corresponds to vertex `i`. Vertex `i` is blocked until iteration 
    `tabu_list[i]`
. `iteration`: Iteration counter, is increased each time `move!` is called on a `TabuList` instance
"""
mutable struct TabuList <: ShortTermMemory 
    g::SimpleGraph
    tabu_tenure::Int
    tabu_list::Vector{Int}
    iteration::Int

    function TabuList(g::SimpleGraph; tabu_tenure::Int = 10)
        new(g, tabu_tenure, fill(0, nv(g)), 0)
    end
end

function reset!(stm::TabuList)
    stm.tabu_list = fill(0, nv(stm.g))
    stm.iteration = 0
end

function get_blocked(stm::TabuList)::Vector{Int}
    filter(v -> stm.tabu_list[v] > stm.iteration , vertices(stm.g))
end

function move!(stm::TabuList, u::Int, v::Int)
    stm.tabu_list[u] = stm.iteration + stm.tabu_tenure
    stm.tabu_list[v] = stm.iteration + stm.tabu_tenure
    stm.iteration += 1
end

"""
    ConfigurationChecking

ConfigurationChecking according to BoundedCC as described in Chen et al. 2021, 
NuQClq: An Effective Local Search Algorithm for Maximum Quasi-Clique Problem. 
Initially, each for each vertex v set `conf_change[v]` and `threshold[v]` to 1. 
Each time a vertex v is added to the solution, increase `conf_change[u]` by 1 for all neighbors of v, 
and increase `threshold[v]` by 1. If `threshold[v]` > `ub_threshold`, then `threshold[v]` is reset to 1.
Each time a vertex v is moved outside of the solution, `conf_change[v]` is reset to 0. 
A vertex v is blocked if `conf_change[v]` < `threshold[v]`. 

- `g`: Input graph
- `ub_threshold`: Upper bound for threshold. When `threshold[v]` > `ub_threshold`, set `threshold[1]` to 1. 
- `threshold`: Contains threshold values for all vertices in `g`. 
- `conf_change`: Contains current configuration for all vertices in `g`.

"""
mutable struct ConfigurationChecking <: ShortTermMemory 
    g::SimpleGraph
    ub_threshold::Int
    threshold::Vector{Int}
    conf_change::Vector{Int}

    function ConfigurationChecking(g::SimpleGraph; ub_threshold::Int = 7)
        new(g, ub_threshold, fill(1, nv(g)), fill(1, nv(g)))
    end
end

function reset!(stm::ConfigurationChecking)
    stm.threshold = fill(1, nv(stm.g))
    stm.conf_change = fill(1, nv(stm.g))
    nothing
end

function get_blocked(stm::ConfigurationChecking)::Vector{Int}
    filter(v -> stm.conf_change[v] < stm.threshold[v], vertices(stm.g))
end

function move!(stm::ConfigurationChecking, u::Int, v::Int)
    # u is moved from S outside of solution
    stm.conf_change[u] = 0

    # v is moved from V ∖ S into S
    for w in neighbors(stm.g, v)
        stm.conf_change[w] += 1
    end
    stm.threshold[v] += 1
    stm.threshold[v] > stm.ub_threshold && (stm.threshold[v] = 1)
    nothing
end