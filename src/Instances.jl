module Instances

export load_instance, load_instances, generate_instance, generate_connected_instance, 
       MQCPInstance, graph_to_file 
using Graphs
using StatsBase

"""
    MQCPInstance

- `graph`: Input graph
- `graph_id`: Name of input graph
- `target_γ`: Target density
- `best_known`: Size of best known solution

"""
struct MQCPInstance
    graph::SimpleGraph
    graph_id::String
    target_γ::Real
    best_known::Int

    function MQCPInstance(graph_id::String, target_γ::Real, best_known::Int; benchmark_id::String="DIMACS")
        graph = load_instance("../inst/$benchmark_id/$graph_id.clq")
        new(graph, graph_id, target_γ, best_known)
    end
end

"""
    load_instance(file)
Load graph instance from file as `SimpleGraph`

- `file`: Path to the instance file
"""
function load_instance(file)
    g = SimpleGraph()
    open(file) do f
        for line in eachline(f)
            startswith(line, "c") && continue
            split_line = split(line, r"\s+")
            startswith(line, "p") && add_vertices!(g, parse(Int, split_line[3]))
            startswith(line, "e") && add_edge!(g, parse(Int, split_line[2]), parse(Int, split_line[3]))
        end
    end
    
    return g
end

"""
    load_instances(dir)
Load all graph instances in directory `dir` as `Dict` of `SimpleGraph`s

- `dir`: Directory to instance files
"""
function load_instances(dir)
    graphs = Dict()
    for file in readdir(dir)
        graph_key = split(file, ".")[1]
        g = load_instance(joinpath(dir, file))
        graphs[graph_key] = g
    end
    return graphs
end

"""
    generate_instance(n, γ)
Generate random simple graph with `n` vertices and density γ with uniformly sampled edges.

- `n`: Number of nodes in graph
- `dens`: Density of graph, `γ`∈(0,1]

"""
function generate_instance(n::Int, dens::Real)
    g = SimpleGraph(n)
    m = Int(n * (n - 1) / 2) # number of edges in complete graph
    edge_list = sample(1:m, round(Int, dens*m); replace=false)
    for num in edge_list
        i,j = _num_to_edge(num, n)
        add_edge!(g, i, j)
    end
    return g
end

function graph_to_file(graph::SimpleGraph, filename::String)
    open("$filename", "w") do f
        write(f, "c random graph: density $(density(graph))\n")
        write(f, "p edge $(nv(graph)) $(ne(graph))\n")
        for e in edges(graph)
            write(f, "e $(src(e)) $(dst(e))\n")
        end
    end
end

function generate_connected_instance(n::Int, dens::Real)
    g, edge_list = generate_random_tree(n)
    m = Int(n * (n - 1) / 2)
    num_edges = round(Int, dens*m) - ne(g)
    add_edges = sample(setdiff(collect(1:m), [_edge_to_num(u, v, n) for (u, v) in edge_list]), num_edges; replace=false)
    for num in add_edges
        u, v = _num_to_edge(num, n)
        add_edge!(g, u, v)
    end
    @assert is_connected(g)
    return g
end

function generate_random_tree(n::Int)
    g = SimpleGraph(n)

    not_connected = Set(collect(vertices(g)))
    connected = Set()
    
    # initial vertex
    v_start = sample(collect(not_connected))
    delete!(not_connected, v_start)
    push!(connected, v_start)
    edge_list = []
    
    while !isempty(not_connected)
        u = sample(collect(not_connected))
        delete!(not_connected, u)
        v = sample(collect(connected))
        push!(connected, u)
        add_edge!(g, u, v)
        push!(edge_list, (min(u,v), max(u,v)))
    end

    return g, edge_list
end

"""
    _num_to_edge(num, n)
Maps integers in the range 1:(n*(n-1)/2) to distinct edges in a simple graph with `n` vertices

- `num`: Edge number
- `n`: Number of vertices in the graph 
"""
function _num_to_edge(num::Int, n::Int)
    i = 1
    j = 1
    k = n-1
    while num > k
        num -= k
        k -= 1
        i += 1
    end
    j = i + num
    return i, j
end

# reverse function of num_to_edge
function _edge_to_num(u::Int, v::Int, n::Int)
    if u == 1 return v-u end
    return n*(u-1) - Int((u-1)*u/2) + v-u
end

end # module