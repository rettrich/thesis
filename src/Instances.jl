module Instances

export load_instance, load_instances, generate_instance, 
       MQCPInstance 
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
            startswith(line, "p") && add_vertices!(g, parse(Int, split(line, " ")[3]))
            startswith(line, "e") && add_edge!(g, parse(Int, split(line, " ")[2]), parse(Int, split(line, " ")[3]))
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
- `γ`: Density of graph, `γ`∈(0,1]

"""
function generate_instance(n::Int, γ::Real)
    g = SimpleGraph(n)
    m = Int(n * (n - 1) / 2) # number of edges in complete graph
    edge_list = sample(1:m, round(Int, γ*m); replace=false)
    for num in edge_list
        i,j = _num_to_edge(num, n)
        add_edge!(g, i, j)
    end
    return g
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

end # module