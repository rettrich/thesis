module Instances

export load_instance, load_instances, generate_instance 
using Graphs
using StatsBase

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
- `γ`: Density of graph

"""
function generate_instance(n, γ)
    g = SimpleGraph(n)
    m = Int(n * (n - 1) / 2) # number of edges
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