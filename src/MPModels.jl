module MPModels

using Graphs
using JuMP

"""
    get_MQCP_model(g, γ)
Create a JuMP MILP model for the Maximum γ-Quasi Clique Problem for given graph `g` and `γ` \in (0,1]

-`g`: Input graph
-`γ`: Problem parameter for the Maximum γ-Quasi Clique Problem
"""
function get_MQCP_model(g::SimpleGraph)
    error("not implemented")
end

"""
    get_MDCP_model(g, k)
Create a JuMP MILP model for the Maximum k-defective Clique Problem for given graph `g` and integer `k`

-`g`: Input graph
-`k`: Problem parameter for the Maximum k-defective Clique Problem
"""
function get_MDCP_model(g::SimpleGraph, k::Int)
    error("not implemented")
end

"""
    get_MPP_model(g, k)
Create a JuMP MILP model for the Maximum k-plex Problem for given graph `g` and integer `k`

-`g`: Input graph
-`k`: Problem parameter for the Maximum k-plex Clique Problem
"""
function get_MPP_model(g::SimpleGraph, k::Int)
    error("not implemented")
end

"""
    solve!(model)
Solve JuMP MILP model and return the set of nodes in solution

-`model`: JuMP MILP model
"""
function solve!(model, graph)
    error("not implemented")
end

end