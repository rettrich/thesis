"""
    FeasibilityChecker

Checks for a specific problem whether or not the given candidate solution is feasible or not

"""
abstract type FeasibilityChecker end

(::FeasibilityChecker)(::SimpleGraph, S::Vector{Int}) = 
    error("FeasibilityChecker: Abstract Functor called")


struct MQCP_FeasibilityChecker <: FeasibilityChecker
    γ::Real
end

(feasibility_checker::MQCP_FeasibilityChecker)(graph::SimpleGraph, S::Vector{Int}) = (density_of_subgraph(graph, S) >= feasibility_checker.γ)

function is_feasible_MQC(g, S, γ)
    return density_of_subgraph(g, S) >= γ 
end

function density_of_subgraph(g, S)
    density(induced_subgraph(g, S)[1])
end