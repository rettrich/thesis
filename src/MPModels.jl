module MPModels

using Graphs
using JuMP
using CPLEX
using Gurobi

export get_MDCP_model, 
       get_MQCP_model, get_MQCP_neighborhood_model, check_MQCP_solution, 
       get_MPP_model, 
       solve_model!

#################################### Private Methods ################################################

function cplex_model(; verbosity::Int=0, eps_int::Real=1e-6, timelimit::Real=Inf)
    model = Model(CPLEX.Optimizer)
    set_optimizer_attribute(model, "CPX_PARAM_EPINT", eps_int)
    set_optimizer_attribute(model, "CPX_PARAM_SCRIND", verbosity)
    if timelimit < Inf
        set_time_limit_sec(model, timelimit)
    end
    return model
end

const GRB_ENV = Ref{Gurobi.Env}()
function __init__()
    GRB_ENV[] = Gurobi.Env()
    return
end

function gurobi_model(; verbosity::Int=0, eps_int::Real=1e-6, timelimit::Real=Inf)
    model = Model(() -> Gurobi.Optimizer(GRB_ENV[]))
    set_optimizer_attribute(model, "OutputFlag", verbosity)
    set_optimizer_attribute(model, "LogToConsole", verbosity)
    set_optimizer_attribute(model, "IntFeasTol", eps_int)
    if timelimit < Inf
        set_time_limit_sec(model, timelimit)
    end
    return model
end

# adjacency matrix of graph
a(g,i,j)::Int = Int(has_edge(g,i,j))

# get integer epsilon from model
function model_epsilon(model)
    if solver_name(model) == "CPLEX"
        return get_optimizer_attribute(model, "CPX_PARAM_EPINT")
    elseif solver_name(model) == "Gurobi"
        return get_optimizer_attribute(model, "IntFeasTol")
    else
        error("solver not supported")
    end
end

# get empty model
function get_empty_model(opt::String; verbosity::Int=0, eps_int::Real=1e-6, timelimit::Real=Inf)
    if opt == "CPLEX"
        model = cplex_model(;verbosity, eps_int, timelimit)
    elseif opt == "Gurobi"
        model = gurobi_model(;verbosity, eps_int, timelimit)
    else
        error("Optimizer $(opt) not supported")
    end
    return model
end

#####################################################################################################
########################### Maximum Quasi-Clique Problem ############################################ 
#####################################################################################################

"""

    get_MQCP_model(g, γ; opt, verbosity, eps_int, timelimit)

Create a JuMP MILP model for the Maximum γ-Quasi Clique Problem for given graph `g` and `γ` ∈ (0,1].
Model from 'On the Maximum Quasi-Clique Problem', Pattillo et al. 2013, variables and constraints quadratic in 
size of vertices in graph `g`. 

-`g`: Input graph
-`γ`: Problem parameter for the Maximum γ-Quasi Clique Problem
-`verbosity`: Verbosity of solver output
-`opt`: Set optimizer ∈ ["CPLEX", "Gurobi"]
-`eps_int`: Machine epsilon for integer accuracy
-`timelimit`: Timelimit for solver

"""
function get_MQCP_model(g::SimpleGraph, γ::Real;
                        opt::String="Gurobi", 
                        verbosity::Int=0, eps_int::Real=1e-6, timelimit::Real=Inf)
    n = nv(g)

    model = get_empty_model(opt; verbosity, eps_int, timelimit)

    # variables x[i] indicate whether node i is in solution
    @variable(model, x[i=1:n], Bin)
    # variables w[i,j] = x[i]*x[j] indicate edges in solution
    @variable(model, w[i=1:n, j=1:n; i < j] ≥ 0)

    @objective(model, Max, sum(x[i] for i in 1:n))

    for i=1:n, j=(i+1):n
        @constraint(model, w[i,j] ≤ x[i])
        @constraint(model, w[i,j] ≤ x[j])
        @constraint(model, w[i,j] ≥ x[i] + x[j] - 1)
    end
    
    @constraint(model, sum(((γ - a(g,i,j))*w[i,j]) for i=1:n, j=(i+1):n) ≤ 0)

    return model
end

"""
    get_MQCP_neighborhood_model(g, γ, candidate_solution, d)

Create a JuMP MILP model for the Maximum γ-Quasi Clique Problem for given graph `g` and `γ` ∈ (0,1] and 
a candidate solution, where the solution can only change up to `d` nodes from the candidate solution. 

-`g`: Input graph
-`γ`: Problem parameter for the Maximum γ-Quasi Clique Problem
-`candidate_solution`: A vector of node indices of the current candidate solution
-`d`: The depth of the look-ahead search. The model defines the best solution that can be found 
    by swapping at most `d` nodes from the `candidate_solution`.
-`verbosity`: Verbosity of solver output
-`eps_int`: Machine epsilon for integer accuracy
-`timelimit`: Timelimit for solver

"""
function get_MQCP_neighborhood_model(g::SimpleGraph, candidate_solution::Vector{Int}, d::Int; 
                                     opt::String="Gurobi", verbosity::Int=0, eps_int::Real=1e-6, timelimit::Real=Inf)
    
    n = nv(g)
    k = length(candidate_solution)
    model = get_empty_model(opt; verbosity, eps_int, timelimit)

    # variables x[i] indicate whether node i is in solution
    @variable(model, x[i=1:n], Bin)
    # variables w[i,j] = x[i]*x[j] indicate edges in solution
    @variable(model, w[i=1:n, j=1:n; i < j] ≥ 0)

    @objective(model, Max, sum(a(g,i,j)*w[i,j] for i=1:n, j=(i+1):n))

    @constraint(model, sum(x[i] for i=1:n) == k)

    @constraint(model, sum(x[i] for i in candidate_solution) ≥ k - d)

    for i=1:n, j=(i+1):n
        @constraint(model, w[i,j] ≤ x[i])
        @constraint(model, w[i,j] ≤ x[j])
        @constraint(model, w[i,j] ≥ x[i] + x[j] - 1)
    end

    return model
end

function check_MQCP_solution(g::SimpleGraph, γ::Real, solution::Vector{Int})
    sol_graph, vmap = induced_subgraph(g, solution)
    n_s = length(solution)
    m_s = ne(sol_graph)
    if m_s ≥ γ * (n_s * (n_s-1) / 2)
        return true
    else
        return false
    end
end

#####################################################################################################
########################### Maximum k-Defective Clique Problem ###################################### 
#####################################################################################################

"""
    get_MDCP_model(g, k)
Create a JuMP MILP model for the Maximum k-defective Clique Problem for given graph `g` and integer `k`

-`g`: Input graph
-`k`: Problem parameter for the Maximum k-defective Clique Problem
"""
function get_MDCP_model(g::SimpleGraph, k::Int)
    error("not implemented")
end

#####################################################################################################
################################## Maximum k-plex Problem ########################################### 
#####################################################################################################

"""
    get_MPP_model(g, k)
Create a JuMP MILP model for the Maximum k-plex Problem for given graph `g` and integer `k`

-`g`: Input graph
-`k`: Problem parameter for the Maximum k-plex Clique Problem
"""
function get_MPP_model(g::SimpleGraph, k::Int)
    error("not implemented")
end

#####################################################################################################

"""
    solve_model!(model, g)
Solve JuMP MILP model and return the set of nodes in solution

-`model`: JuMP MILP model
"""
function solve_model!(model::Model, g::SimpleGraph) :: Union{Nothing, Vector{Int}}
    optimize!(model)
    if primal_status(model) != JuMP.MathOptInterface.FEASIBLE_POINT
        @debug "No feasible solution found"
        return Nothing
    end
    @debug "Termination status: $(termination_status(model))"
    @debug "Primal status: $(primal_status(model))"

    x = value.(model[:x])

    n = nv(g)
    epsilon = model_epsilon(model)
    solution = Int[]

    for i in 1:n
        if x[i] ≥ 1 - epsilon
            push!(solution, i)
        end
    end
    return solution
end

end