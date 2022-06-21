using Graphs
using StatsBase
using thesis
using thesis.Instances
using thesis.MPModels

g = generate_instance(45, 0.75)
γ = 0.9

model = get_MQCP_model(g, γ; opt="Gurobi", verbosity=0)
@time solution = solve_model!(model, g)
println("solution_gurobi is valid? $(check_MQCP_solution(g, γ, solution)), size: $(length(solution))")

function test(k, d)
    count = 0
    candidate_solution = sample(collect(vertices(g)), k; replace=false)
    initial_density = ne(induced_subgraph(g, candidate_solution)[1]) / (k*(k-1)/2)
    improvement = true
    while improvement
        current = ne(induced_subgraph(g, candidate_solution)[1])
        neighborhood_model = get_MQCP_neighborhood_model(g, γ, candidate_solution, d)
        @time candidate_solution = solve_model!(neighborhood_model, g)
        if ne(induced_subgraph(g, candidate_solution)[1]) ≤ current
            improvement = false
        end
        count += 1
    end
    println("solution from search is valid? $(check_MQCP_solution(g, γ, candidate_solution))")
    println("$count iterations, density: $(ne(induced_subgraph(g, candidate_solution)[1]) / (k*(k-1)/2))")
    println("initial density: $initial_density")

end

test(length(solution), 5)



