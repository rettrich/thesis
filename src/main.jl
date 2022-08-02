using Graphs
using StatsBase
using thesis
using thesis.Instances
using thesis.MPModels
using thesis.LookaheadSearch
using thesis.LocalSearch

ENV["JULIA_DEBUG"] = "thesis"



#model = get_MQCP_model(g, γ; opt="CPLEX", verbosity=0)
# @time solution = solve_model!(model, g)
# println("solution is valid? $(check_MQCP_solution(g, γ, solution)), size: $(length(solution))")

function test_MPModels(γ, k, d)
    g = generate_instance(40, 0.75)
    γ = 0.9
    println("Start search with size $k")
    count = 0
    candidate_solution = sample(collect(vertices(g)), k; replace=false)
    println(candidate_solution)
    initial_density = ne(induced_subgraph(g, candidate_solution)[1]) / (k*(k-1)/2)
    improvement = true
    while improvement
        current = ne(induced_subgraph(g, candidate_solution)[1])
        if current / (k*(k+1)/2) > γ
            break
        end
        neighborhood_model = get_MQCP_neighborhood_model(g, γ, candidate_solution, d)
        @time candidate_solution = solve_model!(neighborhood_model, g)
        num_edges = ne(induced_subgraph(g, candidate_solution)[1])
        if num_edges ≤ current
            improvement = false
        end
        
        count += 1
    end
    println("solution from search is valid? $(check_MQCP_solution(g, γ, candidate_solution))")
    println("$count iterations, density: $(ne(induced_subgraph(g, candidate_solution)[1]) / (k*(k-1)/2))")
    println("initial density: $initial_density")
    println(candidate_solution)

end

function test_lookahead_search(γ, d)
    g = generate_instance(150, 0.75)

    model = get_MQCP_model(g, γ; verbosity=1, timelimit=120)
    @time solution = solve_model!(model, g)
    k = length(solution)
    println("k: $k")
    # k = Int(round(0.35*nv(g)))

    candidate_solution = sample(collect(vertices(g)), k; replace=false)
    println("init :, obj: $(ne(induced_subgraph(g, candidate_solution)[1])), γ: $(ne(induced_subgraph(g, candidate_solution)[1]) / (k * (k-1) / 2) )")
    
    # println("\nrecursive search")
    # @time swap_history, rec_obj = recursive_search(g, candidate_solution, d)
    # rec_sol = get_solution(candidate_solution, swap_history)

    improvement = true
    current = ne(induced_subgraph(g, candidate_solution)[1])
    candidate_solution = Set(candidate_solution)

    while improvement
        println("\n beam search")
        @time bs_obj, candidate_solution = beam_search(g, candidate_solution, d; α=30, β=100)
        # @time bs_obj_2, bs_sol_2 = beam_search(g, Set(candidate_solution), d; α=30, β=50)
        
        if bs_obj > current
            current = bs_obj
        else
            improvement = false
        end
    end

    
    
    # println("\nMILP search")
    # model = get_MQCP_neighborhood_model(g, candidate_solution, d; verbosity=1)
    # @time milp_sol = solve_model!(model, g)
    # milp_obj = ne(induced_subgraph(g, milp_sol)[1])

    # println("\nrec  : $(rec_sol), obj: $rec_obj")
    println("\nbs   : obj: $current, γ: $(ne(induced_subgraph(g, collect(candidate_solution))[1]) / (k * (k-1) / 2))")
    # println("\nbs 2 : obj: $bs_obj_2, γ: $(ne(induced_subgraph(g, collect(bs_sol_2))[1]) / (k * (k-1) / 2))")
    # println("milp : $(milp_sol), obj: $milp_obj")


end

function test_local_search()
    g = load_instance("inst/DIMACS/brock400_2.clq")
    sol, obj = local_search_with_EX(g, 0.999; α=10, β=500, d=10)
    println("solution size: $(length(sol))")
end

# test_local_search()

