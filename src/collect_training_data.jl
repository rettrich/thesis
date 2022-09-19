using ArgParse
using Graphs
using thesis.Instances
using thesis.LocalSearch
using thesis.Training
using CSV
using DataFrames
using Distributions
using BSON

# To run, provide a graph instance and target density γ, e.g.:
# julia --project=. .\src\mqcp.jl --graph="DIMACS/brock400_1.clq" --gamma=0.999 --timelimit=60

ENV["JULIA_DEBUG"] = "thesis"

function run_mqcp(i::Int)
    num_vertices = round(Int, rand(Normal(200, 15)))
    dens = rand(Uniform(0.05, 0.2))
    graph = generate_instance(num_vertices, dens)
    γ = 0.99

    construction_heuristic_settings = ConstructionHeuristicSettings(
                                    0.4, # parameter p of exploration construction
                                    0.25, # GRASP parameter α
                                    5,   # beamwidth β of initial construction
                                    50,  # expansion_limit of initial construction
                                    GreedyCompletionHeuristic() # guidance function of lower bound heuristic
                                    )
    short_term_memory = ConfigurationChecking(graph)
    settings = LocalSearchSettings(graph;
                                   construction_heuristic_settings,
                                   short_term_memory,
                                   timelimit=15.0,
                                   max_iter=5000,
                                   next_improvement=false,
                                   record_swap_history=true
                                   )

    solution, swap_history = run_MQCP(graph, γ; settings)

    println(solution)
    println("size of solution: $(length(solution))")
    println("encountered $(length(swap_history)) candidate solutions")
    data = sample_candidate_solutions(swap_history, 100)

    BSON.@save "training_data/low_density/graph_$(lpad(string(i), 3, '0'))_samples.bson" data

end

function generate_graphs(iter=10)
    
    ig = InstanceGenerator(Normal(200, 15), Uniform(0.75, 0.92))
    for i = 1:iter
        graph = sample_graph(ig)
        graph_to_file(graph, "rnd-medium-$(lpad(string(i), 3, "0")).clq")
    end
end

generate_graphs()
# for i = 1:100
#     run_mqcp(i)
# end