using thesis.LocalSearch
using thesis
using thesis.GNNs
using Distributions
using Dates
using BSON

# ENV["JULIA_DEBUG"] = "thesis"

function train_MQCP()
    γ = 0.999

    # initialize components of local search based metaheuristic
    
    # lower bound heuristic: beam search with GreedyCompletionHeuristic as guidance function
    guidance_func = GreedyCompletionHeuristic()
    bs_heuristic = BeamSearch_LowerBoundHeuristic(guidance_func; β=5, γ, expansion_limit=50)

    construction_heuristic = Freq_GRASP_ConstructionHeuristic(0.2, 0.3)

    short_term_memory = TabuList()

    feasibility_checker = MQCP_FeasibilityChecker(γ)
    
    solution_extender = MQCP_GreedySolutionExtender(γ)

    gnn = ResGatedGraphConvGNN(2, [20, 20])
    scoring_function = GNN_ScoringFunction(gnn, 20)

    # local search procedure uses short term memory and scoring function
    local_search_procedure = MQCP_LocalSearchProcedure(γ, short_term_memory, scoring_function)

    timelimit = 60.0
    max_iter = 4000
    next_improvement = false
    record_swap_history = true

    local_search = LocalSearchBasedMH(
            bs_heuristic, construction_heuristic, local_search_procedure, feasibility_checker, solution_extender;
            timelimit, max_iter, next_improvement, record_swap_history)

    instance_generator = Training.InstanceGenerator(Normal(200, 15), Uniform(0.4, 0.6))

    gnn = Training.train!(local_search, instance_generator, gnn)

    BSON.@save "gnn-$(DateTime(Dates.now())).bson" gnn

    nothing
end

train_MQCP()
