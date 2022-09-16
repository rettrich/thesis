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
    # lower_bound_heuristic = BeamSearch_LowerBoundHeuristic(guidance_func; β=1, γ, expansion_limit=10)
    lower_bound_heuristic = SingleVertex_LowerBoundHeuristic()

    construction_heuristic = Freq_GRASP_ConstructionHeuristic(0.2, 0.3)

    short_term_memory = TabuList()

    feasibility_checker = MQCP_FeasibilityChecker(γ)
    
    solution_extender = MQCP_GreedySolutionExtender(γ)

    gnn = ResGatedGraphConvGNN(2, [64, 64, 64]; node_features=[DegreeNodeFeature(), d_S_NodeFeature()])
    scoring_function = GNN_ScoringFunction(gnn, 20)

    # compare with baseline
    baseline_scoring_function = d_S_ScoringFunction()

    # local search procedure uses short term memory and scoring function
    local_search_procedure = MQCP_LocalSearchProcedure(γ, short_term_memory, scoring_function)
    baseline_local_search_procedure = MQCP_LocalSearchProcedure(γ, short_term_memory, baseline_scoring_function)

    timelimit = 100.0
    max_iter = 4000
    next_improvement = false
    record_swap_history = true
    max_restarts = 3 

    local_search = LocalSearchBasedMH(
            lower_bound_heuristic, construction_heuristic, local_search_procedure, feasibility_checker, solution_extender;
            timelimit, max_iter, next_improvement, record_swap_history, max_restarts)

    baseline_local_search = LocalSearchBasedMH(
        lower_bound_heuristic, construction_heuristic, baseline_local_search_procedure, feasibility_checker, solution_extender;
        timelimit, max_iter, next_improvement, record_swap_history=false, max_restarts=100)

    instance_generator = Training.InstanceGenerator(Normal(200, 15), Uniform(0.4, 0.6))

    Training.train!(local_search, instance_generator, gnn; epochs=350, baseline=baseline_local_search)

    BSON.@save "gnn-medium-density-2-3x64.bson" gnn

    nothing
end

train_MQCP()
