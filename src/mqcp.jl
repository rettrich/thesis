using ArgParse
using Graphs
using Revise
using thesis
using thesis.LocalSearch
using thesis.Instances
using thesis.GNNs
using CSV
using CUDA
using DataFrames
using Flux
using BSON
using GraphNeuralNetworks
using MHLib


# To run, provide a graph instance and target density γ, e.g.:
# julia --project=. .\src\mqcp.jl --graph="inst/DIMACS/brock400_1.clq" --gamma=0.999 --timelimit=60

include("mqcp_args.jl")

parse_settings!([settings_cfg, thesis.NodeRepresentationLearning.settings_cfg],
                vcat(ARGS,
                [
                    #"--debug=true"
                ]))

if settings[:debug]
    ENV["JULIA_DEBUG"] = "thesis"
end

function run_mqcp(scoring_function=nothing)
    γ = settings[:gamma]

    # initialize components of local search based metaheuristic
    
    # lower bound heuristic: beam search with GreedyCompletionHeuristic as guidance function
    guidance_func = GreedyCompletionHeuristic()
    # lower_bound_heuristic = BeamSearch_LowerBoundHeuristic(guidance_func; β=parsed_args["beta"], γ, expansion_limit=parsed_args["expansion_limit"])
    lower_bound_heuristic = SingleVertex_LowerBoundHeuristic()

    construction_heuristic = Freq_GRASP_ConstructionHeuristic(settings[:alpha], settings[:p])

    Stm_Type = get_stm(settings[:stm])
    short_term_memory = Stm_Type()

    feasibility_checker = MQCP_FeasibilityChecker(γ)
    
    solution_extender = MQCP_GreedySolutionExtender(γ)

    neighborhood_search = VariableNeighborhoodDescent(settings[:vnd_depth])

    if isnothing(scoring_function)
        scoring_function = d_S_ScoringFunction()
    end

    # local search procedure uses short term memory and scoring function
    local_search_procedure = MQCP_LocalSearchProcedure(γ, short_term_memory, scoring_function)
    println("ScoringFunction: $(typeof(scoring_function))")


    local_search = LocalSearchBasedMH(
            lower_bound_heuristic, construction_heuristic, local_search_procedure, feasibility_checker, solution_extender;
            neighborhood_search, timelimit=settings[:timelimit], max_iter=settings[:max_iter], 
            next_improvement=settings[:next_improvement], record_swap_history=false, max_restarts=settings[:max_restarts],
            sparse_evaluation=settings[:sparse_evaluation],
            )
    
    results = Dict()

    for inst in settings[:graph]
        results[inst.name] = Int[]
        for i = 1:settings[:runs_per_instance]
            println("Start run $i/$settings[:runs_per_instance] for $(inst.name)")
            t = @elapsed solution, _ = run_lsbmh(local_search, inst.graph)
            push!(results[inst.name], length(solution))
            println("Found solution: $(sort(solution)), runtime $t")
        end
        println("Best: $(maximum(results[inst.name])), Avg: $(sum(results[inst.name]) / length(results[inst.name]))")
        println()
    end

    if settings[:write_result] != "-"
        df = DataFrame(GraphID=String[], V=Int[], E=Int[], Dens=Real[], γ=Real[], Avg=Real[], Best=Int[])
        for (graph_id, sols) in results
            push!(df, (
                graph_id,
                nv(graph),
                ne(graph),
                density(graph),
                γ,
                length(sum(sols) / length(sols))),
                maximum(sols)
            )
        end
        
        CSV.write("$(settings[:write_result])/$(settings[:scoring_function]).csv", df)
    end

end

run_mqcp(scoring_function)