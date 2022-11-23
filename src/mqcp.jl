using ArgParse
using Graphs, Flux, GraphNeuralNetworks
using Revise
using thesis, thesis.LocalSearch, thesis.Instances, thesis.GNNs
using CSV, DataFrames
using BSON
using MHLib
using Statistics
# using SimpleWeightedGraphs, SparseArrays # needed to deserialize gnn with bson


# To run, provide a graph instance and target density γ, e.g.:
# julia --project=. .\src\mqcp.jl --graph="inst/DIMACS/brock400_1.clq" --gamma=0.999 --timelimit=60.0

include("mqcp_args.jl")

parse_settings!([settings_cfg, thesis.NodeRepresentationLearning.settings_cfg],
                vcat(ARGS,
                [
                    # "--debug=true",
                    # "--stm=cc",
                    "--timelimit=300.0",
                    # "--graph=inst/BHOSLIB/frb30-15-1.clq",
                    # "--neighborhood_size=30", 
                    "--gamma=0.95",
                    # "score_based_sampling"
                ]))

if settings[:debug]
    ENV["JULIA_DEBUG"] = "thesis"
end

scoring_function = nothing
if settings[:scoring_function] != "-"
    println("Load scoring function $(settings[:scoring_function])")
    BSON.@load settings[:scoring_function] gnn
    scoring_function = Encoder_Decoder_ScoringFunction(gnn, settings[:neighborhood_size])
end

function run_mqcp(scoring_function=nothing)
    γ = settings[:gamma]

    # initialize components of local search based metaheuristic
    
    # lower bound heuristic: beam search with GreedyCompletionHeuristic as guidance function
    guidance_func = GreedyCompletionHeuristic()
    lower_bound_heuristic = BeamSearch_LowerBoundHeuristic(guidance_func; β=settings[:beta], γ, expansion_limit=settings[:expansion_limit])
    # lower_bound_heuristic = SingleVertex_LowerBoundHeuristic()

    construction_heuristic = Freq_GRASP_ConstructionHeuristic(settings[:alpha], settings[:p])

    Stm_Type = get_stm(settings[:stm])
    short_term_memory = Stm_Type()

    feasibility_checker = MQCP_FeasibilityChecker(γ)
    
    solution_extender = MQCP_GreedySolutionExtender(γ)

    if settings[:vnd_depth] > 1
        neighborhood_search = VariableNeighborhoodDescent(settings[:vnd_depth])
    else
        neighborhood_search = Ω_1_NeighborhoodSearch()
    end

    if isnothing(scoring_function)
        scoring_function = d_S_ScoringFunction()
    end

    # local search procedure uses short term memory and scoring function
    local_search_procedure = MQCP_LocalSearchProcedure(γ, short_term_memory, scoring_function)
    println("ScoringFunction: $(typeof(scoring_function))")

    score_based_sampling = settings[:score_based_sampling] && (typeof(scoring_function) <: GNN_ScoringFunction)

    local_search = LocalSearchBasedMH(
            lower_bound_heuristic, construction_heuristic, local_search_procedure, feasibility_checker, solution_extender;
            neighborhood_search, timelimit=settings[:timelimit], max_iter=settings[:max_iter], 
            next_improvement=settings[:next_improvement], record_swap_history=false, max_restarts=settings[:max_restarts],
            sparse_evaluation=settings[:sparse_evaluation], score_based_sampling,
            )
    
    results = Dict()

    for inst in settings[:graph]
        results[inst.name] = (inst.graph, Int[], Real[])
        for i = 1:settings[:runs_per_instance]
            println("Start run $i/$(settings[:runs_per_instance]) for $(inst.name)")
            t = @elapsed solution, _ = run_lsbmh(local_search, inst.graph)
            push!(results[inst.name][2], length(solution))
            push!(results[inst.name][3], t)
            println("Found solution: $(sort(solution)), runtime $t")
        end
        println("Best: $(maximum(results[inst.name][2])), Avg: $(sum(results[inst.name][2]) / length(results[inst.name][2]))")
        println()
    end

    if settings[:write_result] != "-"
        df = DataFrame(GraphID=String[], GNNID=String[], V=Int[], E=Int[], Dens=Real[], γ=Real[], Avg=Real[], Best=Int[], Runtime=Real[])
        for (graph_id, (graph, sols, runtimes)) in results
            push!(df, (
                graph_id,
                settings[:scoring_function],
                nv(graph),
                ne(graph),
                density(graph),
                γ,
                mean(sols),
                maximum(sols),
                median(runtimes),
            ))
        end
        path_to_write = joinpath(settings[:write_result], "results.csv")
        CSV.write(path_to_write, df)
    end

end

run_mqcp(scoring_function)