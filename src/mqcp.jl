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

# To run, provide a graph instance and target density γ, e.g.:
# julia --project=. .\src\mqcp.jl --graph="inst/DIMACS/brock400_1.clq" --gamma=0.999 --timelimit=60


settings = ArgParseSettings()

@add_arg_table settings begin
    "--graph"
        help = "Graph instance"
        arg_type = String
        required = true
        default = "random"
    "--gamma"
        help = "Target density γ"
        arg_type = Float64
        required = true
    "--stm"
        help = "Short term memory type ∈ ['tl', 'cc']"
        arg_type = String
        default = "tl"
    "--alpha"
        help = "GRASP parameter for construction heuristic, 0 ≤ α ≤ 1"
        arg_type = Float64
        default = 0.2
    "--beta"
        help = "Beamwidth of lower bound heuristic"
        arg_type = Int
        default = 5
    "--expansion_limit"
        help = "Limit expansion of nodes in beam search construction"
        arg_type = Int
        default = 50
    "--p"
        help = "Controls how much rarely visited vertices are preferred in construction heuristic, 0 ≤ p ≤ 1"
        arg_type = Float64
        default = 0.4
    "--timelimit"
        help = "Sets timelimit for execution"
        arg_type = Float64
        default = 1000.0
    "--max_iter"
        help = "Search is restarted after max_iter iterations without finding an improving solution"
        arg_type = Int
        default = 4000
    "--next_improvement"
        help = "If true: Search neighborhoods with next improvement strategy, otherwise use best improvement"
        arg_type = Bool
        default = true
    "--debug"
        help = "Enables debug output"
        arg_type = Bool
        default = false
    "--write_result"
        help = "Write result to file if directory is specified"
        arg_type = String
        default = "-"
    "--scoring_function"
        help = "Load scoring function (GNN) from file"
        arg_type = String
        default = "-"
    "--vnd_depth"
        help = "Depth d of variable neighborhood search"
        arg_type = Int
        default = 1
end

parsed_args = parse_args(
    [
        ARGS..., 
        "--graph=inst/DIMACS/brock400_1.clq",
        "--gamma=0.999",
        # "--scoring_function=random",
        "--debug=true",
        "--vnd_depth=2",
        "--max_iter=50",
        "--timelimit=60.0",
    ], 
    settings)

function get_stm(stm_string)
    if stm_string == "tl"
        return TabuList
    elseif stm_string == "cc"
        return ConfigurationChecking
    else
        error("unknown short term memory type")
    end
end

function run_mqcp(scoring_function=nothing; parsed_args)

    if parsed_args["debug"]
        ENV["JULIA_DEBUG"] = "thesis"
    end

    if parsed_args["graph"] == "random"
        graph = generate_instance(200, 0.5)
    else
        graph = load_instance("$(parsed_args["graph"])")
    end
    γ = parsed_args["gamma"]

    # initialize components of local search based metaheuristic
    
    # lower bound heuristic: beam search with GreedyCompletionHeuristic as guidance function
    guidance_func = GreedyCompletionHeuristic()
    # lower_bound_heuristic = BeamSearch_LowerBoundHeuristic(guidance_func; β=parsed_args["beta"], γ, expansion_limit=parsed_args["expansion_limit"])
    lower_bound_heuristic = SingleVertex_LowerBoundHeuristic()

    construction_heuristic = Freq_GRASP_ConstructionHeuristic(parsed_args["alpha"], parsed_args["p"])

    Stm_Type = get_stm(parsed_args["stm"])
    short_term_memory = Stm_Type()

    feasibility_checker = MQCP_FeasibilityChecker(γ)
    
    solution_extender = MQCP_GreedySolutionExtender(γ)

    neighborhood_search = VariableNeighborhoodDescent(parsed_args["vnd_depth"])

    # 
    if isnothing(scoring_function)
        scoring_function = d_S_ScoringFunction()
    end

    # local search procedure uses short term memory and scoring function
    local_search_procedure = MQCP_LocalSearchProcedure(γ, short_term_memory, scoring_function)
    println("ScoringFunction: $(typeof(scoring_function))")

    timelimit = parsed_args["timelimit"]
    max_iter = parsed_args["max_iter"]
    next_improvement = parsed_args["next_improvement"]
    record_swap_history = false

    local_search = LocalSearchBasedMH(
            lower_bound_heuristic, construction_heuristic, local_search_procedure, feasibility_checker, solution_extender;
            neighborhood_search, timelimit, max_iter, next_improvement, record_swap_history)

    println("Settings: ")
    for (arg,val) in parsed_args
        println("  $arg  =>  $val")
    end
    solutions = []
    for i = 1:1
        solution, swap_history = run_lsbmh(local_search, graph)
        push!(solutions, solution)
        println(solution)
        println("size of solution: $(length(solution))")
    end

    if parsed_args["write_result"] != "-"
        df = DataFrame(GraphID=String[], V=Int[], E=Int[], Dens=Real[], γ=Real[], Result=Real[])
        push!(df, (
            parsed_args["graph"],
            nv(graph),
            ne(graph),
            density(graph),
            γ,
            length(sum(solutions) / length(solutions)))
        )
        CSV.write("$(parsed_args["write_result"])/$(parsed_args["scoring_function"])-$(split(parsed_args["graph"], "/")[end]).csv", df)
    end

end

scoring_function = nothing
if parsed_args["scoring_function"] != "-"
    if parsed_args["scoring_function"] == "random"
        println("Using random scoring function")
        scoring_function = Random_ScoringFunction(20)
    elseif parsed_args["scoring_function"] == "encoder-decoder"
        gnn = Encoder_Decoder_GNNModel([64, 64, 64], [32, 32])
        scoring_function = Encoder_Decoder_ScoringFunction(gnn, 20)
        BSON.@save "test-gnn.bson" gnn
    else
        println("Load scoring function $(parsed_args["scoring_function"])")
        BSON.@load parsed_args["scoring_function"] gnn
        scoring_function = SimpleGNN_ScoringFunction(gnn, 20)
    end
end

run_mqcp(scoring_function; parsed_args)