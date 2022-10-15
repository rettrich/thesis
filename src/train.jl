using thesis, thesis.LocalSearch, thesis.GNNs
using Distributions
using Dates
using BSON
using TensorBoardLogger: TBLogger
using Logging
using ArgParse

# ENV["JULIA_DEBUG"] = "thesis"

settings = ArgParseSettings()
@add_arg_table settings begin
    "--feature_set"
        help = "Define input features. Possible values: EgoNet_%d , Degree, Pagerank, DeepWalk, Node2Vec_%f_%f" * 
               "Multiple features can be specified, separated by a '-' (e.g. Degree-EgoNet1-DeepWalk)."
        arg_type = String
        default = "EgoNet1"
    "--dir"
        help = "Directory where logs and models are stored"
        arg_type = String
        default = "logs"
end

parsed_args = parse_args(
    [
        ARGS..., 
        "--feature_set=Node2Vec_2_0.25"
    ], 
    settings)

function parse_feature_set(feature_string)::Vector{<:NodeFeature}
    features = split(feature_string, "-")
    feature_set = []
    for feature in features
        if startswith(feature, "EgoNet")
            d = parse(Int, feature[end])
            push!(feature_set, EgoNetNodeFeature(d))
        elseif startswith(feature, "Degree")
            push!(feature_set, DegreeNodeFeature())
        elseif startswith(feature, "Pagerank")
            push!(feature_set, PageRankNodeFeature())
        elseif startswith(feature, "DeepWalk")
            push!(feature_set, DeepWalkNodeFeature())
        elseif startswith(feature, "Node2Vec")
            p = parse(Float32, split(feature, "_")[2])
            q = parse(Float32, split(feature, "_")[3])
            push!(feature_set, Node2VecNodeFeature(p, q))
        else
            error("Unknown feature '$feature'")
        end
    end
    feature_set
end

function train_MQCP(parsed_args::Dict{String, Any})
    start_time = time()
    γ = 0.999

    # initialize components of local search based metaheuristic
    
    # lower bound heuristic: beam search with GreedyCompletionHeuristic as guidance function
    # guidance_func = GreedyCompletionHeuristic()
    # lower_bound_heuristic = BeamSearch_LowerBoundHeuristic(guidance_func; β=1, γ, expansion_limit=10)

    # use a single vertex as lower bound 
    lower_bound_heuristic = SingleVertex_LowerBoundHeuristic()

    construction_heuristic = Freq_GRASP_ConstructionHeuristic(0.2, 0.3)

    short_term_memory = TabuList()

    feasibility_checker = MQCP_FeasibilityChecker(γ)
    
    solution_extender = MQCP_GreedySolutionExtender(γ)

    # gnn = SimpleGNN(2, [64, 64, 64])
    # scoring_function = SimpleGNN_ScoringFunction(gnn, 20)

    feature_set = parse_feature_set(parsed_args["feature_set"])

    gnn = Encoder_Decoder_GNNModel([64, 64, 64], [32, 32]; 
                                   encoder_factory=GATv2Conv_GNNChainFactory(128), 
                                   node_features=feature_set, 
                                   decoder_features=[d_S_NodeFeature()]
                                   )
    scoring_function = Encoder_Decoder_ScoringFunction(gnn, 20)

    # compare with baseline
    baseline_scoring_function = d_S_ScoringFunction()

    # local search procedure uses short term memory and scoring function
    local_search_procedure = MQCP_LocalSearchProcedure(γ, short_term_memory, scoring_function)
    baseline_local_search_procedure = MQCP_LocalSearchProcedure(γ, short_term_memory, baseline_scoring_function)

    timelimit = 120.0
    max_iter = 4000
    next_improvement = false
    record_swap_history = true
    max_restarts = 1 # abort after fixed number of restarts to save time

    # local search with gnn
    local_search = LocalSearchBasedMH(
            lower_bound_heuristic, construction_heuristic, local_search_procedure, feasibility_checker, solution_extender;
            timelimit, max_iter, next_improvement, record_swap_history, max_restarts)

    # baseline for comparison
    baseline_local_search = LocalSearchBasedMH(
        lower_bound_heuristic, construction_heuristic, baseline_local_search_procedure, feasibility_checker, solution_extender;
        timelimit, max_iter, next_improvement, record_swap_history=false, max_restarts)

    instance_generator = Training.InstanceGenerator(Normal(200, 15), Uniform(0.4, 0.6))

    tblogger = nothing
    run_id = replace("MQCP-$(repr(MIME"text/plain"(), gnn))-feature_set=$(parsed_args["feature_set"])-$(repr(MIME"text/plain"(), instance_generator))-" * 
                        string(now()) * tempname(".")[3:end], ":"=>"-")
    logdir = joinpath("./$(parsed_args["dir"])", run_id)
    tblogger = TBLogger(logdir)

    println("$run_id")

    Training.train!(local_search, instance_generator, gnn; epochs=300, baseline=baseline_local_search, logger=tblogger)

    BSON.@save "./$(parsed_args["dir"])/$run_id.bson" gnn

    println("Total duration: $(time()- start_time)")
    
    nothing
end

train_MQCP(parsed_args)
# parse_feature_set(parsed_args["feature_set"])
