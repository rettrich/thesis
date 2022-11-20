using Revise
using thesis, thesis.LocalSearch, thesis.GNNs, thesis.LookaheadSearch
using Distributions
using Dates
using BSON
using TensorBoardLogger: TBLogger
using Logging
using ArgParse
using MHLib

# ENV["JULIA_DEBUG"] = "thesis"

include("train_args.jl")

parse_settings!([settings_cfg, thesis.NodeRepresentationLearning.settings_cfg],
                vcat(ARGS,
                [
                    "--feature_set=Node2Vec_2_4-Struct2Vec",
                    # "--lookahead_depth=1",
                    # "--lookahead_breadth=50",
                    "--epochs=250",
                    # "--sparse_evaluation=false",
                    # "--debug=true"
                    "--gamma=0.95",
                    "--V=800,50",
                    "--density=0.3,0.4",
                    "--neighborhood_size=20",
                    "--nr_embedding_size=64",
                    "--buffer_capacity=1000",
                    "--num_samples=50",
                    # "--num_solutions=10",
                ]))

if settings[:debug]
    ENV["JULIA_DEBUG"] = "thesis"
end

function train_MQCP()
    start_time = time()
    γ = settings[:gamma]

    # initialize components of local search based metaheuristic
    
    # lower bound heuristic: beam search with GreedyCompletionHeuristic as guidance function
    guidance_func = FeasibleNeighborsHeuristic(false)
    lower_bound_heuristic = BeamSearch_LowerBoundHeuristic(guidance_func; β=5, γ, expansion_limit=10)

    # use a single vertex as lower bound 
    # lower_bound_heuristic = SingleVertex_LowerBoundHeuristic()

    construction_heuristic = Freq_GRASP_ConstructionHeuristic(0.2, 0.3)

    short_term_memory = TabuList()

    feasibility_checker = MQCP_FeasibilityChecker(γ)
    
    solution_extender = MQCP_GreedySolutionExtender(γ)

    # gnn = SimpleGNN(2, [64, 64, 64])
    # scoring_function = SimpleGNN_ScoringFunction(gnn, 20)

    feature_set = parse_feature_set(settings[:feature_set])

    gnn = Encoder_Decoder_GNNModel([64, 64, 64], [32, 32]; 
                                   encoder_factory=GATv2Conv_GNNChainFactory(128, 8), 
                                   node_features=feature_set, 
                                   decoder_features=[d_S_NodeFeature()],
                                   )
    scoring_function = Encoder_Decoder_ScoringFunction(gnn, settings[:neighborhood_size])

    # compare with baseline
    baseline_scoring_function = d_S_ScoringFunction()

    # local search procedure uses short term memory and scoring function
    local_search_procedure = MQCP_LocalSearchProcedure(γ, short_term_memory, scoring_function)
    baseline_local_search_procedure = MQCP_LocalSearchProcedure(γ, short_term_memory, baseline_scoring_function)

    timelimit = 300.0
    max_iter = 2000
    next_improvement = false
    record_swap_history = true
    max_restarts = 3 # abort after fixed number of restarts to save time

    # local search with gnn
    local_search = LocalSearchBasedMH(
            lower_bound_heuristic, construction_heuristic, local_search_procedure, feasibility_checker, solution_extender;
            timelimit, max_iter, next_improvement, record_swap_history, max_restarts, sparse_evaluation=settings[:sparse_evaluation])

    # baseline for comparison
    baseline_local_search = LocalSearchBasedMH(
        lower_bound_heuristic, construction_heuristic, baseline_local_search_procedure, feasibility_checker, solution_extender;
        timelimit, max_iter, next_improvement, record_swap_history=false, max_restarts)

    # instance generator settings
    V = settings[:V]
    dens = settings[:density]
    instance_generator = Training.InstanceGenerator(
                            Normal(V[1], V[2]), 
                            Uniform(dens[1], dens[2]); 
                            ensure_connectivity=settings[:ensure_connectivity],
                            )

    tblogger = nothing
    run_id = replace("MQCP-$(repr(MIME"text/plain"(), gnn))-feature_set=$(settings[:feature_set])-$(repr(MIME"text/plain"(), instance_generator))-" * 
                        string(now()) * tempname(".")[3:end], ":"=>"-")
    logdir = joinpath("./$(settings[:dir])", run_id)
    tblogger = TBLogger(logdir)

    println("$run_id")

    lookahead_func = parse_neighborhood_size()

    Training.train!(local_search, instance_generator, gnn; 
                    lookahead_func, epochs=settings[:epochs], 
                    baseline=baseline_local_search, 
                    num_samples=settings[:num_samples], 
                    batchsize=settings[:batchsize], 
                    num_batches=settings[:num_batches], 
                    warm_up=settings[:warm_up], 
                    buffer_capacity=settings[:buffer_capacity],
                    logger=tblogger)

    BSON.@save "./$(settings[:dir])/$run_id.bson" gnn

    println("Total duration: $(time()- start_time)")
    
    nothing
end

train_MQCP()
