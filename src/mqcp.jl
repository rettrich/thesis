using ArgParse
using Graphs
# using thesis
using thesis.LocalSearch
using thesis.Instances
using thesis.GNNs
using CSV
using DataFrames
using Flux

# To run, provide a graph instance and target density γ, e.g.:
# julia --project=. .\src\mqcp.jl --graph="inst/DIMACS/brock400_1.clq" --gamma=0.999 --timelimit=60

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--graph"
            help = "Graph instance"
            arg_type = String
            required = true
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
    end
    return parse_args(s)
end

function get_stm(stm_string)
    if stm_string == "tl"
        return TabuList
    elseif stm_string == "cc"
        return ConfigurationChecking
    else
        error("unknown short term memory type")
    end
end

# struct LocalSearchBasedMH
#     # key components
#     lower_bound_heuristic::LowerBoundHeuristic
#     construction_heuristic::ConstructionHeuristic
#     short_term_memory::ShortTermMemory
#     scoring_function::ScoringFunction
#     local_search_procedure::LocalSearchProcedure
#     feasibility_checker::FeasibilityChecker
#     solution_extender::SolutionExtender

#     # settings
#     timelimit::Float64
#     max_iter::Int
#     next_improvement::Bool
#     record_swap_history::Bool

function run_mqcp()
    parsed_args = parse_commandline()

    if parsed_args["debug"]
        ENV["JULIA_DEBUG"] = "thesis"
    end
    
    graph = load_instance("$(parsed_args["graph"])")
    γ = parsed_args["gamma"]

    # initialize components of local search based metaheuristic
    
    # lower bound heuristic: beam search with GreedyCompletionHeuristic as guidance function
    guidance_func = GreedyCompletionHeuristic()
    bs_heuristic = BeamSearch_LowerBoundHeuristic(guidance_func; β=parsed_args["beta"], γ, expansion_limit=parsed_args["expansion_limit"])

    construction_heuristic = Freq_GRASP_ConstructionHeuristic(parsed_args["alpha"], parsed_args["p"])

    Stm_Type = get_stm(parsed_args["stm"])
    short_term_memory = Stm_Type()

    feasibility_checker = MQCP_FeasibilityChecker(γ)
    
    solution_extender = MQCP_GreedySolutionExtender(γ)

    scoring_function = d_S_ScoringFunction()
    # gnn = ResGatedGraphConvGNN(2, [20, 20])
    # scoring_function = GNN_ScoringFunction(gnn, 20)

    # local search procedure uses short term memory and scoring function
    local_search_procedure = MQCP_LocalSearchProcedure(γ, short_term_memory, scoring_function)

    timelimit = parsed_args["timelimit"]
    max_iter = parsed_args["max_iter"]
    next_improvement = parsed_args["next_improvement"]
    record_swap_history = false

    local_search = LocalSearchBasedMH(
            bs_heuristic, construction_heuristic, local_search_procedure, feasibility_checker, solution_extender;
            timelimit, max_iter, next_improvement, record_swap_history)

    println("Settings: ")
    for (arg,val) in parsed_args
        println("  $arg  =>  $val")
    end
    solution, swap_history = run_lsbmh(local_search, graph)

    println(solution)
    println("size of solution: $(length(solution))")

    if parsed_args["write_result"] != "-"
        df = DataFrame(GraphID=String[], V=Int[], E=Int[], Dens=Real[], γ=Real[], Result=Int[])
        push!(df, (
            parsed_args["graph"],
            nv(graph),
            ne(graph),
            density(graph),
            γ,
            length(solution)
        ))
        CSV.write("$(parsed_args["write_result"])/$(split(parsed_args["graph"], "/")[end]).csv", df)
    end

end

run_mqcp()