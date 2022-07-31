using ArgParse
using Graphs
using thesis.Instances
using thesis.LocalSearch
using CSV
using DataFrames

# To run, provide a graph instance and target density γ, e.g.:
# julia --project=. .\src\mqcp.jl --graph="DIMACS/brock400_1.clq" --gamma=0.999 --timelimit=60

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
            default = 5000
        "--next_improvement"
            help = "If true: Search neighborhoods with next improvement strategy, otherwise use best improvement"
            arg_type = Bool
            default = true
        "--debug"
            help = "Enables debug output"
            arg_type = Bool
            default = false
        "--write_result"
            help = "Write result to file"
            arg_type = Bool
            default = false
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

function run_mqcp()
    parsed_args = parse_commandline()

    if parsed_args["debug"]
        ENV["JULIA_DEBUG"] = "thesis"
    end
    
    graph = load_instance("$(parsed_args["graph"])")
    γ = parsed_args["gamma"]

    construction_heuristic_settings = ConstructionHeuristicSettings(
                                    parsed_args["p"], # parameter p of exploration construction 
                                    parsed_args["alpha"], # GRASP parameter α
                                    parsed_args["beta"],   # beamwidth β of initial construction
                                    parsed_args["expansion_limit"],  # expansion_limit of initial construction
                                    GreedyCompletionHeuristic() # guidance function of initial construction
                                    )
    Stm_Type = get_stm(parsed_args["stm"])
    short_term_memory = Stm_Type(graph)
    settings = LocalSearchSettings(graph; 
                                   construction_heuristic_settings, 
                                   short_term_memory,
                                   timelimit=parsed_args["timelimit"], 
                                   max_iter=parsed_args["max_iter"], 
                                   next_improvement=parsed_args["next_improvement"], 
                                   )

    println("Settings: ")
    for (arg,val) in parsed_args
        println("  $arg  =>  $val")
    end
    solution = run_MQCP(graph, γ; settings)

    println(solution)
    println("size of solution: $(length(solution))")

    if parsed_args["write_result"]
        df = DataFrame(GraphID=String[], V=Int[], E=Int[], Dens=Real[], γ=Real[], Result=Int[])
        push!(df, (
            parsed_args["graph"],
            nv(graph),
            ne(graph),
            density(graph),
            γ,
            length(solution)
        ))
        CSV.write("$(split(parsed_args["graph"], "/")[end]).csv", df)
    end

end

run_mqcp()