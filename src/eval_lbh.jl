using Revise
using Graphs
using thesis.Instances
using thesis.LocalSearch
using thesis.LookaheadSearch
using Statistics
using DataFrames, CSV
using ArgParse
using MHLib

settings_cfg = ArgParseSettings()
@add_arg_table! settings_cfg begin
    # "--instance_set"
    #     help = "Instance set to evaluate"
    #     arg_type = String
    #     required = true
    "--V"
        help = "Only evaluate graphs of this size"
        arg_type = Int
        required = true
    "--density"
        help = "Density of generated instances"
        arg_type = Float64
    "--gamma"
        help = "Gamma for evaluation"
        arg_type = Float64
        default = 0.999
    "--dir"
        help = "Directory for output"
        arg_type = String
        default = "./"
end

parse_settings!([settings_cfg], 
    vcat(ARGS, [
        # "--instance_set=BHOSLIB",
        # "--V=500",
        # "--density=0.9",
        # "--gamma=0.95",
    ]))

function load_instance_set()
    instances = load_instances("inst/$(settings[:instance_set])")
    instances = filter(el -> (nv(last(el)) == settings[:V] ), instances)
    return instances
end

function evaluate_lbh()
    # instances = load_instance_set()
    instances = [generate_instance(settings[:V], settings[:density]) for _ in 1:20]
    βs = [10, 25, 50]
    εs = [10, 25, 50]

    greedy_completion = GreedyCompletionHeuristic()
    feasible_neighbors = FeasibleNeighborsHeuristic(false)
    greedy_search = GreedySearchHeuristic()

    lower_bound_heuristics = []

    for β in βs, ε in εs
        push!(lower_bound_heuristics, BeamSearch_LowerBoundHeuristic(greedy_completion; β, γ=settings[:gamma], expansion_limit=ε))
        push!(lower_bound_heuristics, BeamSearch_LowerBoundHeuristic(feasible_neighbors; β, γ=settings[:gamma], expansion_limit=ε))
        push!(lower_bound_heuristics, BeamSearch_LowerBoundHeuristic(greedy_search; β, γ=settings[:gamma], expansion_limit=ε))
    end

    df = DataFrame(:Heuristic => String[], :Parameters => String[], Symbol("Solution Size") => Real[], :Runtime => Real[])
    for lbh in lower_bound_heuristics
        println("Running $lbh")
        for (num, graph) in enumerate(instances)
            runtime = @elapsed s = lbh(graph)
            len = length(s)
            push!(df, (
                lbh_to_string(lbh),
                "β=$(lbh.β),ε=$(lbh.expansion_limit)",
                len,
                runtime
            ))
            println("$num, graph $num/$(length(instances)), len: $(len), t: $(runtime)")
        end
    end
    path_to_write = joinpath(settings[:dir], "lbh_random-$(settings[:density])-$(settings[:V]).csv")
    CSV.write(path_to_write, df)
end

function lbh_to_string(lbh::LowerBoundHeuristic)
    if typeof(lbh.guidance_func) == GreedyCompletionHeuristic
        return "Greedy Completion"
    elseif typeof(lbh.guidance_func) == FeasibleNeighborsHeuristic
        return "Feasible Neighbors"
    elseif typeof(lbh.guidance_func) == GreedySearchHeuristic
        return "Greedy Search"
    end
    error("Unknown guidance function $(lbh.guidance_func)")
end

evaluate_lbh()