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
    "--instance_set"
        help = "Instance set to evaluate"
        arg_type = String
        required = true
    "--V"
        help = "Only evaluate graphs of this size"
        arg_type = Int
        required = true
    "--gamma"
        help = "Gamma for evaluation"
        arg_type = Float64
        default = 0.999
end

parse_settings!([settings_cfg], 
    vcat(ARGS, [
        "--instance_set=BHOSLIB",
        "--V=450",
    ]))

function load_instance_set()
    instances = load_instances("inst/$(settings[:instance_set])")
    instances = filter(el -> (nv(last(el)) == settings[:V] ), instances)
    return instances
end

function evaluate_lbh()
    instances = load_instance_set()
    βs = [1, 10, 25, 50]
    εs = [10, 25, 50]

    guidance_func = GreedyCompletionHeuristic()

    lower_bound_heuristics = []

    for β in βs, ε in εs
        push!(lower_bound_heuristics, BeamSearch_LowerBoundHeuristic(guidance_func; β, γ=settings[:gamma], expansion_limit=ε))
    end

    df = DataFrame(β=Int[], ε=Int[], len=Real[], avg_t=Real[])
    for lbh in lower_bound_heuristics
        println("Running $lbh")
        avg_len, avg_runtime = [], []
        for (num, (id, graph)) in enumerate(instances)
            runtime = @elapsed s = lbh(graph)
            len = length(s)
            push!(avg_len, len)
            push!(avg_runtime, runtime)
            println("$id, graph $num/$(length(instances)), len: $(len), t: $(runtime)")
        end
        push!(df, (
                lbh.β,
                lbh.expansion_limit,
                mean(avg_len),
                mean(avg_runtime)
        ))
    end
    CSV.write("lower_bound_heuristic_results-$(settings[:gamma])-$(settings[:V]).csv", df)
end

evaluate_lbh()