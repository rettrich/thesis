settings_cfg = ArgParseSettings()

@add_arg_table! settings_cfg begin
    "--graph"
        help = "Vector of paths to graph instances, separated by comma"
        arg_type = Vector
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
        default = 10
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
        default = 1000
    "--max_restarts"
        help = "Execution stops after max_restarts restarts of the local search."
        arg_type = Int
        default = 10
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
    "--sparse_evaluation"
        help = "If set to false, the NN evaluates the vertices in each iteration. If set to true, the NN only evaluates "*
               "the vertices in case no improving solution was found in the last iteration."
        arg_type = Bool
        default = false
    "--runs_per_instance"
        help = "Number of runs per instance. Output is averaged over all runs."
        arg_type = Int
        default = 1
    "--neighborhood_size"
        help = "Limit the size of sets X, Y of the restricted neighborhood to this size"
        arg_type = Int
        default = 20
    "--score_based_sampling"
        help = "Use score based sampling when using GNN if no improving move was found"
        arg_type = Bool
        default = false
end

function ArgParse.parse_item(::Type{Vector}, x::AbstractString)
    return [(name="$(split(file, "/")[end])", graph=load_instance(file)) for file in split(x, ",")]
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

