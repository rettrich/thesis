settings_cfg = ArgParseSettings()

@add_arg_table! settings_cfg begin
    "--feature_set"
        help = "Define input features. Possible values: EgoNet_%d , Degree, Pagerank, DeepWalk, Node2Vec_%f_%f, Struct2Vec" * 
               "Multiple features can be specified, separated by a '-' (e.g. Degree-EgoNet1-DeepWalk)."
        arg_type = String
        default = "EgoNet1"
    "--lookahead_depth"
        help = "Set neighborhood size for training. Neighborhoods Ω_1, Ω_2, ... Ω_lookahead_depth will be searched exhaustively." *
               "Only feasible for values 1, 2, 3."
        arg_type = Int
        default = 1
    "--lookahead_breadth"
        help = "Set size of restricted neighborhood for Ω_1, Ω_2, ... or 0 if no restriction should be applied. "
        arg_type = Int
        default = 0
    "--dir"
        help = "Directory where logs and models are stored"
        arg_type = String
        default = "logs"
    "--num_samples"
        help = "Number of samples generated per iteration"
        arg_type = Int
        default = 25
    "--num_batches"
        help = "Number of batches used to train in each epoch"
        arg_type = Int
        default = 4
    "--batchsize"
        help = "Number of samples in a batch"
        arg_type = Int
        default = 8
    "--warm_up"
        help = "For reinforcement learning: For the first `warm_up` iterations, use d_S as scoring vector for restricted neighborhood in lookahead search"
        arg_type = Int
        default = 50
    "--epochs"
        help = "Number of training epochs (iterations of training loop)"
        arg_type = Int
        default = 200
    "--sparse_evaluation"
        help = "If set to false, the NN evaluates the vertices in each iteration. If set to true, the NN only evaluates "*
               "the vertices in case no improving solution was found in the last iteration."
        arg_type = Bool
        default = false
end

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
        elseif startswith(feature, "Struct2Vec")
            push!(feature_set, Struct2VecNodeFeature())
        else
            error("Unknown feature '$feature'")
        end
    end
    feature_set
end

function parse_neighborhood_size()
    if settings[:lookahead_depth] == 1
        lookahead_search = Ω_1_LookaheadSearchFunction()
    elseif settings[:lookahead_depth] <= 3
        lookahead_search = Ω_d_LookaheadSearchFunction(settings[:lookahead_depth], settings[:lookahead_breadth])
    else
        error("Neighborhood size $(settings[:neighborhood_size]) not supported")
    end
end