settings_cfg = ArgParseSettings()

@add_arg_table! settings_cfg begin
    "--feature_set"
        help = "Define input features. Possible values: EgoNet%d , Degree, Pagerank, DeepWalk, Node2Vec_%f_%f, Struct2Vec" * 
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
    "--buffer_capacity"
        help = "Capacity of replay buffer"
        arg_type = Int
        default = 1000
    "--debug"
        help = "Enable debug output"
        arg_type = Bool
        default = false
    "--gamma"
        help = "Parameter γ that defines a feasible MQC"
        arg_type = Float64
        default = 0.999
    "--V"
        help = "Specify μ and σ² for number of vertices in randomly generated instances, separated by comma, e.g. 200,15"
        arg_type = Tuple
        default = (200,15)
    "--density"
        help = "Specify upper and lower bound for density of randomly generated instances, separated by comma. e.g. 0.7,0.8"
        arg_type = Tuple
        default = (0.4,0.6)
    "--ensure_connectivity"
        help = "If set to true, only connected graphs are generated"
        arg_type = Bool
        default = false
    "--neighborhood_size"
        help = "Limit the size of sets X, Y of the restricted neighborhood to this size"
        arg_type = Int
        default = 20
    "--nr_embedding_size"
        help = "Embedding size for NodeRepresentationLearning features DeepWalk, Node2Vec, Struct2Vec"
        arg_type = Int
        default = 64
    "--num_solutions"
        help = "Lookahead search returns up to this many solutions"
        arg_type = Int
        default = typemax(Int)
end

function ArgParse.parse_item(::Type{Tuple}, x::AbstractString)
    return Tuple(parse(Float64, i) for i in split(x, ","))
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
            push!(feature_set, Node2VecNodeFeature(p, q; embedding_size=settings[:nr_embedding_size]))
        elseif startswith(feature, "Struct2Vec")
            push!(feature_set, Struct2VecNodeFeature(; embedding_size=settings[:nr_embedding_size]))
        else
            error("Unknown feature '$feature'")
        end
    end
    feature_set
end

function parse_neighborhood_size()
    if settings[:lookahead_depth] == 1
        lookahead_search = Ω_1_LookaheadSearchFunction(; num_solutions=settings[:num_solutions])
    elseif settings[:lookahead_depth] <= 3
        lookahead_search = Ω_d_LookaheadSearchFunction(settings[:lookahead_depth], settings[:lookahead_breadth]; num_solutions=settings[:num_solutions])
    else
        error("Neighborhood size $(settings[:neighborhood_size]) not supported")
    end
end