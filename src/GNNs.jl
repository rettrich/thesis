module GNNs

using Flux, Graphs, GraphNeuralNetworks, CUDA
using Statistics
using thesis.NodeRepresentationLearning
# using BSON
# using thesis
# using Statistics
# using Printf
# using MLUtils
# using Logging

export GNNModel, SimpleGNN, Encoder_Decoder_GNNModel, compute_node_features, device,
    NodeFeature, d_S_NodeFeature, DegreeNodeFeature, DeepWalkNodeFeature, EgoNetNodeFeature, 
    PageRankNodeFeature, Node2VecNodeFeature, Struct2VecNodeFeature, 
    get_feature_list, get_decoder_features,
    GNNChainFactory, ResGatedGraphConv_GNNChainFactory, GATv2Conv_GNNChainFactory,
    batch_support, evaluate,
    loss_func_unbatched

# no trailing commas in export!

# device = CUDA.functional() ? Flux.gpu : Flux.cpu
device = Flux.cpu

include("NodeFeatures.jl")

abstract type GNNModel end

Flux.params(gnn::GNNModel) = Flux.params(gnn.model)

(gnn::GNNModel)(gnn_graph::GNNGraph, inputs::AbstractMatrix) = gnn.model(gnn_graph, inputs) # TODO: only relevant for SimpleGNN

batch_support(gnn::GNNModel)::Bool = gnn.batch_support

get_feature_list(gnn::GNNModel) = gnn.node_features
get_decoder_features(gnn::GNNModel) = nothing
get_loss(gnn::GNNModel) = gnn.loss

AddResidual(l) = Parallel(+, Base.identity, l) # residual connection

abstract type ChainFactory end

"""
    GNNChainFactory

Can be used to create a `GNNChain` of a specific type, e.g. `ResGatedGraphConv` or `GATv2Conv` networks.

"""
abstract type GNNChainFactory <: ChainFactory end

"""
    (::GNNChainFactory)(d_in::Int, dims::Vector{Int}; add_classifier::Bool)

Creates a `GNNChain` of a specific type with a first layer of size `d_in` => `dims[1]`, 
second layer of size `dims[1]` => `dims[2]`, etc. If `add_classifier` is set to true, a Dense layer of size `dims[end] => 1` 
with a sigmoid classifier is added as a final layer. 

"""
(::GNNChainFactory)(d_in::Int, dims::Vector{Int}; add_classifier::Bool)::GNNChain = error("ModelFactory: Abstract Functor called")

struct ResGatedGraphConv_GNNChainFactory <: GNNChainFactory end

function (::ResGatedGraphConv_GNNChainFactory)(d_in::Int, dims::Vector{Int}; add_classifier::Bool = false)::GNNChain
    @assert length(dims) >= 1

    inner_layers_gcn = (AddResidual(ResGatedGraphConv(dims[i] => dims[i+1], relu)) for i in 1:(length(dims)-1))

    inner_layers_batch_norm = (BatchNorm(dims[i+1]) for i in 1:(length(dims)-1))

    inner_layers = collect(Iterators.flatten(zip(inner_layers_gcn, inner_layers_batch_norm)))
    
    model = GNNChain(
        ResGatedGraphConv(d_in => dims[1], relu),
        BatchNorm(dims[1]),
        inner_layers...,
    )

    if add_classifier
        model = GNNChain(
            model,
            Dense(dims[end] => 1, sigmoid),
        )
    end
    
    return model
end

struct GATv2Conv_GNNChainFactory <: GNNChainFactory 
    ff_dim::Int
    heads::Int

    function GATv2Conv_GNNChainFactory(ff_dim::Int=512, heads::Int=1)
        new(ff_dim, heads)
    end
end

"""
    (::GATv2Conv_GNNChainFactory)(d_in::Int, dims::Vector{Int}; add_classifier::Bool = false, ff_dim::Int = 512, heads=8)

Implementation of the GNN architecture from Kool et al 2019 paper "Attention, learn to solve routing problems"
"""
function (factory::GATv2Conv_GNNChainFactory)(d_in::Int, dims::Vector{Int}; add_classifier::Bool = false)::GNNChain
    @assert length(dims) >= 1
    ff_dim = factory.ff_dim
    heads = factory.heads

    gatv2_layers = (GNNChain(AddResidual(GATv2Conv(dims[i] => dims[i+1]; heads, concat=false)), BatchNorm(dims[i+1])) for i in 1:(length(dims)-1))
    
    feed_forward_sublayers = (GNNChain(AddResidual(GNNChain(Dense(dims[i+1] => ff_dim, relu), Dense(ff_dim => dims[i+1]))), BatchNorm(dims[i+1])) for i in 1:length(dims)-1)
    
    inner_layers = collect(Iterators.flatten(zip(gatv2_layers, feed_forward_sublayers)))
    
    model = GNNChain(
            Dense(d_in, dims[1]),
            # BatchNorm(dims[1]), 
            inner_layers...,
    )

    if add_classifier
        model = GNNChain(
            model,
            Dense(dims[end] => 1, sigmoid),
        )
    end

    return model
end

struct Dense_ChainFactory <: ChainFactory end

"""
    Dense_classifier(d_in, dims)

Simple Dense Feed Forward Network with specified dimensions and sigmoid classifier at the final layer. 
"""
function (::Dense_ChainFactory)(d_in::Int, dims::Vector{Int}; add_classifier::Bool = true)::Chain
    @assert length(dims) >= 1

    inner_dense = (Dense(dims[i] => dims[i+1], relu) for i in 1:(length(dims)-1))
    inner_batchnorm = (BatchNorm(dims[i+1]) for i in 1:(length(dims)-1))
    inner_layers = collect(Iterators.flatten(zip(inner_dense, inner_batchnorm)))

    model = Chain(
            Dense(d_in => dims[1]),
            BatchNorm(dims[1]),
            inner_layers...,
            Dense(dims[end] => 1, sigmoid)
    )
    return model
end

"""
    SimpleGNN

A graph neural network that classifies nodes in a single forward pass by applying multiple graph convolutional layers. 
Its corresponding ScoringFunction type `SimpleGNN_ScoringFunction` is only used for testing purposes as it is very slow. 
Use `Encoder_Decoder_GNNModel` and its corresponding ScoringFunction type instead. 

"""
struct SimpleGNN <: GNNModel
    num_layers::Int # number of layers
    d_in::Int # dimension of node feature vectors
    dims::Vector{Int} # output dimensions of GCN layers (dims[i] is output dim of layer i)
    model::GNNChain
    node_features::Vector{<:NodeFeature}
    gnn_type::String
    batch_support::Bool
    loss # loss function for training
    opt

    function SimpleGNN(dims::Vector{Int}; 
                       node_features::Vector{<:NodeFeature}=[DegreeNodeFeature(), d_S_NodeFeature()],
                       loss = Flux.logitbinarycrossentropy,
                       opt=Adam(0.001, (0.9, 0.999)), 
                       model_factory::GNNChainFactory = ResGatedGraphConv_GNNChainFactory(),
                       )
        d_in = sum(map(x -> length(x), node_features))
        model = model_factory(d_in, dims; add_classifier=true) |> device

        loss_func(g::GNNGraph) = loss( vec(model(g, g.ndata.x)), g.ndata.y)

        gnn_type = split(string(typeof(model_factory)), "_")[1]

        batch_support = true

        new(length(dims), d_in, dims, model, node_features, gnn_type, batch_support, loss_func, opt)
    end
end

Base.show(io::IO, ::MIME"text/plain", x::SimpleGNN) = print(io, "$(x.gnn_type)-$(x.d_in)-$(join(x.dims, "-"))")

"""
    Encoder_Decoder_GNNModel

A deep neural network based on the encoder / decoder paradigm. The encoder is a GNNChain which should be used to compute 
node embeddings for a graph. During each iteration, the context (based on the current candidate solution) is derived from 
node embeddings and fed through the simpler decoder Chain, which is e.g. a small Dense feed forward network. 

"""
struct Encoder_Decoder_GNNModel <: GNNModel
    d_in::Int
    encoder_dims::Vector{Int}
    decoder_dims::Vector{Int}
    encoder::GNNChain # more expensive gnn chain
    decoder::Chain    # linear time decoder 
    node_features::Vector{<:NodeFeature} # encoder features
    decoder_features::Union{Nothing, Vector{<:NodeFeature}} # decoder features
    gnn_type::String
    batch_support::Bool
    loss
    opt

    function Encoder_Decoder_GNNModel(encoder_dims::Vector{Int}, decoder_dims::Vector{Int};
                      encoder_factory::GNNChainFactory = ResGatedGraphConv_GNNChainFactory(),
                      decoder_factory::ChainFactory = Dense_ChainFactory(),  
                      node_features::Vector{<:NodeFeature} = [DegreeNodeFeature()],
                      decoder_features::Union{Nothing, Vector{<:NodeFeature}} = nothing,
                      loss = Flux.binarycrossentropy,
                      opt = Adam(0.001, (0.9, 0.999)),
                      batch_support = false
                      )

        # input dimension is sum of dimensions of node features
        d_in = sum(map(x -> length(x), node_features))

        # encoder: GNN, compute node embeddings
        encoder = encoder_factory(d_in, encoder_dims) |> device

        # compute decoder input dimensions
        decoder_feature_len = !isnothing(decoder_features) ? sum(map(x -> length(x), decoder_features)) : 0
        decoder_in = decoder_feature_len + encoder_dims[end]*2

        # decoder from node embeddings + context embedding, used to classify node
        decoder = decoder_factory(decoder_in, decoder_dims) |> device

        gnn_type = "$(split(string(typeof(encoder_factory)), "_")[1])-$(split(string(typeof(decoder_factory)), "_")[1])"

        # loss_func = batch_support ? loss_func_batched : loss_func_unbatched

        new(d_in, encoder_dims, decoder_dims, encoder, decoder, node_features, decoder_features, gnn_type, batch_support, loss, opt)
    end
end

function loss_func_unbatched(gnn::Encoder_Decoder_GNNModel, g::GNNGraph, S::Vector{Int})
    node_embeddings = gnn.encoder(g, g.ndata.x)
    context = repeat(mean(NNlib.gather(node_embeddings, S), dims=2), 1, nv(g))
    if isnothing(gnn.decoder_features)
        decoder_input = vcat(node_embeddings, context)
    else
        decoder_input = vcat(node_embeddings, context, g.ndata.decoder_features)
    end
    output = gnn.decoder(decoder_input) 
    gnn.loss(vec(output), g.ndata.y)
end

# This does not work on the GPU for now, as `getgraph` is not gpu friendly at the moment: 
# https://github.com/CarloLucibello/GraphNeuralNetworks.jl/issues/161
function loss_func_batched(gnn::Encoder_Decoder_GNNModel, batch::GNNGraph) # graphs are batched by taking union of several graphs which are all disconnected          
    node_embeddings = gnn.encoder(batch, batch.ndata.x) # compute node embeddings for all graphs in g

    offset = 0
    context_size = (size(node_embeddings, 1) + decoder_feature_len, size(node_embeddings, 2))
    context_embeddings = fill(0f0, context_size) # context embeddings have same size as node embeddings
    
    for i in 1:batch.num_graphs 
        g = GNNGraph(0) # make g available in current scope

        # loop over graphs that are batched and compute context for each graph
        # context is the mean of all node embeddings of vertices in candidate solution S
        Flux.ChainRulesCore.@ignore_derivatives g = getgraph(batch, [i])
        
        # obtain column indices for features of vertices in S
        S = filter(v -> g.ndata.in_S[v]==1, 1:nv(g))

        # gather mean from corresponding columns in embeddings and repeat for each node in g
        context = repeat(mean(NNlib.gather(node_embeddings[:, (1+offset):(nv(g)+offset)], S), dims=2), 1, nv(g))

        # append decoder features
        if !isnothing(gnn.decoder_features)
            context = vcat(context, g.ndata.decoder_features)
        end
        
        # write back to context embedding matrix
        NNlib.scatter!(+, context_embeddings, context, collect((1+offset):(nv(g)+offset)))
        offset += nv(g) # increase offset, as vertices are always numbered from 1
    end
    decoder_input = vcat(node_embeddings, context_embeddings)
    output = gnn.decoder(decoder_input) 
    gnn.loss(vec(output), batch.ndata.y) 
end

get_decoder_features(gnn::Encoder_Decoder_GNNModel) = gnn.decoder_features

Flux.params(gnn::Encoder_Decoder_GNNModel) = Flux.params(gnn.encoder, gnn.decoder)

Base.show(io::IO, ::MIME"text/plain", x::Encoder_Decoder_GNNModel) = 
    print(io, "$(x.gnn_type)-$(x.d_in)-$(join(x.encoder_dims, "-"))-$(join(x.decoder_dims, "-"))")

encode(gnn::Encoder_Decoder_GNNModel, graph::GNNGraph, embeddings::AbstractMatrix) = gnn.encoder(graph, embeddings)

add_context(embeddings::AbstractMatrix, S) = vcat(embeddings, repeat(mean(NNlib.gather(embeddings, collect(S)), dims=2), 1, size(embeddings, 2)))

function decode(gnn::Encoder_Decoder_GNNModel, graph::GNNGraph, embeddings::AbstractMatrix, S, d_S)
    decoder_feature_list = get_decoder_features(gnn)
    if !isnothing(decoder_feature_list)
        decoder_features = compute_node_features(decoder_feature_list, graph, S, d_S)
        embeddings = vcat(embeddings, decoder_features)
    end
    return gnn.decoder(embeddings)
end

"""
    evaluate(gnn::GNNModel, graph, embeddings, S, d_S)

Simulates a run through the GNN by providing node embeddings and a candidate solution for a graph.

"""
evaluate(gnn::GNNModel, graph::GNNGraph, embeddings::AbstractMatrix, S::Union{Vector{Int}, Set{Int}}, d_S::Vector{Int}) = 
    error("Evaluate called on abstract GNNModel")


"""
    evaluate(gnn::Encoder_Decoder_GNNModel, graph, embeddings, S, d_S)

Simulates a run through the entire Encoder/Decoder architecture: The encoder is run on the feature embeddings, then the context is computed 
and finally the decoder features are added and the embeddings are run evaluated by the decoder. 
Returns a vector of scores for the vertices in the graph. 

"""
function evaluate(gnn::Encoder_Decoder_GNNModel, graph::GNNGraph, embeddings::AbstractMatrix, S::Union{Vector{Int}, Set{Int}}, d_S::Vector{Int})
    node_embeddings = add_context(encode(gnn, graph, embeddings), S)
    return decode(gnn, graph, node_embeddings, S, d_S)
end


"""

Get context embedding of dimension d from node_embeddings, which is a d x n matrix 

"""
function get_context_embeddings(node_embeddings, in_S::Vector{Int})
    repeat(d_S, 1)
    mean(embeddings[:, S], dims=2)
end


Flux.params(gnn::SimpleGNN) = Flux.params(gnn.model)

(gnn::SimpleGNN)(gnn_graph::GNNGraph, inputs::AbstractMatrix) = gnn.model(gnn_graph, inputs)


end # module