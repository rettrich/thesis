module GNN

using Flux

export Encoder, Decoder

struct Encoder

end

function (::Encoder)(g, node_features)

end

struct Decoder

end

function (::Decoder)(graph_embedding, node_embeddings, context_embedding)

end

end