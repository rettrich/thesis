using Graphs
using Revise
using thesis.Instances
using thesis.LocalSearch
using thesis.LookaheadSearch
using StatsBase
using thesis.NodeRepresentationLearning
using Plots, TSne
using Word2Vec
# include("MPModels.jl")
# using .MPModels

ENV["JULIA_DEBUG"] = "thesis"

# struct Node
#     obj_val::Int
#     in::Set{Int} # set of nodes added to candidate solution
#     out::Set{Int} # set of nodes removed from candidate solution
# end

# function Base.isless(a::Node, b::Node)
#     return a.obj_val < b.obj_val
# end

# function Base.isequal(a::Node, b::Node)
#     return issetequal(a.in, b.in) && issetequal(a.out, b.out)
# end

# Base.hash(a::Node, h::UInt) = hash(a.in, hash(a.out, hash(:Node, h)))

# function test()
#     S = Set{Node}()
#     node_1 = Node(123, Set([1,2,3]), Set([3,4,5]))
#     node_2 = Node(123, Set([1,2,5]), Set([3,4,5]))
#     node_3 = Node(123, Set([1,2,3]), Set([3,4,5]))

#     push!(S, node_1)
#     push!(S, node_2)

#     println(S)

#     println("contains node_3? $(node_3 in S)")

#     println("node1 == node3? $(node_1 == node_3)")
#     println("node1 isequal node3? $(isequal(node_1, node_3))")
# end

function test()
    graph = load_instance("inst/DIMACS/brock800_3.clq")
    greedy_heu = GreedyCompletionHeuristicPQVariant()
    # sum_heu = SumOfNeighborsHeuristic(100, 0.5)
    γ = 0.9

    t_greedy = @elapsed begin
        S_greedy = lower_bound_heuristic(graph, γ, greedy_heu; β=5, expansion_limit=10)
    end
    # t_sum = @elapsed begin
    #     S_sum = lower_bound_heuristic(graph, γ, sum_heu; β=10, expansion_limit=50)
    # end
    # @time S_const = construction_heuristic(g, 80, rand(0:10, 800); p=0.5)
    # S_random = sample(1:800, 80; replace=false)
    # println("density constr: $(density(induced_subgraph(g, S_const)[1]))")
    # println("density random: $(density(induced_subgraph(g, S_random)[1]))")

    println("greedy_heu length: $(length(S_greedy)), runtime: $t_greedy")
    # println("sum_heu length: $(length(S_sum)), runtime: $t_sum")

end

function test_local_search()
    graph = load_instance("inst/DIMACS/brock800_3.clq")
    γ = 0.9

    construction_heuristic_settings = ConstructionHeuristicSettings(
                                    0.2, # parameter p of exploration construction
                                    0.2, # GRASP parameter α
                                    1,   # beamwidth β of initial construction
                                    50,  # expansion_limit of initial construction
                                    GreedyCompletionHeuristic() # guidance function of initial construction
                                    )
    short_term_memory = TabuList(graph)
    settings = LocalSearchSettings(graph;
                                   construction_heuristic_settings,
                                   short_term_memory,
                                   timelimit=60.0,
                                   max_iter=4000,
                                   next_improvement=true,
                                   )

    solution = run_MQCP(graph, γ; settings)

    println("solution length: $(length(solution))")
end

function barbell_and_path_graph(clique_size, path_length)
    g = barbell_graph(clique_size, clique_size)
    rem_edge!(g, clique_size, clique_size+1)
    add_vertices!(g, path_length)
    first_path_vertex = clique_size * 2 + 1
    for v in first_path_vertex:(nv(g)-1)
        add_edge!(g, v, v+1)
    end
    add_edge!(g, clique_size, first_path_vertex)
    add_edge!(g, clique_size+1, nv(g))
    println("diameter: $(diameter(g))")
    println("vertices: $(nv(g))")
    return g
end

function test_struct2vec()
    g = barbell_and_path_graph(10, 10)
    rws = RandomWalkSimulator(80, 5)
    learn_embeddings_word2vec(rws, g; walks_per_node=20, embedding_size=64)
end

function test_lookaheads(d::Int)
    upper_bound(g, γ) = floor(Int, 0.5 + 0.5*sqrt(1 + 8*ne(g)/γ))
    g = load_instance("inst/DIMACS/brock400_1.clq")
    γ = 0.999
    # guidance_func = GreedyCompletionHeuristic()
    # lower_bound_heuristic = BeamSearch_LowerBoundHeuristic(guidance_func; β=10, γ, expansion_limit=10)
    # lower_bound = lower_bound_heuristic(g)

    # S = sample(1:nv(g), length(lower_bound); replace=false)
    S = [279, 177, 308, 184, 23, 213, 47, 368, 266, 355, 146, 3, 125, 109, 61, 181, 371, 295, 247, 359, 307, 211, 205]
    best_neighbor = [279, 177, 308, 184, 23, 213, 47, 266, 355, 146, 3, 125, 109, 61, 181, 371, 295, 247, 359, 307, 211, 205, 127]
    V_S = filter(v -> v ∉ S, vertices(g))
    k′_1 = 50
    k′_2 = 20

    d_S = Float32.(calculate_d_S(g, S))

    d_max = maximum([d_S[i] for i in V_S])
    d_min = minimum([d_S[i] for i in S])
    candidates_S = [(d_S[i], i) for i in S]
    candidates_V_S = [(d_S[i], i) for i in V_S]
    candidates_S = filter(v -> v[1] <= d_min+1, candidates_S)
    candidates_V_S = filter(v -> v[1] >= d_max-1, candidates_V_S)
    println(length(candidates_S))
    println(length(candidates_V_S))

    lookahead_1 = Ω_d_LookaheadSearchFunction(d, k′_1, k′_1)
    lookahead_2 = Ω_d_LookaheadSearchFunction(d, k′_2, k′_2)
    lookahead_3 = Ω_d_LookaheadSearchFunction(d, length(candidates_S), length(candidates_V_S))

    t1 = @elapsed sol_1 = lookahead_1(g, S; scores=d_S)
    t2 = @elapsed sol_2 = lookahead_2(g, S; scores=d_S)
    t3 = @elapsed sol_3 = lookahead_3(g, S; scores=d_S)
    println("$t3 - $(sol_3[1])")

    println(ne(induced_subgraph(g, S)[1]))
    println((;sol_1=sol_1[1], sol_2=sol_2[1], len_1=length(sol_1[2]), len_2=length(sol_2[2]), t1, t2))
    println(density(induced_subgraph(g, S)[1]))

    return (sol_1[1], sol_2[1], t1, t2)
end
# get_MQCP_model_F3(g::SimpleGraph, γ::Real; 
#                         lower_bound::Int=1, upper_bound::Int=nv(g),
#                         opt::String="Gurobi", 
#                         verbosity::Int=0, eps_int::Real=1e-6, timelimit::Real=Inf)

function compare(iter::Int, d::Int)
    correct = []
    gaps = []
    times = []
    for i = 1:iter
        l1, l2, t1, t2 = test_lookaheads(d)
        if l1 == l2 
            push!(correct, true)
        else
            push!(correct, false)
        end
        push!(gaps, (l1 - l2) / l2)
        push!(times, (t1, t2))
    end
    println("$(sum(map(x -> Int(x), correct))) of $iter correct")
    println("avg gap: $(sum(gaps)/length(gaps))")
    println("avg runtime: l1: $(sum(map(x -> x[1], times))/iter), l2: $(sum(map(x -> x[2], times))/iter)")
end

compare(1, 2)

# test()

