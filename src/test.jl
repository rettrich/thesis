using Graphs
using thesis.Instances
using thesis.LocalSearch
using StatsBase
using BenchmarkTools

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

test()