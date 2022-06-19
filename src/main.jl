using Graphs
using thesis
using thesis.Instances


g = generate_instance(10, 0.7)
for e in edges(g)
    println(e)
end