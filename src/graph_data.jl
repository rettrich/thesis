using Graphs
using thesis.Instances
using DataFrames
using CSV

ENV["JULIA_DEBUG"] = "thesis"

# instances = load_instances("inst/DIMACS")

# df = DataFrame(Graph=String[], V=Int[], E=Int[], dens=Real[])
# for (graph_id, graph) in instances
#     push!(df, (
#         graph_id, 
#         nv(graph),
#         ne(graph),
#         density(graph)
#     ))
# end

# sort!(df, [:Graph])
# println(df)

# CSV.write("graph_data.csv", df)

df = DataFrame(CSV.File("graph_data.csv"; stringtype=String))

open("DIMACS_table.txt", "w") do io
    write(io, string(latexify(df)));
end