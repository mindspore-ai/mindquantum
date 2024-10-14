using DelimitedFiles
using MaxCut
using PyCall
using CSV
using DataFrames

stim = pyimport("stim")

function read_matrices_from_csv(file_path)
    matrices = []
    current_matrix = []
    open(file_path, "r") do file
        for line in eachline(file)
            if isempty(strip(line))
                if !isempty(current_matrix)
                    push!(matrices, vcat(current_matrix...))
                    current_matrix = []
                end
            else
                push!(current_matrix, parse.(Float64, split(line, ","))')
            end
        end
        if !isempty(current_matrix)
            push!(matrices, vcat(current_matrix...))
        end
    end
    return matrices
end

function save_list_to_csv(data::Vector{T}, file_path::String) where T
    df = DataFrame(Value = data)
    CSV.write(file_path, df)
end

function qubo(matrix)
    positions = []
    values = []
    for i in 1:size(matrix, 1)
        for j in i+1:size(matrix, 2)
            push!(positions, (i-1, j-1))
            push!(values, matrix[i, j])
        end
    end
    return positions, values
end

function halfexpec(lst,ham,n,weights)
    circ = stim.Circuit()
    for i in lst
         circ.append("X", [i-1])
    end
    sim = stim.TableauSimulator()
    sim.do(circ)
    exp_sin = [sim.peek_observable_expectation(stim.PauliString(term)) for term in ham]
    exp = sum([exp_sin[i]*weights[i] / 2 for i in 1:n]) / 2
    return exp
end

function expec(par,matrix)
    lst1, lst2 = par
    edges, weights = qubo(matrix)
    ham = []
    n = length(weights)
    for edge in edges
        i, j = edge
        push!(ham, "Z$i*Z$j")
    end
    exp1 = halfexpec(lst1,ham,n,weights)
    exp2 = halfexpec(lst2,ham,n,weights)
    return exp1 + exp2
end

function GW(matrix, num)

    cut, partition = maxcut(matrix,iter=num,tol=0)
    exp = expec(partition,matrix)

    return exp
end



qubits_list = [150,200]

for i in qubits_list
    println("qubits = $i")
    expectation = []
    file_path = "instance_5we_$i.csv"  
    matrices = read_matrices_from_csv(file_path)

    for mat in matrices
        exp = GW(mat,1)
        println(exp)
        push!(expectation, exp)
    end

    save_list_to_csv(expectation, "data_GW_we_$i.csv")
end


