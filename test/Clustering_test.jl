using Test
#using DelimitedFiles, LinearAlgebra

import Random:seed!
seed!(123)

using BetaML.Clustering


println("*** Testing Clustering...")

# ==================================
# New test
# ==================================
println("Testing initRepreserntative...")

Z₀ = initRepresentatives([1 10.5;1.5 10.8; 1.8 8; 1.7 15; 3.2 40; 3.6 32; 3.6 38],2,initStrategy="given",Z₀=[1.7 15; 3.6 40])

@test isapprox(Z₀,[1.7  15.0; 3.6  40.0])

# ==================================
# New test
# ==================================
println("Testing kmeans...")

(clIdx,Z) = kmeans([1 10.5;1.5 10.8; 1.8 8; 1.7 15; 3.2 40; 3.6 32; 3.3 38; 5.1 -2.3; 5.2 -2.4],3)

@test clIdx == [2, 2, 2, 2, 3, 3, 3, 1, 1]
#@test (clIdx,Z) .== ([2, 2, 2, 2, 3, 3, 3, 1, 1], [5.15 -2.3499999999999996; 1.5 11.075; 3.366666666666667 36.666666666666664])

# ==================================
# New test
# ==================================
println("Testing kmedoids...")
(clIdx,Z) = kmedoids([1 10.5;1.5 10.8; 1.8 8; 1.7 15; 3.2 40; 3.6 32; 3.3 38; 5.1 -2.3; 5.2 -2.4],3,initStrategy="grid")
@test clIdx == [2, 2, 2, 2, 3, 3, 3, 1, 1]

# ==================================
# New test
# ==================================
println("Testing emGMM...")

clusters = emGMM([1 10.5;1.5 0; 1.8 8; 1.7 15; 3.2 40; 0 0; 3.3 38; 0 -2.3; 5.2 -2.4],3,msgStep=0,missingValue=0)
@test isapprox(clusters.BIC,-39.7665224029492)

# ==================================
# New test
# ==================================
println("Testing emGMM...")
out = collFilteringGMM([1 10.5;1.5 0; 1.8 8; 1.7 15; 3.2 40; 0 0; 3.3 38; 0 -2.3; 5.2 -2.4],3,msgStep=0,missingValue=0)
@test isapprox(out.X̂[2,2],14.177888746691615)
