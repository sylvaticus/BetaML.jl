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
println("Testing mixture initialisation and log-pdf...")

m1 = SphericalGaussian()
m2 = SphericalGaussian([1.1,2,3])
m3 = SphericalGaussian(nothing,10.2)
mixtures = [m1,m2,m3]
X = [1 10 20; 1.2 12 missing; 3.1 21 41; 2.9 18 39; 1.5 15 25]
initMixtures!(mixtures,X,minVariance=0.25)
@test sum([sum(m.μ) for m in mixtures]) ≈ 102.2
@test sum([sum(m.σ²) for m in mixtures]) ≈ 19.651086419753085
mask = [true, true, false]
@test lpdf(m1,X[2,:][mask],mask) ≈ -3.818323669882357

m1 = DiagonalGaussian()
m2 = DiagonalGaussian([1.1,2,3])
m3 = DiagonalGaussian(nothing,[0.1,11,25.0])
mixtures = [m1,m2,m3]
initMixtures!(mixtures,X,minVariance=0.25)
@test sum([sum(m.σ²) for m in mixtures]) ≈ 291.27933333333334
@test lpdf(m1,X[2,:][mask],mask) ≈ -3.4365786131066063

m1 = FullGaussian()
m2 = FullGaussian([1.1,2,3])
m3 = FullGaussian(nothing,[0.1 0.2 0.5; 0 2 0.8; 1 0 5])
mixtures = [m1,m2,m3]
initMixtures!(mixtures,X,minVariance=0.25)
@test sum([sum(m.σ²) for m in mixtures]) ≈ 264.77933333333334
@test lpdf(m1,X[2,:][mask],mask) ≈ -3.4365786131066063

# ==================================
# New test
# ==================================
println("Testing em...")
clusters = em([1 10.5;1.5 missing; 1.8 8; 1.7 15; 3.2 40; missing missing; 3.3 38; missing -2.3; 5.2 -2.4],3,verbosity=NONE)
@test isapprox(clusters.BIC,119.04816608007282)

# ==================================
# New test
# ==================================
println("Testing em...")
X = [1 10.5;1.5 missing; 1.8 8; 1.7 15; 3.2 40; missing missing; 3.3 38; missing -2.3; 5.2 -2.4]
out = predictMissing(X,3,mixtures=[SphericalGaussian() for i in 1:3],verbosity=NONE)
@test isapprox(out.X̂[2,2],14.187187936786232)

X = [1 10.5;1.5 missing; 1.8 8; 1.7 15; 3.2 40; missing missing; 3.3 38; missing -2.3; 5.2 -2.4]
out2 = predictMissing(X,3,mixtures=[DiagonalGaussian() for i in 1:3],verbosity=NONE)
@test out2.X̂[2,2] == 11.438358350316872

X = [1 10.5;1.5 missing; 1.8 8; 1.7 15; 3.2 40; missing missing; 3.3 38; missing -2.3; 5.2 -2.4]
out3 = predictMissing(X,3,mixtures=[FullGaussian() for i in 1:3],verbosity=NONE)
@test out3.X̂[2,2] == 11.166652292936876
