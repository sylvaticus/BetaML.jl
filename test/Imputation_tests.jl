using Test
#using Pipe
using BetaML

TESTRNG = FIXEDRNG # This could change...


println("*** Testing Imputations...")

# ------------------------------------------------------------------------------

println("Testing MeanImputer...")

X = [2 missing 10; 20 40 100]
imputerResults = impute(MeanImputer(),X)
x̂ = imputerResults |> imputed
@test x̂[1,2] == 40
@test imputedValues(imputerResults)[1][1,2] == 40.0
@test typeof(imputedValues(imputerResults)[1]) == Matrix{Real}
@test imputerResults|> info == (nImputedValues =1,)

X = [2.0 missing 10; 20 40 100]
mod = MeanImputer(0.5,2)
x̂ = impute(mod,X) |> imputed
@test isapprox(x̂[1,2],21.58333333333)
@test typeof(x̂) == Matrix{Float64}

# ------------------------------------------------------------------------------

println("Testing GMMImputer...")

X = [1 10.5;1.5 missing; 1.8 8; 1.7 15; 3.2 40; missing missing; 3.3 38; missing -2.3; 5.2 -2.4]

mod = GMMImputer(mixtures=[SphericalGaussian() for i in 1:3],verbosity=NONE,initStrategy="grid",rng=copy(TESTRNG))
imputerResults = impute(mod,X)
x̂ = imputerResults |> imputed
@test isapprox(x̂[2,2],14.155186593170251)

mod = GMMImputer(mixtures=[DiagonalGaussian() for i in 1:3],verbosity=NONE,initStrategy="grid",rng=copy(TESTRNG))
imputerResults = impute(mod,X)
x̂ = imputerResults |> imputed
@test isapprox(x̂[2,2],14.588514438886131)

mod = GMMImputer(mixtures=[FullGaussian() for i in 1:3],verbosity=NONE,initStrategy="grid",rng=copy(TESTRNG))
imputerResults = impute(mod,X)
x̂ = imputerResults |> imputed
@test x̂[2,2] ≈ 11.166652292936876

X = [2 missing 10; 2000 4000 1000; 2000 4000 10000; 3 5 12 ]
mod = GMMImputer(K=2,multipleImputations=3,rng=copy(TESTRNG),verbosity=NONE, initStrategy="kmeans")
imputerResults = impute(mod,X)
x̂ = imputerResults |> imputed
X̂s = imputerResults |> imputedValues
x̂ = imputerResults |> imputed