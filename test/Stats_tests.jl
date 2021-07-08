using Test

using StableRNGs
using Distributions
using BetaML

#TESTRNG = FIXEDRNG # This could change...
TESTRNG = StableRNG(123)

println("*** Testing the Stats module...")

# ==================================
# NEW TEST

println("Testing welchSatterthwaite ...")

d = welchSatterthwaite(2,2,20,20)
@test d == 38


obs                     = rand(copy(TESTRNG),Gamma(2,2),50000)
candidates              = 0:0.01:maximum(obs)
medianWithAbs           = mEstimationBruteForce(obs,candidates)
@test medianWithAbs     ≈ 3.35
medianWithHuberLoss     = mEstimationBruteForce(obs,candidates,x->huberLoss(x,0.0000000001))
@test medianWithAbs     ≈ 3.35
meanWithHuberLoss       = mEstimationBruteForce(obs,candidates,x->huberLoss(x,1000))
@test meanWithHuberLoss ≈ 3.98

# ----------------------------------------

lB = 100; uB = 200
obs = rand(copy(TESTRNG),Uniform(lB,uB),10000)
q0  = findQuantile(obs,0.2)
q0_2 = sort(obs)[Int(round(length(obs)*0.2))]
@test isapprox(q0,q0_2,atol=0.01)


# ----------------------------------------
out = goodnessOfFitDiscrete([205,26,25,19],[0.72,0.07,0.12,0.09],α=0.05)
@test out.testValue ≈ 5.889610389610388
@test out.p_value ≈  0.1171061913085063


# ----------------------------------------

data    = ["a","y","y","b","b","a","b"]
support = ["a","b","y"]

@test distribute(data,support) == [2,3,2]

# ----------------------------------------
support = [0,1,2,3]
data    = [339,455,180,26]
θhat    = sum(data .* support)/(sum(data)*3)
out     = goodnessOfFitDiscrete(data,support,Binomial(3,θhat),compressedData=true,α=0.05,d=1)
@test out.testValue == 0.8828551921498722
@test out.p_value == 0.643117653187048

# ----------------------------------------

ksdist = KSDist(7)
cdf(ksdist,0.48342)
#quantile(ksdist,0.95)
Distributions.quantile_bisect(ksdist,0.95)

kong = Kolmogorov()
quantile(kong,0.95)



f₀ = Normal(0,1)
data = rand(Normal(0.1,1) ,500)
out  = ksTest(data,f₀;α=0.05)

N          = length(data)
cdfhat     = collect(1:N) ./ N
cdftrue    = [cdf(f₀,x) for x in sort(data)]

plot(cdftrue)
plot!(cdfhat)