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
@test computeDensity(data,support) == [2,3,2]

# ----------------------------------------
support = [0,1,2,3]
data    = [339,455,180,26]
θhat    = sum(data .* support)/(sum(data)*3)
out     = goodnessOfFitDiscrete(data,support,Binomial(3,θhat),compressedData=true,α=0.05,d=1)
@test out.testValue  ≈  0.8828551921498722
@test out.p_value  ≈  0.643117653187048


#----------------------------------------

f₀   = Uniform(0,1)
data = [0.8,0.7,0.4,0.7,0.2]
out  = ksTest(data,f₀;α=0.05)
@test out.testValue  ≈  0.6708203932499368
@test out.p_value    ≈  0.009598291426747618

# --------------------------------------

#f₀ = Exponential(10)
#f₀ = Normal(0,1)
#f₀ = Uniform(0,10)
f₀ = Normal(0,1)
repetitions = 1000
outs = fill(false,repetitions)
for rep in 1:repetitions
    local data = rand(f₀ ,31)
    local out  = ksTest(data,f₀;α=0.05)
    outs[rep] = out.rejectedH₀
end
@test isapprox(sum(outs)/repetitions,0.05,atol=0.05)

#=
# -------------------------
function computeKSTableValue(f₀,N,α,repetitions=1000)
    Ts = Array{Float64,1}(undef,repetitions)
    for rep in 1:repetitions
        data       = sort(rand(f₀,N))
        N          = length(data)
        cdfhat     = collect(0:N) ./ N
        maxDist    = 0.0
        for (n,x) in enumerate(data)
            dist = max(abs(cdfhat[n]-cdf(f₀,x)), abs(cdfhat[n+1]-cdf(f₀,x)))
            if dist > maxDist
                maxDist = dist
            end
        end
        T          = sqrt(N) * maxDist
        Ts[rep]    = T
    end
    Ts = sort(Ts)
    return Ts[Int(ceil((1-α)*repetitions))]/sqrt(N)
end
(N,α,f₀) = 7,0.05,Normal(20,20)
computeKSTableValue(f₀,N,α,1000000) * sqrt(N)
quantile(Kolmogorov(),1-α)
Distributions.quantile_bisect(KSDist(N),1-α) *sqrt(N)
=#
