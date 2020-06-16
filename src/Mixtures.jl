using Statistics, LinearAlgebra, Distributions, PDMats


export SphericalGaussian, DiagonalGaussian, FullGaussian,
initVariances!, initMixtures!,lpdf,updateVariances!,lpdf


abstract type AbstractGaussian <: Mixture end

mutable struct SphericalGaussian{T <:Number} <: AbstractGaussian
        μ  ::Union{Array{T,1},Nothing}
        σ² ::Union{T,Nothing}
        SphericalGaussian(;μ::Union{Array{T,1},Nothing}=nothing,σ²::Union{T,Nothing}=nothing) where {T} = new{T}(μ,σ²)
        SphericalGaussian(type::Type{T}=Float64) where {T} = new{T}(nothing, nothing)
end

mutable struct DiagonalGaussian{T <:Number} <: AbstractGaussian
    μ::Union{Array{T,1},Nothing}
    σ²::Union{Array{T,1},Nothing}
    DiagonalGaussian(;μ::Union{Array{T,1},Nothing}=nothing,σ²::Union{Array{T,1},Nothing}=nothing) where {T} = new{T}(μ,σ²)
    DiagonalGaussian(::Type{T}=Float64) where {T} = new{T}(nothing, nothing)
end

mutable struct FullGaussian{T <:Number} <: AbstractGaussian
    μ::Union{Array{T,1},Nothing}
    σ²::Union{Array{T,2},Nothing}
    FullGaussian(;μ::Union{Array{T,1},Nothing}=nothing,σ²::Union{Array{T,2},Nothing}=nothing) where {T} = new{T}(μ,σ²)
    FullGaussian(::Type{T}=Float64) where {T} = new{T}(nothing, nothing)
end


function initVariances!(mixtures::Array{T,1}, X; minVariance=0.25) where {T <: SphericalGaussian}
    (N,D) = size(X)
    K = length(mixtures)
    varX_byD = fill(0.0,D)
    for d in 1:D
      varX_byD[d] = var(skipmissing(X[:,d]))
    end
    varX = max(minVariance,mean(varX_byD)/K^2)

    for (i,m) in enumerate(mixtures)
        if isnothing(m.σ²)
            m.σ² = varX
        end
    end
end

function initVariances!(mixtures::Array{T,1}, X; minVariance=0.25) where {T <: DiagonalGaussian}
    (N,D) = size(X)
    K = length(mixtures)
    varX_byD = fill(0.0,D)
    for d in 1:D
      varX_byD[d] = max(minVariance, var(skipmissing(X[:,d])))
    end

    for (i,m) in enumerate(mixtures)
        if isnothing(m.σ²)
            m.σ² = varX_byD
        end
    end

end

function initVariances!(mixtures::Array{T,1}, X; minVariance=0.25) where {T <: FullGaussian}
    (N,D) = size(X)
    K = length(mixtures)
    varX_byD = fill(0.0,D)
    for d in 1:D
      varX_byD[d] = max(minVariance, var(skipmissing(X[:,d])))
    end

    for (i,m) in enumerate(mixtures)
        if isnothing(m.σ²)
            m.σ² = diagm(varX_byD)
        end
    end
end

function initMixtures!(mixtures::Array{T,1}, X; minVariance=0.25, initStrategy="grid") where {T <: AbstractGaussian}
    (N,D) = size(X)
    K = length(mixtures)

    minX = fill(-Inf,D)
    maxX = fill(Inf,D)

    for d in 1:D
       minX[d]  = minimum(skipmissing(X[:,d]))
       maxX[d]  = maximum(skipmissing(X[:,d]))
    end

    # count nothing mean mixtures
    nMM = 0
    for (i,m) in enumerate(mixtures)
        if isnothing(m.μ)
            nMM += 1
        end
    end

    rangedμ = zeros(nMM,D)
    for d in 1:D
        rangedμ[:,d] = collect(range(minX[d], stop=maxX[d], length=nMM))
    end

    j = 1
    for m in mixtures
       if isnothing(m.μ)
           m.μ = rangedμ[j,:]
           j +=1
       end
    end

    initVariances!(mixtures,X,minVariance=minVariance)

end


function lpdf(m::SphericalGaussian,x,mask)
    μ  = m.μ[mask]
    σ² = m.σ²
    #d = IsoNormal(μ,ScalMat(length(μ),σ²))
    #return logpdf(d,x)
    return (- (length(x)/2) * log(2π*σ²)  -  norm(x-μ)^2/(2σ²))
end

function lpdf(m::DiagonalGaussian,x,mask)
    μ  = m.μ[mask]
    σ² = m.σ²[mask]
    d = DiagNormal(μ,PDiagMat(σ²))
    return logpdf(d,x)
end

function lpdf(m::FullGaussian,x,mask)
    μ   = m.μ[mask]
    nmd = length(μ)
    σ²  = reshape(m.σ²[mask*mask'],(nmd,nmd))
    d   = FullNormal(μ,PDMat(σ²))
    return logpdf(d,x)
end

function updateVariances!(mixtures::Array{T,1}, X, post, mask; minVariance=0.25) where {T <: SphericalGaussian}
    # debug stuff..
    #X = [1 10 20; 1.2 12 missing; 3.1 21 41; 2.9 18 39; 1.5 15 25]
    #m1 = SphericalGaussian(μ=[1.0,15,21],σ²=5.0)
    #m2 = SphericalGaussian(μ=[3.0,20,30],σ²=10.0)
    #mixtures= [m1,m2]
    #post = [0.9 0.1; 0.8 0.2; 0.1 0.9; 0.1 0.9; 0.4 0.6]
    #mask = [true true true; true true false; true true true; true true true; true true true]

#    #σ² = [sum([pⱼₓ[n,j] * norm(X[n,:]-μ[j,:])^2 for n in 1:N]) for j in 1:K ] ./ (nⱼ .* D)
#    for k in 1:K
#        den = dot(XdimCount,pⱼₓ[:,k])
#        nom = 0.0
#        for n in 1:N
#            if any(XMask[n,:])
#                nom += pⱼₓ[n,k] * norm(X[n,XMask[n,:]]-μ[k,XMask[n,:]])^2
#            end
#        end
#        if(den> 0 && (nom/den) > minVariance)
#            σ²[k] = nom/den
#        else
#            σ²[k] = minVariance
#        end
#    end

end

#https://github.com/davidavdav/GaussianMixtures.jl/blob/master/src/train.jl
function updateParameters!(mixtures::Array{T,1}, X, post, mask; minVariance=0.25) where {T <: AbstractGaussian}
    # debug stuff..
    #X = [1 10 20; 1.2 12 missing; 3.1 21 41; 2.9 18 39; 1.5 15 25]
    #m1 = SphericalGaussian(μ=[1.0,15,21],σ²=5.0)
    #m2 = SphericalGaussian(μ=[3.0,20,30],σ²=10.0)
    #mixtures= [m1,m2]
    #post = [0.9 0.1; 0.8 0.2; 0.1 0.9; 0.1 0.9; 0.4 0.6]
    #mask = [true true true; true true false; true true true; true true true; true true true]

    (N,D) = size(X)
    K = length(mixtures)

    nⱼ = sum(post,dims=1)'
    n  = sum(nⱼ)
    pⱼ = nⱼ ./ n

    # updating μ...
    for k in 1:K
        m = mixtures[k]
        for d in 1:D
            nᵢⱼ = sum(post[mask[:,d],k])
            if nᵢⱼ > 1
                m.μ[d] = sum(post[mask[:,d],k] .* X[mask[:,d],d])/nᵢⱼ
            end
        end
    end

    updateVariances!(mixtures, X, post, mask; minVariance=minVariance)
end
