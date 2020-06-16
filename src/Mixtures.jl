using Statistics, LinearAlgebra, Distributions

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
