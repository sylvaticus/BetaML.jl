using Statistics, LinearAlgebra, PDMats
import Distributions: IsoNormal, DiagNormal, FullNormal, logpdf
import PDMats: ScalMat, PDiagMat, PDMat

export SphericalGaussian, DiagonalGaussian, FullGaussian,
initVariances!, initMixtures!,lpdf,updateVariances!,lpdf


abstract type AbstractGaussian <: Mixture end

mutable struct SphericalGaussian{T <:Number} <: AbstractGaussian
        μ  ::Union{Array{T,1},Nothing}
        σ² ::Union{T,Nothing}
        #SphericalGaussian(;μ::Union{Array{T,1},Nothing},σ²::Union{T,Nothing}) where {T} = SphericalGaussian(μ,σ²)
        SphericalGaussian(μ::Union{Array{T,1},Nothing},σ²::Union{T,Nothing}=nothing) where {T} = new{T}(μ,σ²)
        SphericalGaussian(type::Type{T}=Float64) where {T} = new{T}(nothing, nothing)
end

mutable struct DiagonalGaussian{T <:Number} <: AbstractGaussian
    μ::Union{Array{T,1},Nothing}
    σ²::Union{Array{T,1},Nothing}
    DiagonalGaussian(μ::Union{Array{T,1},Nothing},σ²::Union{Array{T,1},Nothing}=nothing) where {T} = new{T}(μ,σ²)
    DiagonalGaussian(::Type{T}=Float64) where {T} = new{T}(nothing, nothing)
end

mutable struct FullGaussian{T <:Number} <: AbstractGaussian
    μ::Union{Array{T,1},Nothing}
    σ²::Union{Array{T,2},Nothing}
    FullGaussian(μ::Union{Array{T,1},Nothing},σ²::Union{Array{T,2},Nothing}=nothing) where {T} = new{T}(μ,σ²)
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
    # debug..
    #X = [1 10.5;1.5 missing; 1.8 8; 1.7 15; 3.2 40; missing 2; 3.3 38; missing -2.3; 5.2 -2.4]
    #mixtures = [SphericalGaussian() for i in 1:3]
    # ---
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


npar(mixtures::Array{T,1}) where {T <: SphericalGaussian} = length(mixtures) * length(mixtures[1].μ) + length(mixtures) # K * D + K
npar(mixtures::Array{T,1}) where {T <: DiagonalGaussian}  = length(mixtures) * length(mixtures[1].μ) + length(mixtures) * length(mixtures[1].μ) # K * D + K * D
npar(mixtures::Array{T,1}) where {T <: FullGaussian} = begin K = length(mixtures); D = length(mixtures[1].μ); K * D + K * (D^2+D)/2 end

function updateVariances!(mixtures::Array{T,1}, X, pₙₖ, nkd, Xmask; minVariance=0.25) where {T <: SphericalGaussian}

    # debug stuff..
    #X = [1 10 20; 1.2 12 missing; 3.1 21 41; 2.9 18 39; 1.5 15 25]
    #m1 = SphericalGaussian(μ=[1.0,15,21],σ²=5.0)
    #m2 = SphericalGaussian(μ=[3.0,20,30],σ²=10.0)
    #mixtures= [m1,m2]
    #pₙₖ = [0.9 0.1; 0.8 0.2; 0.1 0.9; 0.1 0.9; 0.4 0.6]
    #Xmask = [true true true; true true false; true true true; true true true; true true true]
    #minVariance=0.25
    # ---

    (N,D) = size(X)
    K = length(mixtures)
    XdimCount = sum(Xmask, dims=2)

    #    #σ² = [sum([pⱼₓ[n,j] * norm(X[n,:]-μ[j,:])^2 for n in 1:N]) for j in 1:K ] ./ (nⱼ .* D)
    for k in 1:K
        nom = 0.0
        den = dot(XdimCount,pₙₖ[:,k])
        m = mixtures[k]
        for n in 1:N
            if any(Xmask[n,:])
                nom += pₙₖ[n,k] * norm(X[n,Xmask[n,:]]-m.μ[Xmask[n,:]])^2
            end
        end
        if(den> 0 && (nom/den) > minVariance)
            m.σ² = nom/den
        else
            m.σ² = minVariance
        end
    end

end

#https://github.com/davidavdav/GaussianMixtures.jl/blob/master/src/train.jl
function updateParameters!(mixtures::Array{T,1}, X, pₙₖ, Xmask; minVariance=0.25) where {T <: AbstractGaussian}
    # debug stuff..
    #X = [1 10 20; 1.2 12 missing; 3.1 21 41; 2.9 18 39; 1.5 15 25]
    #m1 = SphericalGaussian(μ=[1.0,15,21],σ²=5.0)
    #m2 = SphericalGaussian(μ=[3.0,20,30],σ²=10.0)
    #mixtures= [m1,m2]
    #pₙₖ = [0.9 0.1; 0.8 0.2; 0.1 0.9; 0.1 0.9; 0.4 0.6]
    #Xmask = [true true true; true true false; true true true; true true true; true true true]

    (N,D) = size(X)
    K = length(mixtures)

    #nₖ = sum(pₙₖ,dims=1)'
    #n  = sum(nₖ)
    #pₖ = nₖ ./ n

    #n = fill(0.0,K,D)
    nkd = [sum(pₙₖ[Xmask[:,d],k]) for k in 1:K, d in 1:D] # number of point associated to a given mixture for a specific dimension


    # updating μ...
    for k in 1:K
        m = mixtures[k]
        for d in 1:D
            #n[k,d] = sum(pₙₖ[Xmask[:,d],k])
            if nkd[k,d] > 1
                m.μ[d] = sum(pₙₖ[Xmask[:,d],k] .* X[Xmask[:,d],d])/nkd[k,d]
            end
        end
    end

    updateVariances!(mixtures, X, pₙₖ, nkd, Xmask; minVariance=minVariance)
end
