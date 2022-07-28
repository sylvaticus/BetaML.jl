
"""
initRepresentatives(X,K;initStrategy,Z₀)

Initialisate the representatives for a K-Mean or K-Medoids algorithm

# Parameters:
* `X`: a (N x D) data to clusterise
* `K`: Number of cluster wonted
* `initStrategy`: Whether to select the initial representative vectors:
* `random`: randomly in the X space
* `grid`: using a grid approach [default]
* `shuffle`: selecting randomly within the available points
* `given`: using a provided set of initial representatives provided in the `Z₀` parameter
* `Z₀`: Provided (K x D) matrix of initial representatives (used only together with the `given` initStrategy) [default: `nothing`]
* `rng`: Random Number Generator (see [`FIXEDSEED`](@ref)) [deafult: `Random.GLOBAL_RNG`]

# Returns:
* A (K x D) matrix of initial representatives

# Example:
```julia
julia> Z₀ = initRepresentatives([1 10.5;1.5 10.8; 1.8 8; 1.7 15; 3.2 40; 3.6 32; 3.6 38],2,initStrategy="given",Z₀=[1.7 15; 3.6 40])
```
"""
function initRepresentatives(X,K;initStrategy="grid",Z₀=nothing,rng = Random.GLOBAL_RNG)
  X  = makeMatrix(X)
  (N,D) = size(X)
  # Random choice of initial representative vectors (any point, not just in X!)
  minX = minimum(X,dims=1)
  maxX = maximum(X,dims=1)
  Z = zeros(K,D)
  if initStrategy == "random"
      for i in 1:K
          for j in 1:D
              Z[i,j] = rand(rng,Distributions.Uniform(minX[j],maxX[j]))
          end
      end
  elseif initStrategy == "grid"
      for d in 1:D
              # same "space" for each class on each dimension
              Z[:,d] = collect(range(minX[d] + (maxX[d]-minX[d])/(K*2) , stop=maxX[d] - (maxX[d]-minX[d])/(K*2)  , length=K))
              #ex: collect(range(minX[d], stop=maxX[d], length=K))
              #collect(range(s+(e-s)/(K*2), stop=e-(e-s)/(K*2), length=K))
      end
  elseif initStrategy == "given"
      if isnothing(Z₀) error("With the `given` strategy you need to provide the initial set of representatives in the Z₀ parameter.") end
      Z₀ = makeMatrix(Z₀)
      Z = Z₀
  elseif initStrategy == "shuffle"
      zIdx = shuffle(rng,1:size(X)[1])[1:K]
      Z = X[zIdx, :]
  else
      error("initStrategy \"$initStrategy\" not implemented")
  end
  return Z
end

function classAssignation(X,Z,dist)
  cIdx      = zeros(Int64,size(X,1))
  for (i,x) in enumerate(eachrow(X))
      cost = Inf
      for (k,z) in enumerate(eachrow(Z))
         if (dist(x,z)  < cost)
             cost    =  dist(x,z)
             cIdx[i] = k
         end
      end
  end
  return cIdx
end

function updateKMeansRepresentatives!(Z,X,cIdx)
  K,D = size(Z)
  for j in  1:K
      Cⱼ = X[cIdx .== j,:] # Selecting the constituency by boolean selection
      if size(Cⱼ)[1] > 0
          Z[j,:] = sum(Cⱼ,dims=1) ./ size(Cⱼ)[1]
      else
          # move toward the center if no costituency
          xAvg = mean(X,dims=1)'
          Z[j,:] = Z[j,:] .+ ((xAvg - Z[j,:]) .* 0.01)
      end
  end
end

## Basic K-Means Algorithm (Lecture/segment 13.7 of https://www.edx.org/course/machine-learning-with-python-from-linear-models-to)

"""
kmeans(X,K;dist,initStrategy,Z₀)

Compute K-Mean algorithm to identify K clusters of X using Euclidean distance

# Parameters:
* `X`: a (N x D) data to clusterise
* `K`: Number of cluster wonted
* `dist`: Function to employ as distance (see notes). Default to Euclidean distance.
* `initStrategy`: Whether to select the initial representative vectors:
* `random`: randomly in the X space
* `grid`: using a grid approach [default]
* `shuffle`: selecting randomly within the available points
* `given`: using a provided set of initial representatives provided in the `Z₀` parameter
* `Z₀`: Provided (K x D) matrix of initial representatives (used only together with the `given` initStrategy) [default: `nothing`]
* `rng`: Random Number Generator (see [`FIXEDSEED`](@ref)) [deafult: `Random.GLOBAL_RNG`]

# Returns:
* A tuple of two items, the first one being a vector of size N of ids of the clusters associated to each point and the second one the (K x D) matrix of representatives

# Notes:
* Some returned clusters could be empty
* The `dist` parameter can be:
* Any user defined function accepting two vectors and returning a scalar
* An anonymous function with the same characteristics (e.g. `dist = (x,y) -> norm(x-y)^2`)
* One of the above predefined distances: `l1_distance`, `l2_distance`, `l2²_distance`, `cosine_distance`

# Example:
```julia
julia> (clIdx,Z) = kmeans([1 10.5;1.5 10.8; 1.8 8; 1.7 15; 3.2 40; 3.6 32; 3.3 38; 5.1 -2.3; 5.2 -2.4],3)
```
"""
function kmeans(X,K;dist=(x,y) -> norm(x-y),initStrategy="grid",Z₀=nothing,verbosity=STD,rng = Random.GLOBAL_RNG)
  X     = makeMatrix(X)
  (N,D) = size(X)
  # Random choice of initial representative vectors (any point, not just in X!)
  minX  = minimum(X,dims=1)
  maxX  = maximum(X,dims=1)
  Z₀ = initRepresentatives(X,K,initStrategy=initStrategy,Z₀=Z₀,rng=rng)
  Z  = Z₀
  cIdx_prev = zeros(Int64,N)

  # Looping
  while true
      # Determining the constituency of each cluster
      cIdx = classAssignation(X,Z,dist)

      # Determining the new representative by each cluster
      # for (j,z) in enumerate(eachrow(Z))
      updateKMeansRepresentatives!(Z,X,cIdx)

      # Checking termination condition: clusters didn't move any more
      if cIdx == cIdx_prev
          return (cIdx,Z)
      else
          cIdx_prev = cIdx
      end
  end
end

function updateKMedoidsRepresentatives!(Z,X,cIdx,dist)
  K,D = size(Z)
  for j in  1:K
      Cⱼ = X[cIdx .== j,:] # Selecting the constituency by boolean selection
      nⱼ = size(Cⱼ)[1]     # Size of the cluster
      if nⱼ == 0 continue end # empty continuency. Let's not do anything. Stil in the next batch other representatives could move away and points could enter this cluster
      bestCost = Inf
      bestCIdx = 0
      for cIdx in 1:nⱼ      # candidate index
           candidateCost = 0.0
           for tIdx in 1:nⱼ # target index
               candidateCost += dist(Cⱼ[cIdx,:],Cⱼ[tIdx,:])
           end
           if candidateCost < bestCost
               bestCost = candidateCost
               bestCIdx = cIdx
           end
      end
      Z[j,:] = reshape(Cⱼ[bestCIdx,:],1,D)
  end
end

## Basic K-Medoids Algorithm (Lecture/segment 14.3 of https://www.edx.org/course/machine-learning-with-python-from-linear-models-to)
"""
kmedoids(X,K;dist,initStrategy,Z₀)

Compute K-Medoids algorithm to identify K clusters of X using distance definition `dist`

# Parameters:
* `X`: a (n x d) data to clusterise
* `K`: Number of cluster wonted
* `dist`: Function to employ as distance (see notes). Default to Euclidean distance.
* `initStrategy`: Whether to select the initial representative vectors:
* `random`: randomly in the X space
* `grid`: using a grid approach
* `shuffle`: selecting randomly within the available points [default]
* `given`: using a provided set of initial representatives provided in the `Z₀` parameter
* `Z₀`: Provided (K x D) matrix of initial representatives (used only together with the `given` initStrategy) [default: `nothing`]
* `rng`: Random Number Generator (see [`FIXEDSEED`](@ref)) [deafult: `Random.GLOBAL_RNG`]

# Returns:
* A tuple of two items, the first one being a vector of size N of ids of the clusters associated to each point and the second one the (K x D) matrix of representatives

# Notes:
* Some returned clusters could be empty
* The `dist` parameter can be:
* Any user defined function accepting two vectors and returning a scalar
* An anonymous function with the same characteristics (e.g. `dist = (x,y) -> norm(x-y)^2`)
* One of the above predefined distances: `l1_distance`, `l2_distance`, `l2²_distance`, `cosine_distance`

# Example:
```julia
julia> (clIdx,Z) = kmedoids([1 10.5;1.5 10.8; 1.8 8; 1.7 15; 3.2 40; 3.6 32; 3.3 38; 5.1 -2.3; 5.2 -2.4],3,initStrategy="grid")
```
"""
function kmedoids(X,K;dist=(x,y) -> norm(x-y),initStrategy="grid",Z₀=nothing, verbosity=STD, rng = Random.GLOBAL_RNG)
  X  = makeMatrix(X)
  (n,d) = size(X)
  # Random choice of initial representative vectors
  Z₀ = initRepresentatives(X,K,initStrategy=initStrategy,Z₀=Z₀,rng=rng)
  Z = Z₀
  cIdx_prev = zeros(Int64,n)

  # Looping
  while true
      # Determining the constituency of each cluster
      cIdx = classAssignation(X,Z,dist)

      # Determining the new representative by each cluster (within the points member)
      #for (j,z) in enumerate(eachrow(Z))
      updateKMedoidsRepresentatives!(Z,X,cIdx,dist)

      # Checking termination condition: clusters didn't move any more
      if cIdx == cIdx_prev
          return (cIdx,Z)
      else
          cIdx_prev = cIdx
      end
  end

end


# Avi v2..

Base.@kwdef mutable struct KMeansHyperParametersSet <: BetaMLHyperParametersSet
    nClasses::Int64                   = 3
    dist::Function                    = (x,y) -> norm(x-y)
    initStrategy::String              = "Grid"
    initialRepresentatives::Union{Nothing,Matrix{Float64}} = nothing
end

Base.@kwdef mutable struct KMedoidsHyperParametersSet <: BetaMLHyperParametersSet
    nClasses::Int64                   = 3
    dist::Function                    = (x,y) -> norm(x-y)
    initStrategy::String              = "Grid"
    initialRepresentatives::Union{Nothing,Matrix{Float64}} = nothing
end
Base.@kwdef mutable struct KMeansOptionsSet <: BetaMLOptionsSet
    verbosity::Verbosity = STD
    rng                  = Random.GLOBAL_RNG
end
Base.@kwdef mutable struct KMedoidsOptionsSet <: BetaMLOptionsSet
    verbosity::Verbosity = STD
    rng                  = Random.GLOBAL_RNG
end

Base.@kwdef mutable struct KMeansLearnableParameters <: BetaMLLearnableParametersSet
    representatives::Union{Nothing,Matrix{Float64}}  = nothing
    assignments::Vector{Int64}        = Int64[]
end
Base.@kwdef mutable struct KMedoidsLearnableParameters <: BetaMLLearnableParametersSet
    representatives::Union{Nothing,Matrix{Float64}}  = nothing
    assignments::Vector{Int64}        = Int64[]
end

mutable struct KMeansModel <: BetaMLUnsupervisedModel
    hpar::KMeansHyperParametersSet
    opt::KMeansOptionsSet
    par::Union{Nothing,KMeansLearnableParameters}
    trained::Bool
    info::Dict{Symbol,Any}
end

mutable struct KMedoidsModel <: BetaMLUnsupervisedModel
    hpar::KMedoidsHyperParametersSet
    opt::KMedoidsOptionsSet
    par::Union{Nothing,KMedoidsLearnableParameters}
    trained::Bool
    info::Dict{Symbol,Any}
end


function KMeansModel(;kwargs...)
    m = KMeansModel(KMeansHyperParametersSet(),KMeansOptionsSet(),KMeansLearnableParameters(),false,Dict{Symbol,Any}())
    thisobjfields  = fieldnames(nonmissingtype(typeof(m)))
    for (kw,kwv) in kwargs
       for f in thisobjfields
          fobj = getproperty(m,f)
          if kw in fieldnames(typeof(fobj))
              setproperty!(fobj,kw,kwv)
          end
        end
    end
    return m
end

function KMedoidsModel(;kwargs...)
    m = KMedoidsModel(KMedoidsHyperParametersSet(),KMedoidsOptionsSet(),KMedoidsLearnableParameters(),false,Dict{Symbol,Any}())
    thisobjfields  = fieldnames(nonmissingtype(typeof(m)))
    for (kw,kwv) in kwargs
       for f in thisobjfields
          fobj = getproperty(m,f)
          if kw in fieldnames(typeof(fobj))
              setproperty!(fobj,kw,kwv)
          end
        end
    end
    return m
end



"""
    train!(m::KMeansModel,x)
"""
function train!(m::KMeansModel,x)

    # Parameter alias..
    K                      = m.hpar.nClasses
    dist                   = m.hpar.dist
    initStrategy           = m.hpar.initStrategy
    initialRepresentatives = m.hpar.initialRepresentatives
    verbosity              = m.opt.verbosity
    rng                    = m.opt.rng

    if m.trained
        # Note that doing this we give lot of importance to the new data, even if this is few records and the model has bee ntrained with milions of records.
        # So, training 1000 records doesn't give the same output as training 990 records and then training again with 10 records
        verbosity >= STD && @warn "Continuing training of a pre-trained model"
        (clIdx,Z) = kmeans(x,K,dist=dist,Z₀=m.par.representatives,initStrategy="given",verbosity=verbosity,rng=rng)

    else
        (clIdx,Z) = kmeans(x,K,dist=dist,initStrategy=initStrategy,Z₀=initialRepresentatives,verbosity=verbosity,rng=rng)
    end
    m.par  = KMeansLearnableParameters(representatives=Z,assignments=clIdx)

    m.info[:trainedRecords] = get(m.info,:trainedRecords,0) + size(x,1)
    m.info[:dimensions]     = size(x,2)
    m.trained=true
    return true
end   

"""
    train!(m::KMeansModel,x)
"""
function train!(m::KMedoidsModel,x)

    # Parameter alias..
    K                      = m.hpar.nClasses
    dist                   = m.hpar.dist
    initStrategy           = m.hpar.initStrategy
    initialRepresentatives = m.hpar.initialRepresentatives
    verbosity              = m.opt.verbosity
    rng                    = m.opt.rng

    if m.trained
        # Note that doing this we give lot of importance to the new data, even if this is few records and the model has bee ntrained with milions of records.
        # So, training 1000 records doesn't give the same output as training 990 records and then training again with 10 records
        verbosity >= STD && @warn "Continuing training of a pre-trained model"
        (clIdx,Z) = kmedoids(x,K,dist=dist,Z₀=m.par.representatives,initStrategy="given",verbosity=verbosity,rng=rng)

    else
        (clIdx,Z) = kmedoids(x,K,dist=dist,initStrategy=initStrategy,Z₀=initialRepresentatives,verbosity=verbosity,rng=rng)
    end
    m.par  = KMedoidsLearnableParameters(representatives=Z,assignments=clIdx)

    m.info[:trainedRecords] = get(m.info,:trainedRecords,0) + size(x,1)
    m.info[:dimensions]     = size(x,2)
    m.trained=true
    return true
end  

function predict(m::Union{KMeansModel,KMedoidsModel})
    return m.par.assignments
end

function predict(m::Union{KMeansModel,KMedoidsModel},X)
    X               = makeMatrix(X)
    representatives = m.par.representatives
    classes = classAssignation(X,representatives,m.hpar.dist)
    return classes
end

function show(io::IO, ::MIME"text/plain", m::KMeansModel)
    if m.trained == false
        print(io,"KMeansModel - A K-Means Model (untrained)")
    else
        print(io,"KMeansModel - A K-Means Model (trained on $(m.info[:trainedRecords]) records)")
    end
end

function show(io::IO, ::MIME"text/plain", m::KMedoidsModel)
    if m.trained == false
        print(io,"KMedoidsModel - A K-Medoids Model (untrained)")
    else
        print(io,"KMedoidsModel - A K-Medoids Model (trained on $(m.info[:trainedRecords]) records)")
    end
end

function show(io::IO, m::KMeansModel)
    if m.trained == false
        print(io,"KMeansModel - A $(m.hpar.nClasses)-classes K-Means Model (untrained)")
    else
        print(io,"KMeansModel - A $(m.info[:dimensions])-dimensions $(m.hpar.nClasses)-classes K-Means Model (trained on $(m.info[:trainedRecords]) records)")
        println(io,m.info)
        println(io,"Representatives:")
        println(io,m.par.representatives)
    end
end


function show(io::IO, m::KMedoidsModel)
    if m.trained == false
        print(io,"KMedoidsModel - A $(m.hpar.nClasses)-classes K-Medoids Model (untrained)")
    else
        print(io,"KMedoidsModel - A $(m.info[:dimensions])-dimensions $(m.hpar.nClasses)-classes K-Medoids Model (trained on $(m.info[:trainedRecords]) records)")
        println(io,m.info)
        println(io,"Distance function used:")
        println(io,m.hpar.dist)
        println(io,"Representatives:")
        println(io,m.par.representatives)
    end
end