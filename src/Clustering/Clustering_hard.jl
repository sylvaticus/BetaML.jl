"Part of [BetaML](https://github.com/sylvaticus/BetaML.jl). Licence is MIT."

"""
init_representatives(X,K;initialisation_strategy,initial_representatives)

Initialisate the representatives for a K-Mean or K-Medoids algorithm

# Parameters:
* `X`: a (N x D) data to clusterise
* `K`: Number of cluster wonted
* `initialisation_strategy`: Whether to select the initial representative vectors:
* `random`: randomly in the X space
* `grid`: using a grid approach [default]
* `shuffle`: selecting randomly within the available points
* `given`: using a provided set of initial representatives provided in the `initial_representatives` parameter
* `initial_representatives`: Provided (K x D) matrix of initial representatives (used only together with the `given` initialisation_strategy) [default: `nothing`]
* `rng`: Random Number Generator (see [`FIXEDSEED`](@ref)) [deafult: `Random.GLOBAL_RNG`]

# Returns:
* A (K x D) matrix of initial representatives

# Example:
```julia
julia> initial_representatives = init_representatives([1 10.5;1.5 10.8; 1.8 8; 1.7 15; 3.2 40; 3.6 32; 3.6 38],2,initialisation_strategy="given",initial_representatives=[1.7 15; 3.6 40])
```
"""
function init_representatives(X,n_classes;initialisation_strategy="grid",initial_representatives=nothing,rng = Random.GLOBAL_RNG)
  X  = makematrix(X)
  (N,D) = size(X)
  K = n_classes
  # Random choice of initial representative vectors (any point, not just in X!)
  minX = minimum(X,dims=1)
  maxX = maximum(X,dims=1)
  Z = zeros(K,D)
  if initialisation_strategy == "random"
      for i in 1:K
          for j in 1:D
              Z[i,j] = rand(rng,Distributions.Uniform(minX[j],maxX[j]))
          end
      end
  elseif initialisation_strategy == "grid"
      for d in 1:D
              # same "space" for each class on each dimension
              Z[:,d] = collect(range(minX[d] + (maxX[d]-minX[d])/(K*2) , stop=maxX[d] - (maxX[d]-minX[d])/(K*2)  , length=K))
              #ex: collect(range(minX[d], stop=maxX[d], length=K))
              #collect(range(s+(e-s)/(K*2), stop=e-(e-s)/(K*2), length=K))
      end
  elseif initialisation_strategy == "given"
      if isnothing(initial_representatives) error("With the `given` strategy you need to provide the initial set of representatives in the initial_representatives parameter.") end
      initial_representatives = makematrix(initial_representatives)
      Z = initial_representatives
  elseif initialisation_strategy == "shuffle"
      zIdx = shuffle(rng,1:size(X)[1])[1:K]
      Z = X[zIdx, :]
  else
      error("initialisation_strategy \"$initialisation_strategy\" not implemented")
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
kmeans(X,K;dist,initialisation_strategy,initial_representatives)

Compute K-Mean algorithm to identify K clusters of X using Euclidean distance

!!! warning
    This function is no longer exported.  Use `KMeansClusterer` instead. 

# Parameters:
* `X`: a (N x D) data to clusterise
* `K`: Number of cluster wonted
* `dist`: Function to employ as distance (see notes). Default to Euclidean distance.
* `initialisation_strategy`: Whether to select the initial representative vectors:
* `random`: randomly in the X space
* `grid`: using a grid approach [default]
* `shuffle`: selecting randomly within the available points
* `given`: using a provided set of initial representatives provided in the `initial_representatives` parameter
* `initial_representatives`: Provided (K x D) matrix of initial representatives (used only together with the `given` initialisation_strategy) [default: `nothing`]
* `rng`: Random Number Generator (see [`FIXEDSEED`](@ref)) [deafult: `Random.GLOBAL_RNG`]

# Returns:
* A tuple of two items, the first one being a vector of size N of ids of the clusters associated to each point and the second one the (K x D) matrix of representatives

# Notes:
* Some returned clusters could be empty
* The `dist` parameter can be:
  * Any user defined function accepting two vectors and returning a scalar
  * An anonymous function with the same characteristics (e.g. `dist = (x,y) -> norm(x-y)^2`)
  * One of the above predefined distances: `l1_distance`, `l2_distance`, `l2squared_distance`, `cosine_distance`

# Example:
```julia
julia> (clIdx,Z) = kmeans([1 10.5;1.5 10.8; 1.8 8; 1.7 15; 3.2 40; 3.6 32; 3.3 38; 5.1 -2.3; 5.2 -2.4],3)
```
"""
function kmeans(X,K;dist=(x,y) -> norm(x-y),initialisation_strategy="grid",initial_representatives=nothing,verbosity=STD,rng = Random.GLOBAL_RNG)
  X     = makematrix(X)
  (N,D) = size(X)
  # Random choice of initial representative vectors (any point, not just in X!)
  minX  = minimum(X,dims=1)
  maxX  = maximum(X,dims=1)
  initial_representatives = init_representatives(X,K,initialisation_strategy=initialisation_strategy,initial_representatives=initial_representatives,rng=rng)
  Z  = initial_representatives
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
kmedoids(X,K;dist,initialisation_strategy,initial_representatives)

Compute K-Medoids algorithm to identify K clusters of X using distance definition `dist`

!!! warning
    This function is no longer exported. Use `KMedoidsClusterer` instead. 

# Parameters:
* `X`: a (n x d) data to clusterise
* `K`: Number of cluster wonted
* `dist`: Function to employ as distance (see notes). Default to Euclidean distance.
* `initialisation_strategy`: Whether to select the initial representative vectors:
* `random`: randomly in the X space
* `grid`: using a grid approach
* `shuffle`: selecting randomly within the available points [default]
* `given`: using a provided set of initial representatives provided in the `initial_representatives` parameter
* `initial_representatives`: Provided (K x D) matrix of initial representatives (used only together with the `given` initialisation_strategy) [default: `nothing`]
* `rng`: Random Number Generator (see [`FIXEDSEED`](@ref)) [deafult: `Random.GLOBAL_RNG`]

# Returns:
* A tuple of two items, the first one being a vector of size N of ids of the clusters associated to each point and the second one the (K x D) matrix of representatives

# Notes:
* Some returned clusters could be empty
* The `dist` parameter can be:
* Any user defined function accepting two vectors and returning a scalar
* An anonymous function with the same characteristics (e.g. `dist = (x,y) -> norm(x-y)^2`)
* One of the above predefined distances: `l1_distance`, `l2_distance`, `l2squared_distance`, `cosine_distance`

# Example:
```julia
julia> (clIdx,Z) = kmedoids([1 10.5;1.5 10.8; 1.8 8; 1.7 15; 3.2 40; 3.6 32; 3.3 38; 5.1 -2.3; 5.2 -2.4],3,initialisation_strategy="grid")
```
"""
function kmedoids(X,K;dist=(x,y) -> norm(x-y),initialisation_strategy="grid",initial_representatives=nothing, verbosity=STD, rng = Random.GLOBAL_RNG)
  X  = makematrix(X)
  (n,d) = size(X)
  # Random choice of initial representative vectors
  initial_representatives = init_representatives(X,K,initialisation_strategy=initialisation_strategy,initial_representatives=initial_representatives,rng=rng)
  Z = initial_representatives
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

# ------------------------------------------------------------------------------
# Avi v2..

"""
$(TYPEDEF)

Hyperparameters for both the [`KMeansClusterer`](@ref) and [`KMedoidsClusterer`](@ref) models

# Parameters:
$(TYPEDFIELDS)
"""
Base.@kwdef mutable struct KMeansMedoidsHyperParametersSet <: BetaMLHyperParametersSet
    "Number of classes to discriminate the data [def: 3]"
    n_classes::Int64                  = 3
    "Function to employ as distance. Default to the Euclidean distance. Can be one of the predefined distances (`l1_distance`, `l2_distance`, `l2squared_distance`),  `cosine_distance`), any user defined function accepting two vectors and returning a scalar or an anonymous function with the same characteristics. Attention that the `KMeansClusterer` algorithm is not guaranteed to converge with other distances than the Euclidean one."
    dist::Function                    = (x,y) -> norm(x-y)
    """
    The computation method of the vector of the initial representatives.
    One of the following:
    - "random": randomly in the X space [default]
    - "grid": using a grid approach
    - "shuffle": selecting randomly within the available points
    - "given": using a provided set of initial representatives provided in the `initial_representatives` parameter
    """
    initialisation_strategy::String              = "grid"
    "Provided (K x D) matrix of initial representatives (useful only with `initialisation_strategy=\"given\"`) [default: `nothing`]"
    initial_representatives::Union{Nothing,Matrix{Float64}} = nothing
end


Base.@kwdef mutable struct KMeansMedoidsLearnableParameters <: BetaMLLearnableParametersSet
    representatives::Union{Nothing,Matrix{Float64}}  = nothing
end

"""
$(TYPEDEF)

The classical "K-Means" clustering algorithm (unsupervised).

Learn to partition the data and assign each record to one of the `n_classes` classes according to a distance metric (default Euclidean).

For the parameters see [`?KMeansMedoidsHyperParametersSet`](@ref KMeansMedoidsHyperParametersSet) and [`?BetaMLDefaultOptionsSet`](@ref BetaMLDefaultOptionsSet).

# Notes:
- data must be numerical
- online fitting (re-fitting with new data) is supported

# Example :

```julia
julia> using BetaML

julia> X = [1.1 10.1; 0.9 9.8; 10.0 1.1; 12.1 0.8; 0.8 9.8]
5×2 Matrix{Float64}:
  1.1  10.1
  0.9   9.8
 10.0   1.1
 12.1   0.8
  0.8   9.8

julia> mod = KMeansClusterer(n_classes=2)
KMeansClusterer - A K-Means Model (unfitted)

julia> classes = fit!(mod,X)
5-element Vector{Int64}:
 1
 1
 2
 2
 1

julia> newclasses = fit!(mod,[11 0.9])
1-element Vector{Int64}:
 2

julia> info(mod)
Dict{String, Any} with 2 entries:
  "fitted_records" => 6
  "xndims"         => 2

julia> parameters(mod)
BetaML.Clustering.KMeansMedoidsLearnableParameters (a BetaMLLearnableParametersSet struct)
- representatives: [1.13366 9.7209; 11.0 0.9]
```

"""
mutable struct KMeansClusterer <: BetaMLUnsupervisedModel
    hpar::KMeansMedoidsHyperParametersSet
    opt::BetaMLDefaultOptionsSet
    par::Union{Nothing,KMeansMedoidsLearnableParameters}
    cres::Union{Nothing,Vector{Int64}}
    fitted::Bool
    info::Dict{String,Any}
end

"""
$(TYPEDEF)

The classical "K-Medoids" clustering algorithm (unsupervised).

Similar to K-Means, learn to partition the data and assign each record to one of the `n_classes` classes according to a distance metric, but the "representatives" (the cetroids) are guaranteed to be one of the training points. The algorithm work with any arbitrary distance measure (default Euclidean).

For the parameters see [`?KMeansMedoidsHyperParametersSet`](@ref KMeansMedoidsHyperParametersSet) and [`?BetaMLDefaultOptionsSet`](@ref BetaMLDefaultOptionsSet).

# Notes:
- data must be numerical
- online fitting (re-fitting with new data) is supported

# Example:

julia> using BetaML

julia> X = [1.1 10.1; 0.9 9.8; 10.0 1.1; 12.1 0.8; 0.8 9.8]
5×2 Matrix{Float64}:
  1.1  10.1
  0.9   9.8
 10.0   1.1
 12.1   0.8
  0.8   9.8

julia> mod = KMedoidsClusterer(n_classes=2)
KMedoidsClusterer - A K-Medoids Model (unfitted)

julia> classes = fit!(mod,X)
5-element Vector{Int64}:
 1
 1
 2
 2
 1

julia> newclasses = fit!(mod,[11 0.9])
1-element Vector{Int64}:
 2

julia> info(mod)
Dict{String, Any} with 2 entries:
  "fitted_records" => 6
  "xndims"         => 2

julia> parameters(mod)
BetaML.Clustering.KMeansMedoidsLearnableParameters (a BetaMLLearnableParametersSet struct)
- representatives: [0.9 9.8; 11.0 0.9]

"""
mutable struct KMedoidsClusterer <: BetaMLUnsupervisedModel
    hpar::KMeansMedoidsHyperParametersSet
    opt::BetaMLDefaultOptionsSet
    par::Union{Nothing,KMeansMedoidsLearnableParameters}
    cres::Union{Nothing,Vector{Int64}}
    fitted::Bool
    info::Dict{String,Any}
end


function KMeansClusterer(;kwargs...)
    m = KMeansClusterer(KMeansMedoidsHyperParametersSet(),BetaMLDefaultOptionsSet(),KMeansMedoidsLearnableParameters(),nothing,false,Dict{Symbol,Any}())
    thisobjfields  = fieldnames(nonmissingtype(typeof(m)))
    for (kw,kwv) in kwargs
       found = false
       for f in thisobjfields
          fobj = getproperty(m,f)
          if kw in fieldnames(typeof(fobj))
              setproperty!(fobj,kw,kwv)
              found = true
          end
        end
        found || error("Keyword \"$kw\" is not part of this model.")
    end
    return m
end

function KMedoidsClusterer(;kwargs...)
    m = KMedoidsClusterer(KMeansMedoidsHyperParametersSet(),BetaMLDefaultOptionsSet(),KMeansMedoidsLearnableParameters(),nothing,false,Dict{Symbol,Any}())
    thisobjfields  = fieldnames(nonmissingtype(typeof(m)))
    for (kw,kwv) in kwargs
       found = false
       for f in thisobjfields
          fobj = getproperty(m,f)
          if kw in fieldnames(typeof(fobj))
              setproperty!(fobj,kw,kwv)
              found = true
          end
        end
        found || error("Keyword \"$kw\" is not part of this model.")
    end
    return m
end

"""
$(TYPEDSIGNATURES)

Fit the [`KMeansClusterer`](@ref) model to data

"""
function fit!(m::KMeansClusterer,x)

    # Parameter alias..
    K                      = m.hpar.n_classes
    dist                   = m.hpar.dist
    initialisation_strategy           = m.hpar.initialisation_strategy
    initial_representatives = m.hpar.initial_representatives
    cache                  = m.opt.cache
    verbosity              = m.opt.verbosity
    rng                    = m.opt.rng

    if m.fitted
        # Note that doing this we give lot of importance to the new data, even if this is few records and the model has bee fitted with milions of records.
        # So, training 1000 records doesn't give the same output as training 990 records and then training again with 10 records
        verbosity >= HIGH  && @info "Continuing training of a pre-fitted model"
        (clIdx,Z) = kmeans(x,K,dist=dist,initial_representatives=m.par.representatives,initialisation_strategy="given",verbosity=verbosity,rng=rng)

    else
        (clIdx,Z) = kmeans(x,K,dist=dist,initialisation_strategy=initialisation_strategy,initial_representatives=initial_representatives,verbosity=verbosity,rng=rng)
    end
    m.par  = KMeansMedoidsLearnableParameters(representatives=Z)
    m.cres = cache ? clIdx : nothing
    m.info["fitted_records"] = get(m.info,"fitted_records",0) + size(x,1)
    m.info["xndims"]     = size(x,2)
    m.fitted=true
    return cache ? m.cres : nothing
end   

"""
$(TYPEDSIGNATURES)

Fit the [`KMedoidsClusterer`](@ref) model to data

"""
function fit!(m::KMedoidsClusterer,x)

    # Parameter alias..
    K                      = m.hpar.n_classes
    dist                   = m.hpar.dist
    initialisation_strategy           = m.hpar.initialisation_strategy
    initial_representatives = m.hpar.initial_representatives
    cache                  = m.opt.cache
    verbosity              = m.opt.verbosity
    rng                    = m.opt.rng

    if m.fitted
        # Note that doing this we give lot of importance to the new data, even if this is few records and the model has bee fitted with milions of records.
        # So, training 1000 records doesn't give the same output as training 990 records and then training again with 10 records
        verbosity >= HIGH  && @info "Continuing training of a pre-fitted model"
        (clIdx,Z) = kmedoids(x,K,dist=dist,initial_representatives=m.par.representatives,initialisation_strategy="given",verbosity=verbosity,rng=rng)

    else
        (clIdx,Z) = kmedoids(x,K,dist=dist,initialisation_strategy=initialisation_strategy,initial_representatives=initial_representatives,verbosity=verbosity,rng=rng)
    end
    m.par  = KMeansMedoidsLearnableParameters(representatives=Z)
    m.cres = cache ? clIdx : nothing
    m.info["fitted_records"] = get(m.info,"fitted_records",0) + size(x,1)
    m.info["xndims"]     = size(x,2)
    m.fitted=true
    return cache ? m.cres : nothing
end  

"""
$(TYPEDSIGNATURES)

Assign the class of new data using the representatives learned by fitting a [`KMeansClusterer`](@ref) or [`KMedoidsClusterer`](@ref) model.

"""
function predict(m::Union{KMeansClusterer,KMedoidsClusterer},X)
    X               = makematrix(X)
    representatives = m.par.representatives
    classes = classAssignation(X,representatives,m.hpar.dist)
    return classes
end

function show(io::IO, ::MIME"text/plain", m::KMeansClusterer)
    if m.fitted == false
        print(io,"KMeansClusterer - A K-Means Model (unfitted)")
    else
        print(io,"KMeansClusterer - A K-Means Model (fitted on $(m.info["fitted_records"]) records)")
    end
end

function show(io::IO, ::MIME"text/plain", m::KMedoidsClusterer)
    if m.fitted == false
        print(io,"KMedoidsClusterer - A K-Medoids Model (unfitted)")
    else
        print(io,"KMedoidsClusterer - A K-Medoids Model (fitted on $(m.info["fitted_records"]) records)")
    end
end

function show(io::IO, m::KMeansClusterer)
    m.opt.descr != "" && println(io,m.opt.descr)
    if m.fitted == false
        print(io,"KMeansClusterer - A $(m.hpar.n_classes)-classes K-Means Model (unfitted)")
    else
        println(io,"KMeansClusterer - A $(m.info["xndims"])-dimensions $(m.hpar.n_classes)-classes K-Means Model (fitted on $(m.info["fitted_records"]) records)")
        println(io,m.info)
        println(io,"Representatives:")
        println(io,m.par.representatives)
    end
end


function show(io::IO, m::KMedoidsClusterer)
    m.opt.descr != "" && println(io,m.opt.descr)
    if m.fitted == false
        print(io,"KMedoidsClusterer - A $(m.hpar.n_classes)-classes K-Medoids Model (unfitted)")
    else
        println(io,"KMedoidsClusterer - A $(m.info["xndims"])-dimensions $(m.hpar.n_classes)-classes K-Medoids Model (fitted on $(m.info["fitted_records"]) records)")
        println(io,m.info)
        println(io,"Distance function used:")
        println(io,m.hpar.dist)
        println(io,"Representatives:")
        println(io,m.par.representatives)
    end
end
