"Part of [BetaML](https://github.com/sylvaticus/BetaML.jl). Licence is MIT."

# Part of submodule Utils of BetaML - The Beta Machine Learning Toolkit
# Functions typically used for processing (manipulating) data, typically preprocessing data before running a ML model


# ------------------------------------------------------------------------------
# Various reshaping functions
import Base.reshape
""" reshape(myNumber, dims..) - Reshape a number as a n dimensional Array """
reshape(x::T, dims...) where {T <: Number} =   (x = [x]; reshape(x,dims) )
makecolvector(x::T) where {T} =  [x]
makecolvector(x::T) where {T <: AbstractArray} =  reshape(x,length(x))
makerowvector(x::T) where {T <: Number} = return [x]'
makerowvector(x::T) where {T <: AbstractArray} =  reshape(x,1,length(x))
"""Transform an Array{T,1} in an Array{T,2} and leave unchanged Array{T,2}."""
makematrix(x::AbstractArray) = ndims(x) == 1 ? reshape(x, (size(x)...,1)) : x


"""Return wheather an array is sortable, i.e. has methos issort defined"""
issortable(::AbstractArray{T,N})  where {T,N} = hasmethod(isless, Tuple{nonmissingtype(T),nonmissingtype(T)})


allowmissing!(x::AbstractArray{T,N}) where {T,N}    = convert(Union{Array{T,N},Missing},x)
disallowmissing!(x::AbstractArray{T,N}) where {T,N} = convert(Array{nonmissingtype(T),N},x)

"""
    getpermutations(v::AbstractArray{T,1};keepStructure=false)

Return a vector of either (a) all possible permutations (uncollected) or (b) just those based on the unique values of the vector

Useful to measure accuracy where you don't care about the actual name of the labels, like in unsupervised classifications (e.g. clustering)

"""
function getpermutations(v::AbstractArray{T,1};keepStructure=false) where {T}
    if !keepStructure
        return Combinatorics.permutations(v)
    else
        classes       = unique(v)
        nCl           = length(classes)
        N             = size(v,1)
        pSet          = Combinatorics.permutations(1:nCl)
        nP            = length(pSet)
        vPermutations = fill(similar(v),nP)
        vOrigIdx      = [findfirst(x -> x == v[i] , classes) for i in 1:N]
        for (pIdx,perm) in enumerate(pSet)
            vPermutations[pIdx] = classes[perm[vOrigIdx]] # permuted specific version
        end
        return vPermutations
    end
end


""" singleunique(x) Return the unique values of x whether x is an array of arrays, an array or a scalar"""
function singleunique(x::Union{T,AbstractArray{T}}) where {T <: Union{Any,AbstractArray{T2}} where T2 <: Any }
    if typeof(x) <: AbstractArray{T2} where {T2 <: AbstractArray}
        return unique(vcat(unique.(x)...))
    elseif typeof(x) <: AbstractArray{T2} where {T2}
        return unique(x)
    else
        return [x]
    end
end

findfirst(el::T,cont::Array{T};returnTuple=true) where {T<:Union{AbstractString,Number}} = ndims(cont) > 1 && returnTuple ? Tuple(findfirst(x -> isequal(x,el),cont)) : findfirst(x -> isequal(x,el),cont)
#findfirst(el::T,cont::Array{T,N};returnTuple=true) where {T,N} = returnTuple ? Tuple(findfirst(x -> isequal(x,el),cont)) : findfirst(x -> isequal(x,el),cont)
#findfirst(el::T,cont::Array{T,1};returnTuple=true) where {T} =  findfirst(x -> isequal(x,el),cont)


findall(el::T, cont::Array{T};returnTuple=true) where {T} = ndims(cont) > 1 && returnTuple ? Tuple.(findall(x -> isequal(x,el),cont)) : findall(x -> isequal(x,el),cont)


# API V2 for encoders


"""
$(TYPEDEF)

Hyperparameters for both [`OneHotEncoder`](@ref) and [`OrdinalEncoder`](@ref)

# Parameters:
$(FIELDS)

"""
Base.@kwdef mutable struct OneHotEncoderHyperParametersSet <: BetaMLHyperParametersSet
  "The categories to represent as columns. [def: `nothing`, i.e. unique training values or range for integers]. Do not include `missing` in this list."  
  categories::Union{Vector,Nothing} = nothing
  "How to handle categories not seen in training or not present in the provided `categories` array? \"error\" (default) rises an error, \"missing\" labels the whole output with missing values, \"infrequent\" adds a specific column for these categories in one-hot encoding or a single new category for ordinal one."
  handle_unknown::String = "error"
  "Which value during inverse transformation to assign to the \"other\" category (i.e. categories not seen on training or not present in the provided `categories` array? [def: ` nothing`, i.e. typemax(Int64) for integer vectors and \"other\" for other types]. This setting is active only if `handle_unknown=\"infrequent\"` and in that case it MUST be specified if the vector to one-hot encode is neither integer or strings"
  other_categories_name = nothing

end
Base.@kwdef mutable struct OneHotEncoderLearnableParameters <: BetaMLLearnableParametersSet
  categories_applied::Vector = []
  original_vector_eltype::Union{Type,Nothing} = nothing 
end

"""
$(TYPEDEF)

Encode a vector of categorical values as one-hot columns.

The algorithm distinguishes between _missing_ values, for which it returns a one-hot encoded row of missing values, and _other_ categories not in the provided list or not seen during training that are handled according to the `handle_unknown` parameter. 

For the parameters see [`OneHotEncoderHyperParametersSet`](@ref) and [`BetaMLDefaultOptionsSet`](@ref).  This model supports `inverse_predict`.

"""
mutable struct OneHotEncoder <: BetaMLUnsupervisedModel
    hpar::OneHotEncoderHyperParametersSet
    opt::BetaMLDefaultOptionsSet
    par::Union{Nothing,OneHotEncoderLearnableParameters}
    cres::Union{Nothing,Matrix{Bool},Matrix{Union{Bool,Missing}}}
    fitted::Bool
    info::Dict{String,Any}
end

"""
$(TYPEDEF)

Encode a vector of categorical values as integers.

The algorithm distinguishes between _missing_ values, for which it propagate the missing, and _other_ categories not in the provided list or not seen during training that are handled according to the `handle_unknown` parameter. 

For the parameters see [`OneHotEncoderHyperParametersSet`](@ref) and [`BetaMLDefaultOptionsSet`](@ref). This model supports `inverse_predict`.

"""
mutable struct OrdinalEncoder <: BetaMLUnsupervisedModel
    hpar::OneHotEncoderHyperParametersSet
    opt::BetaMLDefaultOptionsSet
    par::Union{Nothing,OneHotEncoderLearnableParameters}
    cres::Union{Nothing,Vector{Int64},Vector{Union{Int64,Missing}}}
    fitted::Bool
    info::Dict{String,Any}
end

function OneHotEncoder(;kwargs...)
    m = OneHotEncoder(OneHotEncoderHyperParametersSet(),BetaMLDefaultOptionsSet(),OneHotEncoderLearnableParameters(),nothing,false,Dict{Symbol,Any}())
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

function OrdinalEncoder(;kwargs...)
    m = OrdinalEncoder(OneHotEncoderHyperParametersSet(),BetaMLDefaultOptionsSet(),OneHotEncoderLearnableParameters(),nothing,false,Dict{Symbol,Any}())
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

function _fit!(m::Union{OneHotEncoder,OrdinalEncoder},x,enctype::Symbol)
    x     = makecolvector(x)
    N     = size(x,1)
    vtype = eltype(x) #    nonmissingtype(eltype(x))

    # Parameter aliases
    categories             = m.hpar.categories
    handle_unknown         = m.hpar.handle_unknown
    other_categories_name  = m.hpar.other_categories_name
    if isnothing(other_categories_name)
        if nonmissingtype(vtype) <: Integer
            other_categories_name = typemax(Int64)
        else
            other_categories_name = "other"
        end
    end
    cache                  = m.opt.cache
    verbosity              = m.opt.verbosity
    rng                    = m.opt.rng

    if nonmissingtype(vtype) <: Number && !(nonmissingtype(vtype) <: Integer)
        # continuous column: we just apply identity
        m.par = OneHotEncoderLearnableParameters([],vtype)
        return cache ? nothing : x
    end
   
    if isnothing(categories)
        if nonmissingtype(vtype) <: Integer
            minx = minimum(x) 
            maxx = maximum(x)
            categories_applied = collect(minx:maxx)
        else
            categories_applied = collect(skipmissing(unique(x)))
        end
    else
        categories_applied = deepcopy(categories)
    end

    handle_unknown == "infrequent" && push!(categories_applied,other_categories_name)
    m.par = OneHotEncoderLearnableParameters(categories_applied,vtype)

    if cache
        if enctype == :onehot
            K    = length(categories_applied)
            outx = fill(false,N,K)
        else 
            K    = 1
            outx = zeros(Int64,N,K)
        end
        for n in 1:N
            if ismissing(x[n]) 
                outx = (enctype == :onehot) ? convert(Matrix{Union{Missing,Bool}},outx) : convert(Matrix{Union{Missing,Int64}},outx)
                outx[n,:] = fill(missing,K)
                continue
            end
            kidx = findfirst(y -> isequal(y,x[n]),categories_applied)
            if isnothing(kidx)
                if handle_unknown == "error"
                    error("Found a category ($(x[n])) not present in the list and the `handle_unknown` is set to `error`. Perhaps you want to swith it to either `missing` or `infrequent`.")
                elseif handle_unknown == "missing"
                    outx = (enctype == :onehot) ? convert(Matrix{Union{Missing,Bool}},outx) : convert(Matrix{Union{Missing,Int64}},outx)
                    outx[n,:] = fill(missing,K);
                    continue
                elseif handle_unknown == "infrequent"
                    outx[n,K] = (enctype == :onehot) ? true : length(categories_applied)
                    continue
                else
                    error("I don't know how to process `handle_unknown == $(handle_unknown)`")
                end
            end
            enctype == :onehot ? (outx[n,kidx] = true) : outx[n,1] = kidx
        end
        m.cres = (enctype == :onehot) ? outx : collect(dropdims(outx,dims=2))
    end

    m.info["fitted_records"] = get(m.info,"fitted_records",0) + size(x,1)
    m.info["n_categories"]   = length(categories_applied)
    m.fitted = true
    return cache ? m.cres : nothing
end
fit!(m::OneHotEncoder,x)  = _fit!(m,x,:onehot)
fit!(m::OrdinalEncoder,x) = _fit!(m,x,:ordinal)

function _predict(m::Union{OneHotEncoder,OrdinalEncoder},x,enctype::Symbol)
    x     = makecolvector(x)
    N     = size(x,1)
    vtype = eltype(x) #    nonmissingtype(eltype(x))

    # Parameter aliases
    handle_unknown         = m.hpar.handle_unknown
    categories_applied     = m.par.categories_applied

    if enctype == :onehot
        K    = length(categories_applied)
        outx = fill(false,N,K)
    else 
        K    = 1
        outx = zeros(Int64,N,K)
    end
    for n in 1:N
        if ismissing(x[n]) 
            outx = (enctype == :onehot) ? convert(Matrix{Union{Missing,Bool}},outx) : convert(Matrix{Union{Missing,Int64}},outx)
            outx[n,:] = fill(missing,K)
            continue
        end
        kidx = findfirst(y -> isequal(y,x[n]),categories_applied)
        if isnothing(kidx)
            if handle_unknown == "error"
                error("Found a category ($(x[n])) not present in the list and the `handle_unknown` is set to `error`. Perhaps you want to swith it to either `missing` or `infrequent`.")
                continue
            elseif handle_unknown == "missing"
                outx = (enctype == :onehot) ? convert(Matrix{Union{Missing,Bool}},outx) : convert(Matrix{Union{Missing,Int64}},outx)
                outx[n,:] = fill(missing,K);
                continue
            elseif handle_unknown == "infrequent"
                outx[n,K] = (enctype == :onehot) ? true : length(categories_applied)
                continue
            else
                error("I don't know how to process `handle_unknown == $(handle_unknown)`")
            end
        else
            enctype == :onehot ? (outx[n,kidx] = true) : outx[n,1] = kidx
        end
    end
    return (enctype == :onehot) ? outx : dropdims(outx,dims=2)
end

# Case where X is a vector of dictionaries
function _predict(m::Union{OneHotEncoder,OrdinalEncoder},x::Vector{<:Dict},enctype::Symbol)

    N     = size(x,1)
    # Parameter aliases
    handle_unknown         = m.hpar.handle_unknown
    categories_applied     = m.par.categories_applied

    if enctype == :onehot
        K    = length(categories_applied)
        outx = fill(0.0,N,K)
    else 
        error("Predictions of a Ordinal Encoded with a vector of dictionary is not supported")
    end
    for n in 1:N
        for (k,v) in x[n]
            kidx = findfirst(y -> isequal(y,k),categories_applied)
            if isnothing(kidx)
                if handle_unknown == "error"
                    error("Found a category ($(k)) not present in the list and the `handle_unknown` is set to `error`. Perhaps you want to swith it to either `missing` or `infrequent`.")
                    continue
                elseif handle_unknown == "missing"
                    outx[n,:] = fill(missing,K);
                    continue
                elseif handle_unknown == "infrequent"
                    outx[n,K] = v
                    continue
                else
                    error("I don't know how to process `handle_unknown == $(handle_unknown)`")
                end
            else
                outx[n,kidx] = v
            end
        end
    end
    return outx
end




predict(m::OneHotEncoder,x)  = _predict(m,x,:onehot)
predict(m::OrdinalEncoder,x) = _predict(m,x,:ordinal)
function _inverse_predict(m,x,enctype::Symbol)
    # Parameter aliases
    handle_unknown         = m.hpar.handle_unknown
    categories_applied     = m.par.categories_applied
    original_vector_eltype = m.par.original_vector_eltype
    other_categories_name  = m.hpar.other_categories_name
    if isnothing(other_categories_name)
        if nonmissingtype(original_vector_eltype ) <: Integer
            other_categories_name = typemax(Int64)
        else
            other_categories_name = "other"
        end
    end

    N,D     = size(x,1),size(x,2)
    outx    = Array{original_vector_eltype,1}(undef,N)

    for n in 1:N
        if enctype == :onehot
            if any(ismissing.(x[n,:]))
                outx[n] = missing
                continue
            elseif handle_unknown == "infrequent" && findfirst(c->c==true,x[n,:]) == D
                outx[n] = other_categories_name
                continue
            end
            outx[n] = categories_applied[findfirst(c->c==true,x[n,:])]
        else
            if ismissing(x[n])
                outx[n] = missing
                continue
            elseif handle_unknown == "infrequent" && x[n] == length(categories_applied)
                outx[n] = other_categories_name
                continue
            end
            outx[n] = categories_applied[x[n]]
        end
    end
    return outx
end
inverse_predict(m::OneHotEncoder,x::AbstractMatrix{<:Union{Int64,Bool,Missing}})  = _inverse_predict(m,x,:onehot)
function inverse_predict(m::OneHotEncoder,x::AbstractMatrix{<:Float64})  
    x2 = fit!(OneHotEncoder(categories=1:size(x,2)),mode(x))
    return inverse_predict(m,x2)
end

inverse_predict(m::OrdinalEncoder,x) = _inverse_predict(m,x,:ordinal)

"""
    partition(data,parts;shuffle,dims,rng)

Partition (by rows) one or more matrices according to the shares in `parts`.

# Parameters
* `data`: A matrix/vector or a vector of matrices/vectors
* `parts`: A vector of the required shares (must sum to 1)
* `shufle`: Whether to randomly shuffle the matrices (preserving the relative order between matrices)
* `dims`: The dimension for which to partition [def: `1`]
* `copy`: Wheter to _copy_ the actual data or only create a reference [def: `true`]
* `rng`: Random Number Generator (see [`FIXEDSEED`](@ref)) [deafult: `Random.GLOBAL_RNG`]

# Notes:
* The sum of parts must be equal to 1
* The number of elements in the specified dimension must be the same for all the arrays in `data`

# Example:
```julia
julia> x = [1:10 11:20]
julia> y = collect(31:40)
julia> ((xtrain,xtest),(ytrain,ytest)) = partition([x,y],[0.7,0.3])
 ```
 """
function partition(data::AbstractArray{T,1},parts::AbstractArray{Float64,1};shuffle=true,dims=1,copy=true,rng = Random.GLOBAL_RNG) where T <: AbstractArray
        # the sets of vector/matrices
        N = size(data[1],dims)
        all(size.(data,dims) .== N) || @error "All matrices passed to `partition` must have the same number of elements for the required dimension"
        ridx = shuffle ? Random.shuffle(rng,1:N) : collect(1:N)
        return partition.(data,Ref(parts);shuffle=shuffle,dims=dims,fixed_ridx = ridx,copy=copy,rng=rng)
end

function partition(data::AbstractArray{T,Ndims}, parts::AbstractArray{Float64,1};shuffle=true,dims=1,fixed_ridx=Int64[],copy=true,rng = Random.GLOBAL_RNG) where {T,Ndims}
    # the individual vector/matrix
    N        = size(data,dims)
    nParts   = size(parts)
    toReturn = toReturn = Array{AbstractArray{T,Ndims},1}(undef,nParts)
    if !(sum(parts) ≈ 1)
        @error "The sum of `parts` in `partition` should total to 1."
    end
    ridx = fixed_ridx
    if (isempty(ridx))
       ridx = shuffle ? Random.shuffle(rng, 1:N) : collect(1:N)
    end
    allDimIdx = convert(Vector{Union{UnitRange{Int64},Vector{Int64}}},[1:i for i in size(data)])
    current = 1
    cumPart = 0.0
    for (i,p) in enumerate(parts)
        cumPart += parts[i]
        final = i == nParts ? N : Int64(round(cumPart*N))
        allDimIdx[dims] = ridx[current:final]
        toReturn[i]     = copy ? data[allDimIdx...] : @views data[allDimIdx...]
        current         = (final +=1)
    end
    return toReturn
end



# API V2 for Scale

abstract type AbstractScaler end
abstract type AbstractScalerLearnableParameter end 

"""
$(TYPEDEF)

Scale the data to a given (def: unit) hypercube

# Parameters
$(FIELDS)

"""
Base.@kwdef mutable struct MinMaxScaler <: AbstractScaler
  "The range of the input. [def: (minimum,maximum)]. Both ranges are functions of the data. You can consider other relative of absolute ranges using e.g. `inputRange=(x->minimum(x)*0.8,x->100)`"  
  inputRange::Tuple{Function,Function} = (minimum,maximum)
  "The range of the scaled output [def: (0,1)]"
  outputRange::Tuple{Real,Real} = (0,1)
end
Base.@kwdef mutable struct MinMaxScalerLearnableParameters <: AbstractScalerLearnableParameter
  inputRangeApplied::Vector{Tuple{Float64,Float64}} = [(-Inf,+Inf)]
end

"""
$(TYPEDEF)

Standardise the input to zero mean and unit standard deviation, aka "Z-score". 
Note that missing values are skipped.

# Parameters
$(FIELDS)

"""
Base.@kwdef mutable struct StandardScaler <: AbstractScaler
    "Scale to unit variance [def: true]"
    scale::Bool=true
    "Center to zero mean [def: true]"
    center::Bool=true
end

Base.@kwdef mutable struct StandardScalerLearnableParameters <: AbstractScalerLearnableParameter
  sfμ::Vector{Float64} = Float64[] # scale factor of mean
  sfσ::Vector{Float64} = Float64[]  # scale vactor of st.dev.
end

function _fit(m::MinMaxScaler,skip,X,cache)
    actualRanges = Tuple{Float64,Float64}[]
    X_scaled = cache ? deepcopy(X) : nothing 
    for (ic,c) in enumerate(eachcol(X))
        if !(ic in skip)
          imin,imax =   (m.inputRange[1](skipmissing(c)),  m.inputRange[2](skipmissing(c)) )
          if cache
            omin, omax = m.outputRange[1], m.outputRange[2]
            X_scaled[:,ic] = (c .- imin) .* ((omax-omin)/(imax-imin)) .+ omin
          end
          push!(actualRanges,(imin,imax))
        else
          push!(actualRanges,(-Inf,+Inf))
        end
    end
    return X_scaled, MinMaxScalerLearnableParameters(actualRanges)
end
function _fit(m::StandardScaler,skip,X,cache)
    nR,nD = size(X)
    sfμ   = zeros(nD)
    sfσ   = ones(nD)
    X_scaled = cache ? deepcopy(X) : nothing 
    for (ic,c) in enumerate(eachcol(X))
        if !(ic in skip)
            μ  = m.center ? mean(skipmissing(c)) : 0.0
            σ² = m.scale  ? var(skipmissing(c),corrected=false) : 1.0 
            sfμ[ic] = - μ
            sfσ[ic] = 1 ./ sqrt.(σ²)
            if cache
                X_scaled[:,ic] = (c .+ sfμ[ic]) .* sfσ[ic] 
            end
        end
    end
 
    return X_scaled, StandardScalerLearnableParameters(sfμ,sfσ)
end

function _predict(m::MinMaxScaler,pars::MinMaxScalerLearnableParameters,skip,X;inverse=false)
    if !inverse
        xnew = deepcopy(X)
        for (ic,c) in enumerate(eachcol(X))
            if !(ic in skip)
                imin,imax = pars.inputRangeApplied[ic]
                omin,omax = m.outputRange
                xnew[:,ic] = (c .- imin) .* ((omax-omin)/(imax-imin)) .+ omin
            end
        end
        return xnew
    else
        xorig = deepcopy(X)
        for (ic,c) in enumerate(eachcol(X))
            if !(ic in skip)
                imin,imax = pars.inputRangeApplied[ic]
                omin,omax = m.outputRange
                xorig[:,ic] = (c .- omin) .* ((imax-imin)/(omax-omin)) .+ imin
            end
        end
        return xorig
    end
end
function _predict(m::StandardScaler,pars::StandardScalerLearnableParameters,skip,X;inverse=false)
    if !inverse
        xnew = deepcopy(X)
        for (ic,c) in enumerate(eachcol(X))
            if !(ic in skip)
                xnew[:,ic] = (c .+ pars.sfμ[ic]) .* pars.sfσ[ic] 
            end
        end
        return xnew
    else
        xorig = deepcopy(X)
        for (ic,c) in enumerate(eachcol(X))
            if !(ic in skip)
                xorig[:,ic] = (c ./ pars.sfσ[ic] .- pars.sfμ[ic])  
            end
        end
        return xorig
    end
end

"""
$(TYPEDEF)

Hyperparameters for the Scaler transformer

## Parameters
$(FIELDS)
"""
Base.@kwdef mutable struct ScalerHyperParametersSet <: BetaMLHyperParametersSet
    "The specific scaler method to employ with its own parameters. See [`StandardScaler`](@ref) [def] or [`MinMaxScaler`](@ref)."
    method::AbstractScaler = StandardScaler()
    "The positional ids of the columns to skip scaling (eg. categorical columns, dummies,...) [def: `[]`]"
    skip::Vector{Int64}    = Int64[]
end

Base.@kwdef mutable struct ScalerLearnableParameters <: BetaMLLearnableParametersSet
   scalerpars::AbstractScalerLearnableParameter = StandardScalerLearnableParameters()
end

"""
$(TYPEDEF)

Scale the data according to the specific chosen method (def: `StandardScaler`) 

For the parameters see [`ScalerHyperParametersSet`](@ref) and [`BetaMLDefaultOptionsSet`](@ref) 

```
julia>m = Scaler(MinMaxScaler(inputRange=(x->minimum(x)*0.8,maximum),outputRange=(0,256)),skip=[3,7,8])
```

"""
mutable struct Scaler <: BetaMLUnsupervisedModel
    hpar::ScalerHyperParametersSet
    opt::BetaMLDefaultOptionsSet
    par::Union{Nothing,ScalerLearnableParameters}
    cres::Union{Nothing,Matrix}
    fitted::Bool
    info::Dict{String,Any}
end

function Scaler(;kwargs...)
    m = Scaler(ScalerHyperParametersSet(),BetaMLDefaultOptionsSet(),ScalerLearnableParameters(),nothing,false,Dict{Symbol,Any}())
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

function Scaler(method;kwargs...)
    m = Scaler(ScalerHyperParametersSet(method=method),BetaMLDefaultOptionsSet(),ScalerLearnableParameters(),nothing,false,Dict{Symbol,Any}())
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

function fit!(m::Scaler,x)

    # Parameter alias..
    scaler                 = m.hpar.method
    skip                   = m.hpar.skip
    cache                  = m.opt.cache
    verbosity              = m.opt.verbosity
    rng                    = m.opt.rng

    if m.fitted
        verbosity >= STD && @warn "This model doesn't support online training. Training will be performed based on new data only."
    end

    m.cres,m.par.scalerpars = _fit(scaler,skip,x,cache)
    m.info["fitted_records"] = get(m.info,"fitted_records",0) + size(x,1)
    m.info["xndims"]     = size(x,2)
    m.fitted = true
    return cache ? m.cres : nothing 
end   

function predict(m::Scaler,x)
    return _predict(m.hpar.method,m.par.scalerpars,m.hpar.skip,x;inverse=false)
end  

function inverse_predict(m::Scaler,x)
    return _predict(m.hpar.method,m.par.scalerpars,m.hpar.skip,x;inverse=true)
end  

"""
$(TYPEDEF)

Hyperparameters for the PCA transformer

## Parameters
$(FIELDS)

"""
Base.@kwdef mutable struct PCAHyperParametersSet <: BetaMLHyperParametersSet
   "The number of dimensions to maintain (with `outdims <= size(X,2)` ) [def: `nothing`, i.e. the number of output dimensions is determined from the parameter `max_unexplained_var`]"
   outdims::Union{Nothing,Int64} = nothing
   "The maximum proportion of variance that we are willing to accept when reducing the number of dimensions in our data [def: 0.05]. It doesn't have any effect when the output number of dimensions is explicitly chosen with the parameter `outdims`"
   max_unexplained_var::Float64  = 0.05
end

Base.@kwdef mutable struct PCALearnableParameters <: BetaMLLearnableParametersSet
   eigen_out::Union{Eigen,Nothing}     =nothing
   outdims_actual::Union{Int64,Nothing}=nothing
end

"""
$(TYPEDEF)

Perform a Principal Component Analysis, a dimensionality reduction tecnique employing a linear trasformation of the original matrix by the eigenvectors of the covariance matrix.

PCA returns the matrix reprojected among the dimensions of maximum variance.

For the parameters see [`PCAHyperParametersSet`](@ref) and [`BetaMLDefaultOptionsSet`](@ref) 

## Example :
```julia
julia> X = [1 10 100; 1.1 15 120; 0.95 23 90; 0.99 17 120; 1.05 8 90; 1.1 12 95]
6×3 Matrix{Float64}:
 1.0   10.0  100.0
 1.1   15.0  120.0
 0.95  23.0   90.0
 0.99  17.0  120.0
 1.05   8.0   90.0
 1.1   12.0   95.0

julia> mod = PCA(max_unexplained_var=0.05) # the default
A PCA BetaMLModel (unfitted)

julia> reproj_X = fit!(mod,X)
6×2 Matrix{Float64}:
 100.449    3.1783
 120.743    6.80764
  91.3551  16.8275
 120.878    8.80372
  90.3363   1.86179
  95.5965   5.51254

julia> info(mod)
Dict{String, Any} with 5 entries:
  "explained_var_by_dim" => [0.873992, 0.999989, 1.0]
  "fitted_records"       => 6
  "prop_explained_var"   => 0.999989
  "retained_dims"        => 2
  "xndims"               => 3
```

## Notes:
- PCA doesn't automatically scale the data. It is suggested to apply the [`Scaler`](@ref) model before running it. 
- Missing data are not supported. Impute them first, see the [`Imputation`](Imputation.html) module.
- If one doesn't know _a priori_ the maximum unexplained variance that he is willling to accept, nor the wished number of dimensions, he can run the model with all the dimensions in output (i.e. with `outdims=size(X,2)`), analise the proportions of explained cumulative variance by dimensions in `info(mod,""explained_var_by_dim")`, choose the number of dimensions K according to his needs and finally pick from the reprojected matrix only the number of dimensions required, i.e. `out.X[:,1:K]`.
"""
mutable struct PCA <: BetaMLUnsupervisedModel
    hpar::PCAHyperParametersSet
    opt::BetaMLDefaultOptionsSet
    par::Union{Nothing,PCALearnableParameters}
    cres::Union{Nothing,Matrix}
    fitted::Bool
    info::Dict{String,Any}
end

function PCA(;kwargs...)
    m = PCA(PCAHyperParametersSet(),BetaMLDefaultOptionsSet(),PCALearnableParameters(),nothing,false,Dict{Symbol,Any}())
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



function fit!(m::PCA,X)

    # Parameter alias..
    outdims                  = m.hpar.outdims
    max_unexplained_var = m.hpar.max_unexplained_var
    cache                    = m.opt.cache
    verbosity                = m.opt.verbosity
    rng                      = m.opt.rng

    if m.fitted
        verbosity >= STD && @warn "This model doesn't support online training. Training will be performed based on new data only."
    end

    (N,D) = size(X)
    if !isnothing(outdims) && outdims > D
        @error("The parameter `outdims` must be ≤ of the number of dimensions of the input data matrix")
    end
    Σ = (1/N) * X'*(I-(1/N)*ones(N)*ones(N)')*X
    E = eigen(Σ) # eigenvalues are ordered from the smallest to the largest
    # Finding oudims_actual
    totvar  = sum(E.values)
    explained_var_by_dim = cumsum(reverse(E.values)) ./ totvar
    outdims_actual = isnothing(outdims) ? findfirst(x -> x >= (1-max_unexplained_var), explained_var_by_dim)  : outdims
    m.par.eigen_out = E
    m.par.outdims_actual = outdims_actual

    if cache
        P = E.vectors[:,end:-1:D-outdims_actual+1]
        m.cres = X*P
    end

    m.info["fitted_records"]  = get(m.info,"fitted_records",0) + N
    m.info["xndims"]     = D
    m.info["explained_var_by_dim"] = explained_var_by_dim
    m.info["prop_explained_var"]   = explained_var_by_dim[outdims_actual]
    m.info["retained_dims"]        = outdims_actual
    m.fitted=true
    return cache ? m.cres : nothing
end   

function predict(m::PCA,X)
    D = size(m.par.eigen_out.vectors,2)
    P = m.par.eigen_out.vectors[:,end:-1:D-m.par.outdims_actual+1]
    return X*P
end  




"""
    cols_with_missing(x)

Retuyrn an array with the ids of the columns where there is at least a missing value.
"""
function cols_with_missing(x)
    cols_with_missing = Int64[]
    (N,D) = size(x)
    for d in 1:D
        for n in 1:N
            if ismissing(x[n,d])
                push!(cols_with_missing,d)
                break
            end
        end
    end
    return cols_with_missing
end

"""
$(TYPEDSIGNATURES)

Perform cross_validation according to `sampler` rule by calling the function f and collecting its output

# Parameters
- `f`: The user-defined function that consume the specific train and validation data and return somehting (often the associated validation error). See later
- `data`: A single n-dimenasional array or a vector of them (e.g. X,Y), depending on the tasks required by `f`.
- sampler: An istance of a ` AbstractDataSampler`, defining the "rules" for sampling at each iteration. [def: `KFold(nsplits=5,nrepeats=1,shuffle=true,rng=Random.GLOBAL_RNG)` ]. Note that the RNG passed to the `f` function is the `RNG` passed to the sampler
- `dims`: The dimension over performing the cross_validation i.e. the dimension containing the observations [def: `1`]
- `verbosity`: The verbosity to print information during each iteration (this can also be printed in the `f` function) [def: `STD`]
- `return_statistics`: Wheter cross_validation should return the statistics of the output of `f` (mean and standard deviation) or the whole outputs [def: `true`].

# Notes

cross_validation works by calling the function `f`, defined by the user, passing to it the tuple `trainData`, `valData` and `rng` and collecting the result of the function f. The specific method for which `trainData`, and `valData` are selected at each iteration depends on the specific `sampler`, whith a single 5 k-fold rule being the default.

This approach is very flexible because the specific model to employ or the metric to use is left within the user-provided function. The only thing that cross_validation does is provide the model defined in the function `f` with the opportune data (and the random number generator).

**Input of the user-provided function**
`trainData` and `valData` are both themselves tuples. In supervised models, cross_validations `data` should be a tuple of (X,Y) and `trainData` and `valData` will be equivalent to (xtrain, ytrain) and (xval, yval). In unsupervised models `data` is a single array, but the training and validation data should still need to be accessed as  `trainData[1]` and `valData[1]`.
**Output of the user-provided function**
The user-defined function can return whatever. However, if `return_statistics` is left on its default `true` value the user-defined function must return a single scalar (e.g. some error measure) so that the mean and the standard deviation are returned.

Note that `cross_validation` can beconveniently be employed using the `do` syntax, as Julia automatically rewrite `cross_validation(data,...) trainData,valData,rng  ...user defined body... end` as `cross_validation(f(trainData,valData,rng ), data,...)`

# Example

```
julia> X = [11:19 21:29 31:39 41:49 51:59 61:69];
julia> Y = [1:9;];
julia> sampler = KFold(nsplits=3);
julia> (μ,σ) = cross_validation([X,Y],sampler) do trainData,valData,rng
                 (xtrain,ytrain) = trainData; (xval,yval) = valData
                 trainedModel    = buildForest(xtrain,ytrain,30)
                 ŷval            = predict(trainedModel,xval)
                 ϵ               = relative_mean_error(yval,ŷval,normrec=false)
                 return ϵ
               end
(0.3202242202242202, 0.04307662219315022)
```

"""
function cross_validation(f,data,sampler=KFold(nsplits=5,nrepeats=1,shuffle=true,rng=Random.GLOBAL_RNG);dims=1,verbosity=STD, return_statistics=true)
    iterResults = []
    for (i,iterData) in enumerate(SamplerWithData(sampler,data,dims))
       iterResult = f(iterData[1],iterData[2],sampler.rng)
       push!(iterResults,iterResult)
       verbosity > HIGH && println("Done iteration $i. This iteration output: $iterResult")
    end
    if return_statistics  return (mean(iterResults),std(iterResults)) else return iterResults end
end

   

"""
$(TYPEDEF)

Simple grid method for hyper-parameters validation of supervised models.

All parameters are tested using cross-validation and then the "best" combination is used. 

# Notes:
- the default loss is suitable for 1-dimensional output supervised models

## Parameters:
$(TYPEDFIELDS)
"""
Base.@kwdef mutable struct GridSearch <: AutoTuneMethod
    "Loss function to use. [def: [`l2loss_by_cv`](@ref)`]. Any function that takes a model, data (a vector of arrays, even if we work only with X) and (using the `rng` keyword) a RNG and return a scalar loss."
    loss::Function = l2loss_by_cv
    "Share of the (data) resources to use for the autotuning [def: 0.1]. With `res_share=1` all the dataset is used for autotuning, it can be very time consuming!"
    res_share::Float64 = 0.1
    "Dictionary of parameter names (String) and associated vector of values to test. Note that you can easily sample these values from a distribution with rand(distr_object,n_values). The number of points you provide for a given parameter can be interpreted as proportional to the prior you have on the importance of that parameter for the algorithm quality."
    hpranges::Dict{String,Any} = Dict{String,Any}()
    "Use multithreads in the search for the best hyperparameters [def: `false`]"
    multithreads::Bool = false
end



"""
$(TYPEDEF)

Hyper-parameters validation of supervised models that search the parameters space trouth successive halving

All parameters are tested on a small sub-sample, then the "best" combinations are kept for a second round that use more samples and so on untill only one hyperparameter combination is left.

# Notes:
- the default loss is suitable for 1-dimensional output supervised models, and applies itself cross-validation. Any function that accepts a model, some data and return a scalar loss can be used
- the rate at which the potential candidate combinations of hyperparameters shrink is controlled by the number of data shares defined in `res_shared` (i.e. the epochs): more epochs are choosen, lower the "shrink" coefficient

## Parameters:
$(TYPEDFIELDS)
"""
Base.@kwdef mutable struct SuccessiveHalvingSearch <: AutoTuneMethod
    "Loss function to use. [def: [`l2loss_by_cv`](@ref)`]. Any function that takes a model, data (a vector of arrays, even if we work only with X) and (using the `rng` keyword) a RNG and return a scalar loss."
    loss::Function = l2loss_by_cv
    """Shares of the (data) resources to use for the autotuning in the successive iterations [def: `[0.05, 0.2, 0.3]`]. With `res_share=1` all the dataset is used for autotuning, it can be very time consuming!
    The number of models is reduced of the same share in order to arrive with a single model. Increase the number of `res_shares` in order to increase the number of models kept at each iteration.
    """
    res_shares::Vector{Float64} = [0.08, 0.1, 0.13, 0.15, 0.2, 0.3, 0.4]
    "Dictionary of parameter names (String) and associated vector of values to test. Note that you can easily sample these values from a distribution with rand(distr_object,n_values). The number of points you provide for a given parameter can be interpreted as proportional to the prior you have on the importance of that parameter for the algorithm quality."
    hpranges::Dict{String,Any} = Dict{String,Any}()
    "Use multiple threads in the search for the best hyperparameters [def: `false`]"
    multithreads::Bool = false
end

"Transform a Dict(parameters => possible range) in a vector of Dict(parameters=>parvalues)"
function _hpranges_2_candidates(hpranges)
    parLengths = Int64[]
    for (k,v) in hpranges
        push!(parLengths,length(v))
    end
    candidates = Dict{String,Any}[]
    for ij in CartesianIndices(Tuple(parLengths)) 
        thishpars = Dict{String,Any}()
        i = 1
        for (k,v) in hpranges
            thishpars[k] = hpranges[k][Tuple(ij)[i]]
            i += 1
        end
        #thishpars = NamedTuple{Tuple(keys(thishpars))}(values(thishpars)) # dict to namedtouple, also  ntuple = (; dict...)
        push!(candidates,thishpars)
    end
    return candidates
end

"""
$(TYPEDSIGNATURES)

Hyperparameter autotuning using the [`GridSearch`](@ref) method.

"""
function tune!(m,method::GridSearch,data)
    options(m).verbosity >= STD && println("Starting hp autotuning (could take a while..)")
    options(m).verbosity >= HIGH && println(method)   
    hpranges   = method.hpranges
    candidates = _hpranges_2_candidates(hpranges)
    rng             = options(m).rng
    multithreads  = method.multithreads && Threads.nthreads() > 1
    compLock        = ReentrantLock()
    best_candidate  = Dict()
    lowest_loss     = Inf
    n_orig          = size(data[1],1)
    res_share       = method.res_share
    if n_orig * res_share < 10 
        res_share = 10 / n_orig # trick to avoid training on 1-sample, where some models have problems
    end 
    subs = partition([data...],[res_share,1-res_share],rng=rng, copy=true)
    sampleddata = (collect([subs[i][1] for i in 1:length(subs)])...,)
    masterSeed = rand(rng,100:typemax(Int64))
    rngs       = generate_parallel_rngs(rng,Threads.nthreads()) 
    n_candidates = length(candidates)
    @threadsif multithreads for c in 1:n_candidates
        candidate = candidates[c]
        tsrng = rngs[Threads.threadid()] 
        Random.seed!(tsrng,masterSeed+c*10)
        options(m).verbosity == FULL && println("Testing candidate $candidate")
        mc = deepcopy(m)
        mc.opt.autotune = false
        mc.opt.verbosity = NONE
        sethp!(mc,candidate) 
        μ =  method.loss(mc,sampleddata;rng=tsrng)   
        options(m).verbosity == FULL && println(" -- predicted loss: $μ")  
        if multithreads 
            lock(compLock) ## This step can't be run in parallel...
        end
        try
            if μ < lowest_loss
                lowest_loss = μ
                best_candidate = candidate
            end
        finally
            if multithreads
                unlock(compLock)
            end
        end
    end
    sethp!(m,best_candidate) 
end

"""
$(TYPEDSIGNATURES)

Hyperparameter autotuning using the [`SuccessiveHalvingSearch`](@ref) method.

"""
function tune!(m,method::SuccessiveHalvingSearch,data)
    options(m).verbosity >= STD && println("Starting hp autotuning (could take a while..)")
    options(m).verbosity >= HIGH && println(method)   
    hpranges   = method.hpranges
    res_shares = method.res_shares
    rng             = options(m).rng
    multithreads    = method.multithreads && Threads.nthreads() > 1
    compLock        = ReentrantLock()
    epochs          = length(res_shares)
    candidates = _hpranges_2_candidates(hpranges)
    ncandidates = length(candidates)
    shrinkfactor = ncandidates^(1/epochs)
    n_orig          = size(data[1],1)

    for e in 1:epochs
        res_share       = res_shares[e]
        if n_orig * res_share < 10 
            res_share = 10 / n_orig # trick to avoid training on 1-sample, where some models have problems
        end 
        esubs = partition([data...],[res_share,1-res_share],copy=false,rng=rng)
        epochdata = (collect([esubs[i][1] for i in 1:length(esubs)])...,)
        ncandidates_thisepoch = Int(round(ncandidates/shrinkfactor^(e-1)))
        ncandidates_tokeep = Int(round(ncandidates/shrinkfactor^e))
        options(m).verbosity >= STD && println("(e $e / $epochs) N data / candidates / candidates to retain : $(n_orig * res_share) \t $ncandidates_thisepoch $ncandidates_tokeep")
        scores = Vector{Tuple{Float64,Dict}}(undef,ncandidates_thisepoch)
        masterSeed = rand(rng,100:typemax(Int64))
        rngs       = generate_parallel_rngs(rng,Threads.nthreads()) 
        n_candidates = length(candidates)
        ncandidates_thisepoch == n_candidates || error("Problem with number of candidates!")
        @threadsif multithreads for c in 1:n_candidates
            candidate=candidates[c]
            tsrng = rngs[Threads.threadid()]
            Random.seed!(tsrng,masterSeed+c*10) 
            options(m).verbosity == FULL && println("(e $e) Testing candidate $candidate")
            mc = deepcopy(m)
            mc.opt.autotune = false
            mc.opt.verbosity = NONE
            sethp!(mc,candidate) 
            μ =  method.loss(mc,epochdata;rng=tsrng)   
            options(m).verbosity == FULL && println(" -- predicted loss: $μ")  
            scores[c] = (μ,candidate)
        end
        sort!(scores,by=first)
        options(m).verbosity == FULL && println("(e $e) Scores: \n $scores")
        candidates = [scores[i][2] for i in 1:ncandidates_tokeep]     
    end
    length(candidates) == 1 || error("Here we should have a single candidate remained!")
    sethp!(m,candidates[1]) 
end

"""
$(TYPEDSIGNATURES)

Hyperparameter autotuning.

"""
function autotune!(m,data) # or autotune!(m,data) ???
    if !(options(m).autotune)
        return m
    end
    # let's sure data is always a tuple of arrays, even for unsupervised models
    if !(eltype(data) <: AbstractArray) # data is a single array
        data = (data,)
    end
    n = size(data[1],1)
    n >= 10 || error("Too few records to autotune the model. At very least I need 1O records ($n provided)")
    tune!(m,hyperparameters(m).tunemethod,data)
    return nothing
end

"""
   class_counts_with_labels(x)

Return a dictionary that counts the number of each unique item (rows) in a dataset.

"""
function class_counts_with_labels(x;classes=nothing)
    dims = ndims(x)
    if dims == 1
        T = eltype(x)
    else
        T = Array{eltype(x),1}
    end
    if classes != nothing
        counts = Dict([u=>0 for u in classes])
    else
        counts = Dict{T,Int64}()  # a dictionary of label -> count.
    end
    for i in 1:size(x,1)
        if dims == 1
            label = x[i]
        else
            label = x[i,:]
        end
        if !(label in keys(counts))
            counts[label] = 1
        else
            counts[label] += 1
        end
    end
    return counts
end

"""
   class_counts(x;classes=nothing)

Return a (unsorted) vector with the counts of each unique item (element or rows) in a dataset.

If order is important or not all classes are present in the data, a preset vectors of classes can be given in the parameter `classes`

"""
function class_counts(x; classes=nothing)
   if classes == nothing # order doesn't matter
      return values(class_counts_with_labels(x;classes=classes))
   else
       cWithLabels = class_counts_with_labels(x;classes=classes)
       return [cWithLabels[k] for k in classes]
   end
end






"""
   mode(dict::Dict{T,Float64};rng)

Return the key with highest mode (using rand in case of multimodal values)

"""
function mode(dict::Dict{T,Float64};rng = Random.GLOBAL_RNG) where {T}
    mks = [k for (k,v) in dict if v==maximum(values(dict))]
    if length(mks) == 1
        return mks[1]
    else
        return mks[rand(rng,1:length(mks))]
    end
end

"""
   mode(v::AbstractVector{T};rng)

Return the position with the highest value in an array, interpreted as mode (using rand in case of multimodal values)

"""
function mode(v::AbstractVector{T};rng = Random.GLOBAL_RNG) where {T <: Number}
    mpos = findall(x -> x == maximum(v),v)
    if length(mpos) == 1
        return mpos[1]
    else
        return mpos[rand(rng,1:length(mpos))]
    end
end


"""
  mode(elements,rng)

Given a vector of dictionaries whose key is numerical (e.g. probabilities), a vector of vectors or a matrix, it returns the mode of each element (dictionary, vector or row) in terms of the key or the position.

Use it to return a unique value from a multiclass classifier returning probabilities.

# Note:
- If multiple classes have the highest mode, one is returned at random (use the parameter `rng` to fix the stochasticity)

"""
function mode(dicts::AbstractArray{Dict{T,Float64}};rng = Random.GLOBAL_RNG) where {T}
    return mode.(dicts;rng=rng)
end

function mode(vals::AbstractArray{T,1};rng = Random.GLOBAL_RNG) where {T <: AbstractArray{T2,1} where T2 <: Number}
    return mode.(vals;rng=rng)
end
function mode(vals::AbstractArray{T,2};rng = Random.GLOBAL_RNG) where {T <: Number}
    return [mode(r;rng=rng) for r in eachrow(vals)]
end


"""
   mean_dicts(dicts)

Compute the mean of the values of an array of dictionaries.

Given `dicts` an array of dictionaries, `mean_dicts` first compute the union of the keys and then average the values.
If the original valueas are probabilities (non-negative items summing to 1), the result is also a probability distribution.

"""
function mean_dicts(dicts; weights=ones(length(dicts)))
    if length(dicts) == 1
        return dicts[1]
    end
    T = eltype(keys(dicts[1]))
    allkeys = union([keys(i) for i in dicts]...)
    outDict = Dict{T,Float64}()
    ndicts = length(dicts)
    totWeights = sum(weights)
    for k in allkeys
        v = 0
        for (i,d) in enumerate(dicts)
            if k in keys(d)
                v += (d[k])*(weights[i]/totWeights)
            end
        end
        outDict[k] = v
    end

    return outDict
end

# ------------------------------------------------------------------------------
# Other mathematical/computational functions

""" LogSumExp for efficiently computing log(sum(exp.(x))) """
lse(x) = maximum(x)+log(sum(exp.(x .- maximum(x))))
""" Sterling number: number of partitions of a set of n elements in k sets """
sterling(n::BigInt,k::BigInt) = (1/factorial(k)) * sum((-1)^i * binomial(k,i)* (k-i)^n for i in 0:k)
sterling(n::Int64,k::Int64)   = sterling(BigInt(n),BigInt(k))
