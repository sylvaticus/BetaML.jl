import Base.iterate
abstract type  AbstractDataSampler end

mutable struct SamplerWithData{Ts <: AbstractDataSampler, Td <: AbstractArray}
    sampler::Ts
    data::Td
    dims::Int64
end

# To implement a new sampler:
# - create a new structure child of AbstractDataSampler
# - override iterate(iter::SamplerWithData{yoursampler} and iterate(iter::SamplerWithData{yoursampler},state) considering that
#=
for i in iter   # or  "for i = iter"
    # body
end

# --> is rewritten to :

next = iterate(iter)
while next !== nothing
    (i, state) = next
    # body
    next = iterate(iter, state)
end
=#
"""
   KFold(nSplits=5,nRepeats=1,shuffle=true,rng=Random.GLOBAL_RNG)

Iterator for k-fold crossValidation strategy.

"""
mutable struct KFold <: AbstractDataSampler
    nSplits::Int64
    nRepeats::Int64
    shuffle::Bool
    rng::AbstractRNG
    function KFold(;nSplits=5,nRepeats=1,shuffle=true,rng=Random.GLOBAL_RNG)
        return new(nSplits,nRepeats,shuffle,rng)
    end
end

# Implementation of the Julia iteration  API for SamplerWithData{KFold}
function iterate(iter::SamplerWithData{KFold})
     # First iteration, I need to create the subsamples
     K    = iter.sampler.nSplits
     D    = iter.dims
     if eltype(iter.data) <: AbstractArray # data has multiple arrays, like X,Y
       subs = collect(zip(partition(iter.data,fill(1/K,K),shuffle=iter.sampler.shuffle,dims=D,rng=iter.sampler.rng,copy=false)...))
     else # data is a single matrix/tensor
       subs = collect(zip(partition(iter.data,fill(1/K,K),shuffle=iter.sampler.shuffle,dims=D,rng=iter.sampler.rng,copy=false)))
     end
     i    = (cat.(subs[2:K]...,dims=D),subs[1])
     next = (subs,2)
     return (i,next)
end

function iterate(iter::SamplerWithData{KFold},state)
     # Further iteration, I need to create the subsamples only if it is a new interaction
     K    = iter.sampler.nSplits
     D    = iter.dims
     nRep = iter.sampler.nRepeats
     subs    = state[1]
     counter = state[2]
     counter <= (K * nRep) || return nothing  # If we are done all the splits by the repetitions we are done
     kpart = counter % K
     if kpart == 1 # new round, we repartition in k parts
         if eltype(iter.data) <: AbstractArray # data has multiple arrays, like X,Y
             subs = collect(zip(partition(iter.data,fill(1/K,K),shuffle=iter.sampler.shuffle,dims=D,rng=iter.sampler.rng,copy=false)...))
         else # data is a single matrix
             subs = collect(zip(partition(iter.data,fill(1/K,K),shuffle=iter.sampler.shuffle,dims=D,rng=iter.sampler.rng,copy=false)))
         end
         i    = (cat.(subs[2:end]...,dims=D),subs[1])
         next = (subs,counter+1)
         return (i,next)
     else
        if kpart == 0 # the modulo returns the last element as zero instead as K
            i   = (cat.(subs[1:K-1]...,dims=D),subs[end])
        else
            i   = (cat.(subs[1:kpart-1]...,subs[kpart+1:end]...,dims=D),subs[kpart])
        end
        next  = (subs,counter+1)
        return (i,next)
    end
end




#=
# old  iterationfunctions
# with this the single matrix X case doesn't need to be called with d[1][1] but just d[1]
# the case XY here doesn't work


function iterate(iter::SamplerWithData{KFold})
     # First iteration, I need to create the subsamples
     K    = iter.sampler.nSplits
     D    = iter.dims
     subs = partition(iter.data,fill(1/K,K),shuffle=iter.sampler.shuffle,dims=D,rng=iter.sampler.rng,copy=false)
     i    = (cat(subs[2:end]...,dims=D),subs[1])
     next = (subs,2)
     return (i,next)
end

function iterate(iter::SamplerWithData{KFold},state)
     # Further iteration, I need to create the subsamples only if it is a new interaction
     K    = iter.sampler.nSplits
     D    = iter.dims
     nRep = iter.sampler.nRepeats
     subs    = state[1]
     counter = state[2]
     counter <= (K * nRep) || return nothing  # If we are done all the splits by the repetitions we are done
     kpart = counter % K
     if kpart == 1 # new round, we repartition in k parts
         subs = partition(iter.data,fill(1/K,K),shuffle=iter.sampler.shuffle,dims=D,rng=iter.sampler.rng,copy=false)
         i    = (cat(subs[2:end]...,dims=D),subs[1])
         next = (subs,counter+1)
         return (i,next)
     else
        if kpart == 0 # the modulo returns the last element as zero instead as K
            i   = (cat(subs[1:K-1]...,dims=D),subs[end])
        else
            i   = (cat(subs[1:kpart-1]...,subs[kpart+1:end]...,dims=D),subs[kpart])
        end
        next  = (subs,counter+1)
        return (i,next)
    end
end

=#
