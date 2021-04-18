# Part of submodule Utils of BetaML - the Beta Machine Learning Toolkit
# Various helper/ utility functions concerning stochastiticy management


"""
    FIXEDSEED

Fixed seed to allow reproducible results.
This is the seed used to obtain the same results under unit tests.

Use it with:
- `myAlgorithm(;rng=FIXEDRNG)`             # always produce the same sequence of results on each run of the script ("pulling" from the same rng object on different calls)
- `myAlgorithm(;rng=copy(FIXEDRNG)`        # always produce the same result (new rng object on each call)
"""
const FIXEDSEED = 123

"""
    FIXEDRNG

Fixed ring to allow reproducible results

Use it with:
- `myAlgorithm(;rng=FIXEDRNG)`         # always produce the same sequence of results on each run of the script ("pulling" from the same rng object on different calls)
- `myAlgorithm(;rng=copy(FIXEDRNG))`   # always produce the same result (new rng object on each function call)

"""
const FIXEDRNG  = StableRNG(FIXEDSEED) #StableRNG(FIXEDSEED) Random.default_rng() #MersenneTwister(FIXEDSEED)
#const FIXEDRNG  = MersenneTwister(FIXEDSEED) #StableRNG(FIXEDSEED) Random.default_rng()


"""
    generateParallelRngs(rng::AbstractRNG, n::Integer;reSeed=false)

For multi-threaded models, return n independent random number generators (one per thread) to be used in threaded computations.

Note that each ring is a _copy_ of the original random ring. This means that code that _use_ these RNGs will not change the original RNG state.

Use it with `rngs = generateParallelRngs(rng,Threads.nthreads())` to have a separate rng per thread.
By default the function doesn't re-seed the RNG, as you may want to have a loop index based re-seeding strategy rather than a threadid-based one (to guarantee the same result independently of the number of threads).
If you prefer, you can instead re-seed the RNG here (using the parameter `reSeed=true`), such that each thread has a different seed. Be aware however that the stream  of number generated will depend from the number of threads at run time.
"""
function generateParallelRngs(rng::AbstractRNG, n::Integer;reSeed=false)
    if reSeed
        seeds = [rand(rng,100:18446744073709551615) for i in 1:n] # some RNGs have issues with too small seed
        rngs  = [deepcopy(rng) for i in 1:n]
        return Random.seed!.(rngs,seeds)
    else
        return [deepcopy(rng) for i in 1:n]
    end
end





"""
    shuffle(data;dims,rng)

Shuffle a vector of n-dimensional arrays across dimension `dims` keeping the same order between the arrays

# Parameters
- `data`: The vector of arrays to shuffle
- `dims`: The dimension over to apply the shuffle [def: `1`]
- `rng`:  An `AbstractRNG` to apply for the shuffle

# Notes
- All the arrays must have the same size for the dimension to shuffle

# Example
```
julia> a = [1 2 30; 10 20 30]; b = [100 200 300];
julia> (aShuffled, bShuffled) = shuffle([a,b],dims=2)
2-element Vector{Matrix{Int64}}:
 [1 30 2; 10 30 20]
 [100 300 200]
 ```
"""
function shuffle(data::AbstractArray{T,1};dims=1,rng=Random.GLOBAL_RNG)  where T <: AbstractArray
    Ns = [size(m,dims) for m in data]
    length(Set(Ns)) == 1 || @error "In `shuffle(arrays)` all individual arrays need to have the same size on the dimension specified"
    N    = Ns[1]
    ridx = Random.shuffle(rng, 1:N)
    out = similar(data)
    for (i,a) in enumerate(data)
       aidx = convert(Vector{Union{UnitRange{Int64},Vector{Int64}}},[1:i for i in size(a)])
       #aidx = [collect(1:i) for i in size(a)]
       aidx[dims] = ridx
       out[i] = a[aidx...]
    end
    return out
end
shuffle(rng::AbstractRNG,data::AbstractArray{T,1};dims=1) where T <: AbstractArray = shuffle(data;dims=dims,rng=rng)
