"Part of [BetaML](https://github.com/sylvaticus/BetaML.jl). Licence is MIT."

# Part of submodule Utils of BetaML - the Beta Machine Learning Toolkit
# Various helper/ utility functions concerning stochasticity management


#StableRNG(FIXEDSEED) Random.default_rng() #MersenneTwister(FIXEDSEED)
#const FIXEDRNG  = MersenneTwister(FIXEDSEED) #StableRNG(FIXEDSEED) Random.default_rng()


"""
    generate_parallel_rngs(rng::AbstractRNG, n::Integer;reSeed=false)

DEPRECATED: There seem to be issues when julia is run with -t1,1
For nthread-independent replicable parallel computation, rather sample a masterseed before the parallel computation and then, inside the parallel loop, deepcopy the RNG and reseed it using the masterseed and an i-th specific seed. 
For multi-threaded models, return n independent random number generators (one per thread) to be used in threaded computations.

Note that each rng is a _copy_ of the original rng. This means that code that _uses_ these RNGs will not change the original RNG state.

Use it with `rngs = generate_parallel_rngs(rng,Threads.nthreads())` to have a separate rng per thread.
**Attention**: `Threads.threadid()` of working threads is between `Threads.nthreads(:interactive)` and `Threads.nthreads(:interactive)+Threads.nthreads(:default)`
By default the function doesn't re-seed the RNG, as you may want to have a loop index based re-seeding strategy rather than a threadid-based one (to guarantee the same result independently of the number of threads).
If you prefer, you can instead re-seed the RNG here (using the parameter `reSeed=true`), such that each thread has a different seed. Be aware however that the stream  of numbers generated will depend on the number of threads at run time.
"""
function generate_parallel_rngs(rng::AbstractRNG, n::Integer;reSeed=false)
    if reSeed
        seeds = [rand(rng,100:Int(ceil(typemax(Int64)/1000))) for i in 1:n] # some RNGs have issues with too small seed
        rngs  = [deepcopy(rng) for i in 1:n]
        return Random.seed!.(rngs,seeds)
    else
        return [deepcopy(rng) for i in 1:n]
    end
end





"""
    consistent_shuffle(data;dims,rng)

Shuffle a vector of n-dimensional arrays across dimension `dims` keeping the same order between the arrays

# Parameters
- `data`: The vector of arrays to shuffle
- `dims`: The dimension over which to apply the shuffle [def: `1`]
- `rng`:  An `AbstractRNG` to use for the shuffle

# Notes
- All the arrays must have the same size for the dimension to shuffle

# Example
```
julia> a = [1 2 30; 10 20 30]; b = [100 200 300];
julia> (aShuffled, bShuffled) = consistent_shuffle([a,b],dims=2)
2-element Vector{Matrix{Int64}}:
 [1 30 2; 10 30 20]
 [100 300 200]
 ```
"""
function consistent_shuffle(data::AbstractArray{T,1};dims=1,rng=Random.GLOBAL_RNG) where T <: Any
    #= old code, fast for small data, slow for big elements to shuffle
    Ns = [size(m,dims) for m in data]
    length(Set(Ns)) == 1 || @error "In `consistent_shuffle(arrays)` all individual arrays need to have the same size on the dimension specified"
    N    = Ns[1]
    ridx = Random.shuffle(rng, 1:N)
    out = similar(data)
    for (i,a) in enumerate(data)
       aidx = convert(Vector{Union{UnitRange{Int64},Vector{Int64}}},[1:i for i in size(a)])
       aidx[dims] = ridx
       out[i] = a[aidx...]
    end
    return out
    =#
    Ns = [size(m,dims) for m in data]
    length(Set(Ns)) == 1 || @error "In `consistent_shuffle(arrays)` all individual arrays need to have the same size on the dimension specified"
    ix = shuffle(rng,1:size(data[1],dims)) # `shuffle` rather than `randperm`, as it is guaranteed to be stable across Julia versions with StableRNGs
    return mapslices.(x->x[ix], data, dims=dims)
end
consistent_shuffle(rng::AbstractRNG,data::AbstractArray{T,1};dims=1) where T <: Any = consistent_shuffle(data;dims=dims,rng=rng)