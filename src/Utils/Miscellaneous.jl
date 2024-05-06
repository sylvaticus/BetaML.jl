"Part of [BetaML](https://github.com/sylvaticus/BetaML.jl). Licence is MIT."

# Part of submodule Utils of BetaML _ the Beta Machine Learning Toolkit
# Miscelaneaous funcitons / types

using Base.Threads
using Base.Threads: threadid, threading_run

@static if VERSION ≥ v"1.9-rc1"
    # This makes the threadsif macro work
    using Base.Threads: threadpoolsize
end

"""
$(TYPEDSIGNATURES)

Conditionally apply multi-threading to `for` loops.
This is a variation on `Base.Threads.@threads` that adds a run-time boolean flag to enable or disable threading. 
    
# Example:
```julia
function optimize(objectives; use_threads=true)
    @threadsif use_threads for k = 1:length(objectives)
    # ...
    end
end

# Notes:
- Borrowed from https://github.com/JuliaQuantumControl/QuantumControlBase.jl/blob/master/src/conditionalthreads.jl
```
"""
macro threadsif(cond, loop)
    if !(isa(loop, Expr) && loop.head === :for)
        throw(ArgumentError("@threadsif requires a `for` loop expression"))
    end
    if !(loop.args[1] isa Expr && loop.args[1].head === :(=))
        throw(ArgumentError("nested outer loops are not currently supported by @threadsif"))
    end
    quote
        if $(esc(cond))
            $(Threads._threadsfor(loop.args[1], loop.args[2], :static))
        else
            $(esc(loop))
        end
    end
end

# Attention, it uses Julia internals!
get_parametric_types(obj) = typeof(obj).parameters

isinteger_bml(_)          = false
isinteger_bml(_::Integer) = true
isinteger_bml(_::Nothing) = error("Trying to run isinteger() over a `Nothing` value")
isinteger_bml(_::Missing) = missing
isinteger_bml(x::AbstractFloat)   = isinteger(x)

"""
    online_mean(new;mean=0.0,n=0)

Update the mean with new values.   
"""
online_mean(new;mean=0.0,n=0) =  ((mean*n)+new)/(n+1)
