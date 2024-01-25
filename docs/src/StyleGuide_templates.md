# Style guide and template for BetaML developers

## Master Style guide

The code in BetaML should follow the official [Julia Style Guide](https://docs.julialang.org/en/v1/manual/style-guide/).

## Names style

- Each file name should start with a capital letter, no spaces allowed (and each file content should start with: `"Part of [BetaML](https://github.com/sylvaticus/BetaML.jl). Licence is MIT."`)
- Type names use the so-called "CamelCase" convention, where the words are separated by a capital letter rather than `_` ,while function names use lower letters only, with words eventually separated (but only when really neeed for readibility) by an `_`;
- In the code and documentation we refer with `N` the number of observations/records, `D` the number of dimensions and `K` the number of classes/categories;
- Error/accuracy/loss functions want firt `y` and then `ŷ`
- In API exposed to users, strings are preferred to symbols

## Docstrings

Please apply the following templates when writing a docstring for BetaML:

- Functions (add `@docs` if the function is not on the root module level, like for inner constructors, i.e. `@docs """ foo()x ...."""`):

```
"""
$(TYPEDSIGNATURES)

One line description

[Further description]

# Parameters:



# Returns:
- Elements the funtion need

# Notes:
- notes

# Example:
` ` `julia
julia> [code]
[output]
` ` `
"""
```

- Structs

```
"""
$(TYPEDEF)

One line description

[Further description]

# Fields: (if relevant)
$(TYPEDFIELDS)

# Notes:

# Example:
` ` `julia
julia> [code]
[output]
` ` `

"""
```

- Enums:

```
"""
$(TYPEDEF)

One line description

[Further description]


# Notes:

"""
```

- Constants

```
"""
[4 spaces] [Constant name]

One line description

[Further description]


# Notes:

"""
```

- Modules

```
"""
[4 spaces] [Module name]

One line description

Detailed description on the module objectives, content and organisation

"""
```

## Internal links

To refer to a documented object: ```[`NAME`](@ref)``` or ```[`NAME`](@ref manual_id)```.
In particular for internal links use ```[`?NAME`](@ref ?NAME)```

To create an id manually: ```[Title](@id manual_id)```

## Data organisation

- While some functions provide a `dims` parameter, most BetaML algorithms expect the input data layout with observations organised by rows and fields/features by columns.
- While some algorithms accept as input DataFrames, the usage of standard arrays is encourages (if the data is passed to the function as dataframe, it may be converted to standard arrays somewhere inside inner loops, leading to great inefficiencies).



