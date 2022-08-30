# Style guide and template for BetaML developers

### Master Style guide

The Style guide should follow the official Julia Style Guide: https://docs.julialang.org/en/v1/manual/style-guide/


### File management

Each file name shoudl start with a capital letter, no spaces allowed, and each file content should start with:

`"Part of [BetaML](https://github.com/sylvaticus/BetaML.jl). Licence is MIT."`

### Docstrings

Please apply the following templates when writing a docstring for BetaML:

- Functions

```
"""
$(TYPEDSIGNATURES)

One line description

[Further description]

# Parameters:

$(TYPEDFIELDS)

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

### Internal links

To refer to a documented object: `[\`NAME\`](@ref)` or `[\`NAME\`](@ref manual_id)`.
In particular for internal links use `[\`?NAME\`](@ref ?NAME)`

To create a id manually: `[Title](@id manual_id)`.