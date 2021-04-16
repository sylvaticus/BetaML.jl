```@meta
EditURL = "<unknown>/src/tutorials/Getting started/betaml_tutorial_getting_started.jl"
```

# [Getting started](@id getting_started)
## This is markdown title
This is also markdown

```@example betaml_tutorial_getting_started
# This is a normal comment

a = 1
b = a + 1
println("*** I am a line of code that is executed, and b is $b")
b
```

A markdown cell

```@example betaml_tutorial_getting_started
using BenchmarkTools
c = b + 1
c
```

```@example betaml_tutorial_getting_started
k = 100000
@btime sum(1:k);
nothing #hide
```

```@example betaml_tutorial_getting_started
a = [1,2]
```

```@example betaml_tutorial_getting_started
println("hello")
```

## [Dealing with stochasticity](@id dealing_with_stochasticity)

Most models have some stochastic components and support a `rng` parameter. By default, the outputs of these models will hence not be absolutelly equal on each run. If you want to be sure that the output of a model remain constant given the same inputs you can pass a fixed Random Number Generator to the `rng` parameter. Use it with:

- `myAlgorithm(;rng=FIXEDRNG)`               # always produce the same sequence of results on each run of the script ("pulling" from the same rng object on different calls)
- `myAlgorithm(;rng=StableRNG(SOMEINTEGER))` # always produce the same result (new rng object on each call)

In particular, use `rng=StableRNG(FIXEDSEED)` to retrieve the exacty output as in the documentation or in the unit tests.


Most of the stochasticity appears in _training_ a model. However in few cases (e.g. decision trees with missing values) some stocasticity appears also in _predicting_ new data with a trained model. In such cases the model doesn't stire the random seed, so that you can choose at _predict_ time to use a fixed or a variable random seed.

[View this file on Github](<unknown>/src/tutorials/Getting started/betaml_tutorial_getting_started.jl).

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

