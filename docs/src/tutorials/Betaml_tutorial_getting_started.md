# [Getting started](@id getting_started)

## Introduction

This "tutorial" part of the documentation presents a step-by-step guide to the main algorithms and utility functions provided by BetaML and comparisons with the leading packages in each field.
Aside this page, the tutorial is divided in the following sections:

-  [Classification tutorial](@ref classification_tutorial) - Topics: _Decision trees and random forests, neural networks (softmax), dealing with stochasticity, loading data from internet_
-  [Regression tutorial](@ref regression_tutorial) - Topics: _Decision trees, Random forests, neural networks, hyper-parameters autotuning, one-hot encoding, continuous error measures_
-  [Clustering tutorial](@ref clustering_tutorial) - Topics: _k-means, kmedoids, generative (gaussian) mixture models (gmm), cross-validation, ordinal encoding_

Detailed usage instructions on each algorithm can be found on each model struct (listed [here](@ref models_list)), while theoretical notes describing most of them can be found at the companion repository [https://github.com/sylvaticus/MITx_6.86x](https://github.com/sylvaticus/MITx_6.86x).

The overall "philosophy" of BetaML is to support simple machine learning tasks easily and make complex tasks possible. An the most basic level, the majority of  algorithms have default parameters suitable for a basic analysis. A great level of flexibility can be already achieved by just employing the full set of model parameters, for example changing the distance function in `KMedoidsClusterer` to `l1_distance` (aka "Manhattan distance").
Finally, the greatest flexibility can be obtained by customising BetaML and writing, for example, its own neural network layer type (by subclassing `AbstractLayer`), its own sampler (by subclassing `AbstractDataSampler`) or its own mixture component (by subclassing `AbstractMixture`),
In such a cases, while not required by any means, please consider to give it back to the community and open a pull request to integrate your work in BetaML.

If you are looking for an introductory book on Julia, you could consider "[Julia Quick Syntax Reference](https://www.julia-book.com/)" (Apress,2019) or the online course "[Introduction to Scientific Programming and Machine Learning with Julia](https://sylvaticus.github.io/SPMLJ/stable/)".

A few conventions applied across the library:
- Type names use the so-called "CamelCase" convention, where the words are separated by a capital letter rather than `_` ,while function names use lower letters only, with words eventually separated (but only when really neeed for readibility) by an `_`;
- While some functions provide a `dims` parameter, most BetaML algorithms expect the input data layout with observations organised by rows and fields/features by columns. Almost everywhere in the code and documentation we refer with `N` the number of observations/records, `D` the number of dimensions and `K` the number of classes/categories;
- While some algorithms accept as input DataFrames, the usage of standard arrays is encourages (if the data is passed to the function as dataframe, it may be converted to standard arrays somewhere inside inner loops, leading to great inefficiencies)
- The accuracy/error/loss measures expect the ground true `y` and then the estimated `ŷ` (in this order)


## [Using BetaML from other programming languages](@id using_betaml_from_other_languages)

Thanks to respectively [PyJulia](https://github.com/JuliaPy/pyjulia) and [JuliaCall](https://github.com/Non-Contradiction/JuliaCall), using BetaML in Python or R is almost as simple as using a native library.
In both cases we need first to download and install the Julia binaries for our operating system from [JuliaLang.org](https://julialang.org/). Be sure that Julia is working by opening the Julia terminal and e.g. typing `println("hello world")` (JuliaCall has an option to install a private-to-R version of Julia from within R).
Also, in both case we do not need to think to converting Python/R objects to Julia objects when calling a Julia function and converting back the result from the Julia object to a Pytoh or R object, as this is handled automatically by PyJulia and JuliaCall, at least for simple types (arrays, strings,...)

### Use BetaML in Python

```
$ python3 -m pip install --user julia   # the name of the package in `pip` is `julia`, not `PyJulia`
```
For the sake of this tutorial, let's also install in Python a package that contains the dataset that we will use:
```
$ python3 -m pip install --user sklearn # only for retrieving the dataset in the python way
```
We can now open a Python terminal and, to obtain an interface to Julia, just run:

```python
>>> import julia
>>> julia.install() # Only once to set-up in julia the julia packages required by PyJulia
>>> jl = julia.Julia(compiled_modules=False)
```
If we have multiple Julia versions, we can specify the one to use in Python passing `julia="/path/to/julia/binary/executable"` (e.g. `julia = "/home/myUser/lib/julia-1.8.0/bin/julia"`) to the `install()` function.

The `compiled_module=False` in the Julia constructor is a workaround to the common situation when the Python interpreter is statically linked to `libpython`, but it will slow down the interactive experience, as it will disable Julia packages pre-compilation, and every time we will use a module for the first time, this will need to be compiled first.
Other, more efficient but also more complicate, workarounds are given in the package documentation, under the https://pyjulia.readthedocs.io/en/stable/troubleshooting.html[Troubleshooting section].

Let's now add to Julia the BetaML package. We can surely do it from within Julia, but we can also do it while remaining in Python:

```python
>>> jl.eval('using Pkg; Pkg.add("BetaML")') # Only once to install BetaML
```

While `jl.eval('some Julia code')` evaluates any arbitrary Julia code (see below), most of the time we can use Julia in a more direct way. Let's start by importing the BetaML Julia package as a submodule of the Python Julia module:

```python
>>> from julia import BetaML
```

As you can see, it is no different than importing any other Python module.

For the data, let's load it "Python side":

```python
>>> from sklearn import datasets
>>> iris = datasets.load_iris()
>>> X    = iris.data[:, :4]
>>> y    = iris.target + 1 # Julia arrays start from 1 not 0
```
Note that `X` and `y` are Numpy arrays.

We can now call BetaML functions as we would do for any other Python library functions. In particular, we can pass to the functions (and retrieve) complex data types without worrying too much about the conversion between Python and Julia types, as these are converted automatically:

```python
>>> (Xs,ys) = BetaML.shuffle([X,y]) # X and y are first converted to julia arrays and then the returned julia arrays are converted back to python Numpy arrays
>>> m       = BetaML.KMeansClusterer(n_classes=3)
>>> yhat    = BetaML.fit_ex(m,Xs) # Python doesn't allow exclamation marks in function names, so we use `fit_ex(⋅)` instead of `fit!(⋅)`
>>> acc     = BetaML.accuracy(yhat,ys,ignorelabels=True)
>>> acc
 0.8933333333333333
```

Note: If we are using the `jl.eval()` interface, the objects we use must be already known to julia. To pass objects from Python to Julia, import the julia `Main` module (the root module in julia) and assign the needed variables, e.g.

```python
>>> X_python = [1,2,3,2,4]
>>> from julia import Main
>>> Main.X_julia = X_python
>>> jl.eval('BetaML.gini(X_julia)')
0.7199999999999999
```

Another alternative is to "eval" only the function name and pass the (python) objects in the function call:

```python
>>> jl.eval('BetaML.gini')(X_python)
0.7199999999999999
```

Using either the direct call or the `eval` function you should be able to use all the BetaML functionalities directly from Python. If you run into problems using BetaML from Python, [open an issue](https://github.com/sylvaticus/BetaML.jl/issues/new) specifying your set-up.


### Use BetaML in R

We start by installing the `JuliaCall` R package:

```{r}
> install.packages("JuliaCall")
> library(JuliaCall)
> julia_setup(installJulia = FALSE) # use installJulia = FALSE to let R download and install a private copy of julia
```

Note that, differently than `PyJulia`, the "setup" function needs to be called every time we start a new R section, not just when we install the `JuliaCall` package.
If we don't have `julia` in the path of our system, or if we have multiple versions and we want to specify the one to work with, we can pass the `JULIA_HOME = "/path/to/julia/binary/executable/directory"` (e.g. `JULIA_HOME = "/home/myUser/lib/julia-1.1.0/bin"`) parameter to the `julia_setup` call. Or just let `JuliaCall` automatically download and install a private copy of julia.

`JuliaCall` depends for some things (like object conversion between Julia and R) from the Julia `RCall` package. If we don't already have it installed in Julia, it will try to install it automatically.

As in Python, let's start from the data loaded from R and do some work with them in Julia:

```{r}
> library(datasets)
> X     <- as.matrix(sapply(iris[,1:4], as.numeric))
> y     <- sapply(iris[,5], as.integer)
> xsize <- dim(X)
```

Let's install BetaML. As we did in Python, we can install a Julia package from Julia itself or from within R:

```{r}
> julia_eval('using Pkg; Pkg.add("BetaML")')
```

We can now "import" the BetaML julia package (in julia a "Package" is basically a module plus some metadata that facilitate its discovery and integration with other packages, like the reuired set) and call its functions with the `julia_call("juliaFunction",args)` R function:

```{r}
> julia_eval("using BetaML")
> shuffled <- julia_call("shuffle",list(X,y))
> Xs       <- matrix(sapply(shuffled[1],as.numeric), nrow=xsize[1])
> ys       <- as.vector(sapply(shuffled[2], as.integer))
> m        <- julia_eval('KMeansClusterer(n_classes=3)')
> yhat     <- julia_call("fit_ex",m,Xs)
> acc      <- julia_call("accuracy",yhat,ys,ignorelabels=TRUE)
> acc
[1] 0.8933333
```

As alternative, we can embed Julia code directly in R using the `julia_eval()` function:

```{r}
kMeansR  <- julia_eval('
  function accFromKmeans(x,k,y)
    m    = KMeansClusterer(n_classes=Int(k))
    yhat = fit!(m,x)
    acc  = accuracy(yhat,y,ignorelabels=true)
    return acc
  end
')
```

We can then call the above function in R in one of the following three ways:
1. `kMeansR(Xs,3,ys)`
2. `julia_assign("Xs_julia", Xs); julia_assign("ys_julia", ys); julia_eval("accFromKmeans(Xs_julia,3,ys_julia)")`
3. `julia_call("accFromKmeans",Xs,3,ys)`.

While other "convenience" functions are provided by the package, using  `julia_call`, or `julia_assign` followed by `julia_eval`, should suffix to use `BetaML` from R. If you run into problems using BetaML from R, [open an issue](https://github.com/sylvaticus/BetaML.jl/issues/new) specifying your set-up.

## [Dealing with stochasticity and reproducibility](@id dealing_with_stochasticity)

Machine Learning workflows include stochastic components in several steps: in the data sampling, in the model initialisation and often in the models's own algorithms (and sometimes also in the prediction step).
All BetaML models with a stochastic components support a `rng` parameter, standing for _Random Number Generator_. A RNG is a "machine" that streams a flow of random numbers. The flow itself however is deterministically determined for each "seed" (an integer number) that the RNG has been told to use.
Normally this seed changes at each running of the script/model, so that stochastic models are indeed stochastic and their output differs at each run.

If we want to obtain reproductible results we can fix the seed at the very beginning of our model with `Random.seed!([AnInteger])`. Now our model or script will pick up a specific flow of random numbers, but this flow will always be the same, so that its results will always be the same.

However the default Julia RNG guarantee to provide the same flow of random numbers, conditional to the seed, only within minor versions of Julia. If we want to "guarantee" reproducibility of the results with different versions of Julia, or "fix" only some parts of our script, we can call the individual functions passing [`FIXEDRNG`](@ref), an instance of `StableRNG(FIXEDSEED)` provided by `BetaML`, to the `rng` parameter. Use it with:


- `MyModel(;rng=FIXEDRNG)`               : always produce the same sequence of results on each run of the script ("pulling" from the same rng object on different calls)
- `MyModel(;rng=StableRNG(SOMEINTEGER))` : always produce the same result (new identical rng object on each call)

This is very convenient expecially during model development, as a model that use `(...,rng=StableRNG(an_integer))` will provides stochastic results that are isolated (i.e. they don't depend from the consumption of the random stream from other parts of the model).

In particular, use `rng=StableRNG(FIXEDSEED)` or `rng=copy(FIXEDRNG)` with [`FIXEDSEED`](@ref)  to retrieve the exact output as in the documentation or in the unit tests.

Most of the stochasticity appears in _training_ a model. However in few cases (e.g. decision trees with missing values) some stochasticity appears also in _predicting_ new data using a trained model. In such cases the model doesn't restrict the random seed, so that you can choose at _predict_ time to use a fixed or a variable random seed.

Finally, if you plan to use multiple threads and want to provide the same stochastic output independent to the number of threads used, have a look at [`generate_parallel_rngs`](@ref).

"Reproducible stochasticity" is only one of the elements needed for a reproductible output. The other two are (a) the inputs the workflow uses and (b) the code that is evaluated.
Concerning the second point Julia has a very modern package system that guarantee reproducible code evaluation (with a few exception linked to using external libraries, but BetaML models are all implemented in Julia itself). Without going in detail, you can use a pattern like this at the beginning of your machine learning workflows:

```        
using Pkg  
cd(@__DIR__)            
Pkg.activate(".")  # Activate a "local" environment, specific to this folder
Pkg.instantiate()  # Download and install the required packages if not already available 
```

This will tell Julia to load the exact version of dependent packages, and recursively of their dependencies, from a `Manifest.toml` file that is automatically created in the script's folder, and automatically updated, when you add or update a package in your workflow.
Note that these locals "environments" are very "cheap" (packages are not actually copied to each environment on your system, only referenced) and the environment doen't need to be in the same script folder as in this example, can be any folder you want to "activate".

## Saving and loading trained models

Trained models can be saved on disk using the [`model_save`](@ref) function, and retrieved with [`model_load`](@ref).
The advantage over the serialization functionality in Julia core is that the two functions are actually wrappers around equivalent [JLD2](https://juliaio.github.io/JLD2.jl/stable/) package functions, and should maintain compatibility across different Julia versions. 