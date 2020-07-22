---
title: 'BetaML: The Beta Machine Learning Toolkit, a self-contained repository of Machine Learning algorithms in Julia'
tags:
  - Julia
  - machine learning
  - neural networks
  - deep learning
  - clustering
  - perceptron
  - data science
authors:
  - name: Antonello Lobianco^[Corresponding author.]
    orcid: 0000-0002-1534-8697
    affiliation: "1, 2, 3, 4, 5, 6" # (Multiple affiliations must be quoted)
affiliations:
 - name: Université de Lorraine
   index: 1
 - name: Université de Strasbourg
   index: 2
 - name: AgroParisTech
   index: 3
 - name: CNRS
   index: 4
 - name: INRAE
   index: 5
 - name: BETA
   index: 6
date: 1 August 2020
bibliography: docs/paper/paper.bib
---

This paper is work in progress !!

<!-- Test it with:

`pandoc --filter pandoc-citeproc --bibliography docs/paper/paper.bib  paper.md -o paper.pdf`

-->

# Summary

A serie of _machine learning_ algorithms has been implemented and bundled in a single package for the Julia language.
Currently, algorithms are available in the area of classification (perceptron, kernel perceptron, pegasos), neural networks and clustering (kmeans, kmenoids, EM, missing values attribution). Development of these algorithms started following the theoretical notes of the MOOC class "Machine Learning with Python: from Linear Models to Deep Learning" from MITx/edX.

This paper presents the motivations and the general approach of the package and gives an overview of its organisation. We refer the reader to the [package documentation](https://sylvaticus.github.io/BetaML.jl/stable) for instructions on how to use the various algorithms provided or to the MOOC notes available on GitHub [@Lobianco:2020] for their mathematical backgrounds.


# Motivations and objectives

`BetaML` provides one of the simplest way to run ML algorithms in Julia. While many packages already implement specific ML algorithms in Julia, these are fragmented across different packages and often value performances more than usability. For example the popular Deep Learning library Flux [@Innes:2018], while extremely performant and flexible, adopt some choices that for a beginner could appear odd, like avoiding the neural network object from the training process, or require all parameters explicitly defined.
In `BetaML` we made the choice to allow the user to experiment with the hyperparameters of the algorithms learning them one step at the time. So for most functions we provide reasonable default parameters that can be overridden when needed. For example, modelling, training and collecting predictions from an artificial neural network with one hidden layer can be as simple as:

```julia
using BetaML.Nn
mynn   = buildNetwork([DenseLayer(nIn,nHidden),
                       DenseLayer(nHidden,nOut)],
                       squaredCost)
train!(mynn,xtrain,ytrain)
ytrain_est = predict(mynn,xtrain)           
ytest_est  = predict(mynn,xtest)
```

While much better results can be obtained (in general) by scaling the variables and/or tuning their activation functions or the optimisation algorithm, this code snippet already runs the model using common practices like random mini-batches.

Still BetaML offers a fair level of flexibility. As we didn't aim for heavy optimisation, we were able to keep the API (Application Programming Interface) both beginner-friendly and flexible.
For example, one can implement its own neural network layer, optimisation algorithm, or mixture model, define the initial conditions in almost all the stochastic algorithms or specify its own distance metric in the clustering algorithms.
To help beginners, many parameters and functions have pretty longer but more explicit names than usual. For example the Dense layer is a `DenseLayer`, the RBF kernel is `radialKernel`, etc.

A few packages try to provide a common framework of the ML ecosystem of Julia packages, notably MLJ.jl [@Blaom:2019], build up on existing ML specialised packages. While they avoid the problem of "reinventing the wheel", the wrapping level unintentionally introduce some complications for the end-user, like the need to load the models and, for MLJ, to learn MLJ-specific concepts as _model_ or _machine_. Also it make difficult for the user to trace back to the code performing the computations and modify for his own needs.

We chose instead to bundle the main ML algorithms directly within the package. This offers a complementary approach that we believe is more beginner-friendly.

<!--
Parameters Start gradually

Simplicity not like Flux that rather than passing the neural network object to the train function in order to flexibility and optimise pass ....



Of course you can get much better results (in general) by scaling the variables, adding further layer(s) and/or tuning their activation functions or the optimisation algorithm (have a look at the notebooks 1 or at the documentation for that), but the idea is that while we can offer a fair level of flexibility (you can choose or define your own activation function, easy define your own layers, choose weight initialisation, choose or implement the optimisation algorithm and its parameters, choose the training parameters - epochs, batchsize,…-, the callback function to get informations during (long) training,…), still we try to keep it one step at the time. So for most stuff we provide default parameters that can be overridden when needed rather than pretend that the user already know and provide all the needed parameters.
-->

We believe that the BetaML flexibility and simplicity, together with the efficiency and usability of a Just in Time compiled language like Julia and the convenience to have several ML algorithms and data-science utilities all in the same package,
will support the needs of
<!-- can address significantly better the needs of  -->
students and researchers that, contrary to industrial practitioners, often don't need to work  with very large datasets that don't fit in memory or algorithms that require distributed computation.

<!--
Other approaches

MLJ: @Blaom:2019
Flux: @Innes:2018
Knet: @Yuret:2016

-->

<!-- Students and reserachers may find appealling to perform their research or as a platform to develop their own algorithms -->


# Package organisation

The BetaML toolkit is currently composed of 4 modules: `Utils` provides common data-science utility functions to be used in the other modules, `Perceptron` supplies linear and non-linear classifiers based on the classical Perceptron algorithm, `Nn` allows implementing and training artificial neural networks, and `Clustering` includes several clustering algorithms and missing value attribution / collaborative filtering algorithms based on clustering.

`Perceptron`, `Nn` and `Clustering` all import and re-export the `Utils` function, so the final users normally doesn't need to deal with `Utils`, but just with the module of interest.



## The `Utils` module

The `Utils` module is intended to provide functionalities that are either: (a) used in other modules but are not strictly part of that specific module's logic (for example activation functions would be most likely used in neural networks, but could be of more general usage); (b) general methods that are used alongside the ML algorithms implemented in the other modules, e.g. to improve their predictions capabilities or (c) general methods to assess the goodness of fits of ML algorithms.

Concerning the fist category `Utils` provides "classical" activation functions (and their respective derivatives) like `relu`, `sigmoid`, `softmax`, but also more recent implementations like elu [@Clevert:2015], celu [@Barron:2017], plu [@Nicolae:2018], softplus [@Glorot:2011] and mish [@Misra:2019]. Kernel functions (`radialKernel` - aka "KBF", `polynomialKernel`), distance metrics (`l1_distance` - aka "Manhattan", `l2_distance`, `l2²_distance`, `cosine_distance`), and functions typically used to improve numerical stability (`lse`) are also provided with the intention to be available in the different ML algorithms.

Often ML algorithms work better if the data is normalised or dimensions are reduced to those explaining the greatest extent of data variability. This is the purpose of the functions `scale` and `pca` respectively. `scale` scales the data to $\mu=0$ and $\sigma=1$, optionally skipping dimensions that don't need to be normalised. The related function `getScaleFactors` saves the scaling factors so that inverse scaling (typically for the predictions of the ML algorithm) can be applied. `pca` performs Principal Component Analysis, where the user can specify the wanted dimensions or the maximum approximation error that he is willing to accept either _ex-ante_ or _ex-post_, after having analysed the distribution of the explained variance by number of dimensions. Other "general support" functions provided are `oneHotEncoder` and `batch`.

Concerning the last category, several functions are provided to assess the goodness of fit of a single datapoint or of the whole dataset, whether the output of the ML algorithm is in $R^n$ or categorical. Notably, `accuracy` provides categorical accuracy given a probabilistic prediction (PMF) of a datapoint.
<!--, with a parameter `tol` to determine the tollerance of the prediction, i.e. if considering "correct" only a prediction where the value with highest probability is the true value (`tol` = 1), or consider instead the set of `tol` maximum values.-->
finally, the Bayesian information criterion `bic` and Akaike information criterion `aic` functions can be used for regularisation.

## The `Perceptron` module

It provides the classical Perceptron linear classifier, a _kernelised_ version of it and "Pegasos" [@Shalev-Shwartz:2011], a gradient-descent based implementation.

The basic Perceptron classifier is implemented in the `perceptron` function, where the user can provide the initial weights and retrieve both the final and the average parameters of the classifiers. In `kernelPerceptron` the user can either pass one of the kernel implemented in `Utils` or implement its own kernel function. `pegasos` performs the classification using a basic stochastic descent method^[We plan to generalise the Pegasos algorithm to use the optimisation algorithms implemented for neural networks.]. Finally `predict` predicts the binary label given the feature vector and the linear coefficients or the error distribution as obtained by the kernel Perceptron algorithm.

## The `Nn` module

Artificial neural networks can be implemented using the functions provided by the `Nn` module.
Currently only feed-forward networks for regression or classification tasks are fully provided, but more complex layers (convolutional, pooling, recursive,...) can be eventually defined and implemented directly by the user.
The instantiation of the layers required by the network can be done indeed either using one of the layer provided (`DenseLayer`, `DenseNoBiasLayer` or `VectorFunctionLayer`, the latter one being a parameterless layer whose activation function, like `softMax`, is applied to the ensemble of the neurons rather than individually on each of them) or by creating a user-defined layer by subclassing the `Layer` type and implementing the functions `forward`, `backward`, `getParams`, `getGradient`, `setParams` and `size`.

While in the provided layers the computation of the derivatives for `backward` and `getParams` is coded manually^[For the derivatives of the activation function the user can (a) provide one of the derivative functions defined in `Utils`, (b) implement it by himself, or (c) just leave the library use automatic differentiation (using Zygote) to compute it.], for complex user-defined layers the two functions can benefit of automatic differentiation packages like `Zygote`[@Innes:2018b], eventually wrapped in the function `autoJacobian` defined in `Utils`.

Once the layers are defined, the neural network is modelled by setting the layers in an array, giving the network a cost function (default to ) and a name. The `show` function can be employ to print the structure of the network.

The training of the model is done with the highly parametrisable `train!` function. In a similar way than for the definition of the layers, one can use for training one of the "standard" optimisation algorithms provided (`SGD` and `ADAM`, @Kingma:2014), either using their default values or by fine-tuning their parameters, or by defining the optimisation algorithm by subclassing the  `OptimisationAlgorithm` class and implementing the `singleUpdate!` and eventually `initOptAlg!` methods. Note that the `singleUpdate!` function provides the algorithm with quite a large set of information from the training process, allowing a wide class of optimisation algorithms to be implemented.


# `Clustering` module

Both the classical `kmeans` and `kmedoids` algorithms are provided (with the difference being that the clusters "representatives" can be in any $R^n$ point in `kmeans`, while are restricted to be one of the data point in `kmedoids`), where different measure metrics can be provided (either those defined in `Utils` or user-provided ones) as well as different initialisation strategies (`random`, `grid`, `shuffle` - randomly within the available points, `given`).

Alongside these "hard clustering" algorithms, the `Clustering` module provides `em`, an implementation of the Expectation-Maximisation algorithm to estimate a generative mixture model, with variance-free and variance-constrained Gaussian mixtures already provided (and again, one can write his own mixture by subclassing `Mixture` and implementing `initMixtures!`, `lpdf`, `updateParameters!` and `npar`) and with `kmeans` or `grid` initialisation strategy supported.

Notably the `em` algorithm would accept an input data whose values are missing in one or all dimensions (and in the former case learning would use only the available dimensions).

This, together with the probabilistic assignment nature of the em algorithm, allows it to be used as base for missing values assignment or even collaborative filtering/recommandation systems in the `predictMissing` function.

<!--

# Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for fidgit.

For a quick reference, the following citation commands can be used:
- `arobase author:2001`  ->  "Author et al. (2001)"
- `[arobase author:2001]` -> "(Author et al., 2001)"
- `[arobase author1:2001; arobase author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Fenced code blocks are rendered with syntax highlighting:

```python
for n in range(10):
    yield f(n)
```
-->

# Acknowledgements

This work was supported by: (i) the French National Research Agency through the Laboratory of Excellence ARBRE, part of the "Investissements d'Avenir" Program (ANR 11 – LABX-0002-01), and the ORACLE project (ANR-10-CEPL-011); (ii) a grant overseen by Office National des Forêts through the Forêts pour Demain International Teaching and Research Chair.

# References
