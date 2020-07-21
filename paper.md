---
title: 'BetaML: The Beta Machine Learning Toolkit, a repository of Machine Learning algorithms in Julia'
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
Currently, algorithms are in the area of classification (perceptron, kernel perceptron, pegasos), neural networks and clustering (kmeans, kmenoids, EM, missing values attribution). Development of these algorithms started following the theoretical notes of the MOOC class "Machine Learning with Python: from Linear Models to Deep Learning" from MITx/edX.

This paper presents the general approach of the package and gives an overview of its organisation. We refer the reader to the [package documentation](https://sylvaticus.github.io/BetaML.jl/stable) for instructions on how to use the various algorithms provided or to the MOOC notes available on GitHub for their mathematical backgrounds.


# Statement of need

Parameters Start gradually

Simplicity not like Flux that rather than passing the neural network object to the train function in order to flexibility and optimise pass ....


In BetaML modelling, training and collecting predictions from an artificial neural network with one hidden layer can be as simple as:

```julia
mynn   = buildNetwork([DenseLayer(nIn,nHidden),DenseLayer(nHidden,nOut)],squaredCost)
train!(mynn,xtrain,ytrain)
ytrain_est = predict(mynn,xtrain)           
ytest_est  = predict(mynn,xtest)
```
Of course you can get much better results (in general) by scaling the variables, adding further layer(s) and/or tuning their activation functions or the optimisation algorithm (have a look at the notebooks 1 or at the documentation for that), but the idea is that while we can offer a fair level of flexibility (you can choose or define your own activation function, easy define your own layers, choose weight initialisation, choose or implement the optimisation algorithm and its parameters, choose the training parameters - epochs, batchsize,…-, the callback function to get informations during (long) training,…), still we try to keep it one step at the time. So for most stuff we provide default parameters that can be overridden when needed rather than pretend that the user already know and provide all the needed parameters.

We believe that the BetaML flexibility and simplicity, together with the efficiency and usability of a Just in Time compiled language like Julia and the convenience to have several ML algorithms and data-science utilities all in the same package,
will support the needs of
<!-- can address significantly better the needs of  -->
students and researchers that, contrary to industrial practitioners, often don't need to work  with very large datasets that don't fit in memory or algorithms that require distributed computation.


# Package organisation

The BetaML toolkit is currently composed of 4 modules: `Utils` provides common data-science utility functions to be used in the other modules, `Perceptron` supplies linear and non-linear classifiers, `Nn` allows implementing and training artificial neural networks, and `Clustering` includes several clustering algorithms and missing value attribution / collaborative filtering algorithms based on clustering.

`Perceptron`, `Nn` and `Clustering` all import and re-export the `Utils` function, so the final users normally doesn't need to deal with `Utils`, but just with the module of interest.



## The `Utils` module

The `Utils` module is intended to provide functionalities that are either: (a) used in other modules but are not strictly part of that specific module's logic (for example activation functions would be most likely used in neural networks, but could be of more general usage); (b) general methods that are used alongside the ML algorithms implemented in the other modules, e.g. to improve their predictions capabilities or (c) general methods to assess the goodness of fits of ML algorithms.+
Concerning the fist category `Utils` provides "classical" activation functions (and their respective derivatives) like `relu`, `sigmoid`, `softmax`, but also more recent implementations like elu [TODO cite], celu[TODO cite], plu[TODO cite], softplus[TODO cite] and mish[TODO cite]. Kernel functions (`radialKernel` - aka `KBF`, `polynomialKernel`), distance metrics (`l1_distance` - aka "Manhattan", `l2_distance`, `l2²_distance`, `cosine_distance`), and functions typically used to improve numerical stability (`lse`) are also provided with the intention to be available in the different ML algorithms.+
Often ML algorithms work better if the data is normalised or dimensions are reduced to those explaining the greatest extent of data variability. This is the purpose of the functions `scale` and `pca` respectively. `scale` scales the data to $\mu=0$ and $\sigma=1$, optionally skipping dimensions that don't need to be normalised. The related function `getScaleFactors` save the scaling factors so that inverse scaling (typically for the predictions of the ML algorithm) can be applied. `pca` perform Principal Component Analysis, where the user can specify the wanted dimensions or the maximum approximation error that he is willing to accept either ex-ante or ex-post, after having analysed the distribution of the explained variance by number of dimensions. Other "general support" functions provided are `oneHotEncoder` and `batch`.+
Concerning the last category, several functions are provided to assess the goodness of fit of a single datapoint or the whole dataset, whether the output of the ML algorithm is in $R^n$ or categorical. Notably, `accuracy` provides categorical accuracy given a probabilistic prediction (PMF) of a datapoint, with a parameter `tol` to determine the tollerance of the prediction, i.e. if considering "correct" only a prediction where the value with highest probability is the true value (`tol` = 1), or consider instead the set of `tol` maximum values.


## The `Perceptron` module

It provides the classical Perceptron linear classifier, a _kernelised_ version of it and "Pegasos" [@Shalev-Shwartz:2011], a gradient-descent based implementation.

The basic Perceptron classifier is implemented in the `perceptron` function, where the user can provide the initial weights and retrieve both the final and the average parameters of the classifiers. In `kernelPerceptron` the user can either pass one of the kernel implemented in `Utils` or implement its own kernel function. `pegasos` performs the classification using a basic stochastic descent method^[We plan to generalise the `pegasus` algorithm to use the optimisation algorithms implemented for neural networks.]. Finally `predict` predicts the binary label given the feature vector and the linear coefficients or the error distribution as obtained by the kernel Perceptron algorithm.

## The `Nn` module

Artificial neural networks can be implemented using the functions provided by the `Nn` module.
Currently only feed-forward networks for regression or classification tasks are fully provided, but more complex layers (convolutional, pooling, recursive,...) can be eventually defined and implemented directly by the user.
The instantiation of the layers required by the network can be done indeed either using one of the layer provided (`DenseLayer`, `DenseNoBiasLayer` or `VectorFunctionLayer`, the latter one being a parameterless layer whose activation function, like `softMax`, is applied to the ensemble of the neurons rather than individually on each of them) or by creating a user-defined layer by subclassing the `Layer` type and implementing the functions `forward`, `backward`, `getParams`, `getGradient`, `setParams` and `size`.

While in the provided layers the computation of the derivatives for `backward` and `getParams` is coded manually^[For the derivatives of the activation function the user can provide one of the derivative functions defined in `Utils`, implement it by himself, or just leave the library use automatic differentiation (using Zygote) to compute it.], for complex user-defined layers the two functions can benefit of automatic differentiation packages like `Zigote`[TODO Citeme], eventually wrapped in the function `autoJacobian` defined in `Utils`.

Once the layers are defined, the neural network is modelled by setting the layers in an array, giving the network a cost function (default to ) and a name. The `show` function can be employ to print the structure of the network.

The training of the model is done with the highly parametrisable `train!` function. In a similar way than for the definition of the layers, one can use for training one of the "standard" optimisation algorithms provided (`SGD` and `ADAM`, TODO cite https://arxiv.org/pdf/1412.6980.pdf)), either using their default values or by fine-tuning their parameters, or by defining the optimisation algorithm by subclassing the  `OptimisationAlgorithm` class and implementing the `singleUpdate!` and eventually `initOptAlg!` methods. Note that the `singleUpdate!` function provides the algorithm with quite a large set of information from the training process, allowing a wide class of optimisation algorithms to be implemented.


# `Clustering` module


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
