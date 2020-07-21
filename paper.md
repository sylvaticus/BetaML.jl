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

# Summary

We implemented a serie of _machine learning_ algorithms in a package for the Julia language.
Currently, algorithms are in the area of classification (perceptron, kernel perceptron, pegasus), neural networks and clustering (kmeans, kmenoids, EM, missing values attribution). Development of these algorithms follows the theoretical notes of the MOOC class "Machine Learning with Python: from Linear Models to Deep Learning" from MITx/edX.






# Statement of need

# Provided features

The toolkit is currently composed by 4 modules: `Utils`, providing common data-science utility functions to be used in the other modules, `Perceptron` providing linear and non-linear classifiers, `Nn` to implement and train artificial neural networks, and `Clustering`, with several clustering algorithms and missing value attribution/ collaborative filtering based on clustering.
`Perceptron`, `Nn` and `Clustering` all import and re-export the Utile function, so the final users normally doesn't need to deal with `Utils`, but just with the module of interest.

## The `Utils` module

Function to preprocess the data
`scale and `pca`



## The `Perceptron` module

It provides the classical `Perceptron` binomial classifier, a kernelised version of it and `Perceptron`, a ...
The user can leave the default KBF kernel or provide its own function )that we don't check for correctness)


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
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Fenced code blocks are rendered with syntax highlighting:

```python
for n in range(10):
    yield f(n)
```

# Acknowledgements

This work was supported by: (i) the French National Research Agency through the Laboratory of Excellence ARBRE, part of the “Investissements d'Avenir” Program (ANR 11 – LABX-0002-01), and the ORACLE project (ANR-10-CEPL-011); (ii) a grant overseen by Office National des Forêts through the Forêts pour Demain International Teaching and Research Chair.

# References
