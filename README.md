# Beta Machine Learning Toolkit

<img src="assets/bmlt_logo.png" width="300"/>

The **Beta** (or _Basic_ if your prefer) **Machine Learning Toolkit** is a repository with several basic Machine Learning algorithms, started from implementing in the Julia language the concepts taught in the [MITX 6.86x - Machine Learning with Python: from Linear Models to Deep Learning](https://www.edx.org/course/machine-learning-with-python-from-linear-models-to) course.

Theoretical notes describing most of these algorithms are at the companion repository https://github.com/sylvaticus/MITx_6.86x.

This stuff most likely has value only didactically, as the approaches are the "vanilla" ones, i.e. the simplest possible ones, and GPU is not supported here.
For "serious" machine learning work in Julia I suggest to use either [Flux](https://fluxml.ai/) or [Knet](https://github.com/denizyuret/Knet.jl).

That said, Julia is a relatively fast language and most hard job is done in matrix operations whose underlying libraries are multithreaded, so it is reasonably fast for small explanatory tasks.

**You can run the code by yourself (folder "notebooks") in myBinder, a temporary public online computational environment:
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/sylvaticus/bmlt.jl/master)**
Note: every first time after a commit is made on this repository it takes a (very) long time to load such environment for the (unlucky) user that triggers the process, as the temporary environment need to be created. Subsequent users should find a cached version of the computational environment and the load time should be much smaller.

By the way, if you are looking for an introductory book on Julia, have a look on my "[Julia Quick Syntax Reference](https://www.julia-book.com/)"(Apress,2019).

## Documentation

Documentation for most functions can be retrieved using the inline Julia help system (just press the question mark and then on the special prompt type the function name).

A proper documentation is work in progress.


## Acknowledgements

The development of this package at the _Bureau d'Economie Théorique et Appliquée_ (BETA, Nancy) was supported by the French National Research Agency through the [Laboratory of Excellence ARBRE](http://mycor.nancy.inra.fr/ARBRE/), a part of the “Investissements d'Avenir” Program (ANR 11 – LABX-0002-01).
