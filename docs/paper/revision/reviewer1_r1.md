Okay, here's an **update** of my review from the [pre-review thread](https://github.com/openjournals/joss-reviews/issues/2512)

## What the package provides

The package under review provides pure-julia implementations of two
tree-based models, three clustering models, a perceptron model (with 3
variations) and a basic neural network model. In passing, it should be
noted that all or almost all of these algorithms have existing julia
implementations (e.g., DecisionTree.jl, Clustering.jl, Flux.jl). The package
is used in a course on Machine Learning but integration between the
package and the course is quite loose, as far as I could ascertain
(more on this below).

~~Apart from a library of loss functions, the package provides no
other tools.~~ In addition to the models the package provides a number
of loss functions, as well as activation functions for the neural
network models, and some tools to rescale data. I did not see tools to
automate resampling (such as cross-validation), hyper parameter
optimization, and no model composition (pipelining). The quality of
the model implementations looks good to me, although the author warns
us that "the code is not heavily optimized and GPU [for neural
networks] is not supported "

## Existing machine learning toolboxes in Julia

For context, consider the following multi-paradigm ML
toolboxes written in Julia which are relatively mature, by Julia standards:

package          | number of models | resampling  | hyper-parameter optimization | composition
-----------------|------------------|-------------|------------------------------|-------------
[ScikitLearn.jl](https://github.com/cstjean/ScikitLearn.jl)   | > 150            | yes         | yes                          | basic
[AutoMLPipeline.jl](https://github.com/IBM/AutoMLPipeline.jl)| > 100            | no          | no                           | medium
[MLJ.jl](https://joss.theoj.org/papers/10.21105/joss.02704)           | 151              | yes         | yes                          | advanced

In addition to these are several excellent and mature packages
dedicated to neural-networks, the most popular being the AD-driven
Flux.jl package. So far, these provide limited meta-functionality,
although MLJ now provides an interface to certain classes of Flux
models ([MLJFlux](https://github.com/alan-turing-institute/MLJFlux.jl)) and
ScikitLearn.jl provides interfaces to python neural network models
sufficient for small datasets and pedagogical use.

Disclaimer: I am a designer/contributor to MLJ.

**According to the [JOSS requirements](https://joss.theoj.org/about),
Submissions should "Have an obvious research application."**  In its
current state of maturity, BetaML is not a serious competitor to the
frameworks above, for contributing directly to research. However, the
author argues that it has pedagogical advantages over existing tools.

## Value as pedagogical tool

I don't think there are many rigorous machine learning courses or
texts closely integrated with models and tools implemented in julia
and it would be useful to have more of these. ~~The degree of
integration in this case was difficult for me to ascertain because I
couldn't see how to access the course notes without formally
registering for the course (which is, however, free).~~ I was also
disappointed to find only one link from doc-strings to course
materials; from this "back door" to the course notes I could find no
reference back to the package, however. Perhaps there is better
integration in course exercises? I couldn't figure this out.

**edit** Okay, I see that I missed the link to the course notes, as
opposed to the course itself. However the notes make only references
to python code and so do not appear to be directly integrated with the
package BetaML.

The remaining argument for BetaML's pedagogical value rests on a
number of perceived drawbacks of existing toolboxes, for the
beginner. Quoting from the JOSS manuscript:

1. "For example the popular Deep Learning library Flux (Mike Innes,
   2018), while extremely performant and flexible, adopts some
   designing choices that for a beginner could appear odd, for example
   avoiding the neural network object from the training process, or
   requiring all parameters to be explicitly defined. In BetaML we
   made the choice to allow the user to experiment with the
   hyperparameters of the algorithms learning them one step at the
   time. Hence for most functions we provide reasonable default
   parameters that can be overridden when needed."

2. "To help beginners, many parameters and functions have pretty
   longer but more explicit names than usual. For example the Dense
   layer is a DenseLayer, the RBF kernel is radialKernel, etc."

3. "While avoiding the problem of “reinventing the wheel”, the
   wrapping level unin- tentionally introduces some complications for
   the end-user, like the need to load the models and learn
   MLJ-specific concepts as model or machine.  We chose instead to
   bundle the main ML algorithms directly within the package. This
   offers a complementary approach that we feel is more
   beginner-friendly."

Let me respond to these:

1. These cricitism only apply to dedicated neural network
   packages, such as Flux.jl; all of the toolboxes listed
   above provide default hyper parameters for every model. In the case
   of neural networks, user-friendly interaction close to the kind
   sought here is available either by using the MLJFlux.jl models
   (available directly through MLJ) or by using the python models
   provided through ScikitLearn.jl.

2. Yes, shorter names are obstacles for the beginner but hardly
   insurmountable. For example, one could provide a cheat sheet
   summarizing the models and other functionality needed for the
   machine learning course (and omitting all the rest).

3. Yes, not needing to load in model code is slightly more
   friendly. On the other hand, in MLJ for example, one can load and
   instantiate a model with a single macro. So the main complication
   is having to ensure relevant libraries are in your environment. But
   this could be solved easily with a `BeginnerPackage` which curates
   all the necessary dependencies. I am not convinced beginners should
   find the idea of separating hyper-parameters and learned parameters
   (the "machines" in MLJ) that daunting. I suggest the author's
   criticism may have more to do with their lack of familiarity than a
   difficulty for newcomers, who do not have the same preconceptions
   from using other frameworks. In any case, the point is moot, as one
   can interact with MLJ models directly via a "model" interface and
   ignore machines. To see this, I have
   [translated](https://github.com/ablaom/ForBetaMLReview) part of a
   BetaML notebook into MLJ syntax. There's hardly any difference - if
   anything the presentation is simpler (less hassle when splitting
   data horizontally and vertically).

In summary, while existing toolboxes might present a course instructor
with a few challenges, these are hardly game-changers. The advantages of
introducing a student to a powerful, mature, professional toolbox *ab*
*initio* far outweigh any drawbacks, in my view.

## Conclusions

To meet the requirements of JOSS, I think either: (i) The BetaML
package needs to demonstrate tighter integration with ~~easily
accessible~~ course materials; or (ii) BetaML needs very substantial
enhancements to make it competitive with existing toolboxes.

Frankly, a believe a greater service to the Julia open-source software
community would be to integrate the author's course materials with one
of the mature ML toolboxes. In the case of MLJ, I would be more than
happy to provide guidance for such a project.

---

## Sundry comments

I didn't have too much trouble installing the package or running the
demos, except when running a notebook on top of an existing Julia
environment (see commment below).

- **added** The repository states quite clearly that the primary
  purpose of the package is dilectic (for teaching purposes). If this
  is true, the paper should state this clearly in the "Summary" (not
  just that it was developed in response to the course).

- **added** The authors should reference for comparison the toolboxes
  ScitkitLearn.jl and AutoMLPipeline.jl

- The README.md should provide links to the toolboxes listed in
  the table above, for the student who "graduates" from BetaML.

- Some or most intended users will be new to Julia, so I suggest
  including with the installation instructions something about how to
  set up a julia environment that includes BetaML. Something like
  [this](https://alan-turing-institute.github.io/MLJ.jl/dev/#Installation-1), for example.

- I found it weird that the front-facing demo is an *unsupervised*
  model. A more "Hello World" example might be to train a Decision
  Tree.

- The way users load the built-in datasets seems pretty awkward. Maybe
  just define some functions to do this? E.g.,
  `load_bike_sharing()`. Might be instructive to have examples where
  data is pulled in using `RDatasets`, `UrlDownload` or similar?

- A cheat-sheet summarizing the model fitting functions and the loss
  functions would be helpful. Or you could have functions `models()` and
  `loss_functions()` that list these.

- I found it pretty annoying to split data by hand the way this is
  done in the notebooks and even beginners might find this
  annoying. One utility function here would go a long way to making
  life easier here (something like the `partition` function in the
  MLJ, which you are welcome to lift).

- The notebooks are not portable as they do not come with a
  Manifest.toml. One suggestion on how to handle this is
  [here](https://github.com/ablaom/ForBetaMLReview/blob/main/bike_sharing.ipynb)
  but you should add a comment in the notebook explaining that the
  notebook is only valid if it is accompanied by the Manifest.toml. I
  think an even better solution is provided by InstantiateFromUrl.jl
  but I haven't tried this yet.

- The name `em` for the expectation-maximization clustering algorithm
  is very terse, and likely to conflict with a user variable.  I admit, I had
  to dig up the doc-string to find out what it was.

_Originally posted by @ablaom in https://github.com/openjournals/joss-reviews/issues/2849#issuecomment-730694698_
