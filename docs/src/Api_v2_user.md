# BetaML Api v2

!!! info Compatibility
    The API described below is _experimental_ in BetaML 0.7 and will be default in BetaML 0.8, when at the same time the old API will be deprecated. In 0.7 not all BetaML models may have this new API implemented.


The following API is designed to further simply the usage of the various ML models provided by BetaML introducing a common workflow. This is the _user_ documentation. Refer to the developer documentation to read how the API is implemented. 

## Supervised models

This refer to models designed to _learn_ a relation between some features (often noted with X) and labels (often noted with Y) in order to predict the label of new data given the observed features alone. Perceptron, decision trees or neural networks are common examples.

### Model constructor

The first step is to build the model constructor by passing (using keyword arguments) the agorithm hyperparameters and various options (cache results flag, debug levels, random number generators, ...)


```
m = ModelName(⋅)
```

### Training of the model

The second step is to _fit_ (aka _train_) the model:
```
fit!(m,X,y)
```
For online algorithms, i.e. models that support updating of the learned parameters with new data, `fit!` can be repeated as new data arrive, altought not all algorithms guarantee that training each record at the time is equivalent to train all the records at once. In some algorithms the "old training" could be used as initial conditions, without consideration if these has been achieved with hundread or millions of records, and the new data we use for training become much more important than the old one for the determination of the learned parameters.
As a naming convention, while we would have preferred the name "train" for this funtion, as it makes explicit that we are here changing the learned parameters of the model, it seems that most other ML libraries call this step "fit", so we stuck with it. 

### Prediction

Trained models can be used to predict `y` given new `X`:

```
ŷ = predict(m,X)
```

As for convenience, if the model has been trained while having the `cache` option set on `true` (by default) the `ŷ` of the last training is retained in the  model object and it can be retrieved simply with `predict(m)`. This is particularly useful for unsupervised and transformer models (next section)

### Other functions

Models can be resetted to lose the learned information with `reset!(m)` and training information (other than the algorithm learned parameters) can be retrieved with `info(m)`.

## Unupervised and transformed models

This relate to models that learn a "structure" from the data itself (without any label attached from which to learn) and report either some new information using this learned structure (e.g. a cluster class) or directly process a transformation of the data itself, like `PCA` or missing imputers.

The main differences with supervised models is that the `fit!` function takes only the features and that the `predict` one take only the (trained) model as argument - models that do generalise to new data can accept also a `predict(m,newX)` version that uses what has been learn in `fit!`:

```
m = Model()
fit!(m,X)
result = predict(m) # or result = predict(m, newX)
```

As for supervised models, the `predict(m)` version relies on the `cache` option of the model being `true` at training time.
Some model allow an inverse transformation, that using the parameters learned ar trainign time (e.g. the scale factors) performs an inverse tranformation of new data to the space of the training data (e.g. the unscaled space).
Use the `inv=true` keyword for that, e.g. `predict(mod,xnew,inv=true)`.
