
# Examples


## Supervised learning

### Regression

#### Estimating the bike sharing demand

The task is to estimate the influence of several variables (like the weather, the season, the day of the week..) on the demand of shared bicycles, so that the authority in charge of the service can organise the service in the best way.

Data origin:
- original full dataset (by hour, not used here): https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset
- simplified dataset (by day, with some simple scaling): https://www.hds.utc.fr/~tdenoeux/dokuwiki/en/aec
  - description: https://www.hds.utc.fr/~tdenoeux/dokuwiki/_media/en/exam_2019_ace_.pdf
  - data: https://www.hds.utc.fr/~tdenoeux/dokuwiki/_media/en/bike_sharing_day.csv.zip

Note that even if we are estimating a time serie, we are not using here a recurrent neural network as we assume the temporal dependence to be negligible (i.e. $Y_t = f(X_t)$ alone).





### Classification


## Unsupervised lerarning


# Notebooks
The following notebooks provide runnable examples of the package functionality:

- Pegasus classifiers: [[Static notebook](https://github.com/sylvaticus/BetaML.jl/blob/master/notebooks/Perceptron.ipynb)] - [[myBinder](https://mybinder.org/v2/gh/sylvaticus/BetaML.jl/master?filepath=notebooks%2FPerceptron.ipynb)]
- Decision Trees and Random Forest regression on Bike sharing demand forecast (daily data): [[Static notebook](https://github.com/sylvaticus/BetaML.jl/blob/master/notebooks/DecisionTrees%20-%20Bike%20sharing%20demand%20forecast%20(daily%20db).ipynb)] - [[myBinder](https://mybinder.org/v2/gh/sylvaticus/BetaML.jl/master?filepath=notebooks%2FDecisionTrees%20-%20Bike%20sharing%20demand%20forecast%20(daily%20db).ipynb)]
- Neural Networks: [[Static notebook](https://github.com/sylvaticus/BetaML.jl/blob/master/notebooks/Nn.ipynb)] - [[myBinder](https://mybinder.org/v2/gh/sylvaticus/BetaML.jl/master?filepath=notebooks%2FNn.ipynb)]
  - Bike sharing demand forecast (daily data): [[Static notebook](https://github.com/sylvaticus/BetaML.jl/blob/master/notebooks/NN%20-%20Bike%20sharing%20demand%20forecast%20(daily%20db).ipynb)] - [[myBinder](https://mybinder.org/v2/gh/sylvaticus/BetaML.jl/master?filepath=notebooks%2FNN%20-%20Bike%20sharing%20demand%20forecast%20(daily%20db).ipynb)]
- Clustering: [[Static notebook](https://github.com/sylvaticus/BetaML.jl/blob/master/notebooks/Clustering.ipynb)] - [[myBinder](https://mybinder.org/v2/gh/sylvaticus/BetaML.jl/master?filepath=notebooks%2FClustering.ipynb)]


Note: the live, runnable computational environment is a temporary new copy made at each connection. The first time after a commit is done on this repository a new environment has to be set (instead of just being copied), and the server may take several minutes.

This is only if you are the unlucky user triggering the rebuild of the environment after the commit.
