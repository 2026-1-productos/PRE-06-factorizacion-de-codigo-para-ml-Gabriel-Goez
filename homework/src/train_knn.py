from sklearn.neighbors import KNeighborsRegressor

from ._internals.run_experiment import run_experiment

run_experiment(estimator=KNeighborsRegressor(n_neighbors=5))
