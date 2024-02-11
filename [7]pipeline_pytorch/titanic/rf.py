import numpy as np
from sklearn.ensemble import RandomForestRegressor

class CustomRF(RandomForestRegressor):
  def __init__(
      self,
      n_estimators=100,
      *,
      criterion="squared_error",
      max_depth=None,
      min_samples_split=2,
      min_samples_leaf=1,
      min_weight_fraction_leaf=0.0,
      max_features=1.0,
      max_leaf_nodes=None,
      min_impurity_decrease=0.0,
      bootstrap=True,
      oob_score=False,
      n_jobs=None,
      random_state=None,
      verbose=0,
      warm_start=False,
      ccp_alpha=0.0,
      max_samples=None,
      threshold=0.5,
  ):
    super().__init__(
      n_estimators=n_estimators,
      criterion=criterion,
      max_depth=max_depth,
      min_samples_split=min_samples_split,
      min_samples_leaf=min_samples_leaf,
      min_weight_fraction_leaf=min_weight_fraction_leaf,
      max_features=max_features,
      max_leaf_nodes=max_leaf_nodes,
      min_impurity_decrease=min_impurity_decrease,
      bootstrap=bootstrap,
      oob_score=oob_score,
      n_jobs=n_jobs,
      random_state=random_state,
      verbose=verbose,
      warm_start=warm_start,
      ccp_alpha=ccp_alpha,
      max_samples=max_samples,
    )
    self.threshold = threshold
  
  def predict(self, *args, **kwargs):
    prd = super().predict(*args, **kwargs)
    return (prd > self.threshold).astype(np.float32)