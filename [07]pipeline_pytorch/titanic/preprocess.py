import pandas as pd
import numpy as np

def get_X(df:pd.DataFrame, features:iter=["Pclass", "Sex", "SibSp", "Parch"]):
  '''Make feature vectors from a DataFrame.

  Args:
      df: DataFrame
      features: selected columns
  '''
  # from https://www.kaggle.com/code/alexisbcook/titanic-tutorial
  return pd.get_dummies(df[features]).to_numpy(dtype=np.float32)

def get_y(df:pd.DataFrame):
  '''Make the target from a DataFrame.

  Args:
      df: DataFrame
  '''
  return df.Survived.to_numpy(dtype=np.float32)