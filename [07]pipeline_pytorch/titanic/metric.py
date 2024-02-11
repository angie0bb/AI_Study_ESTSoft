import numpy as np

def cm_to_metrics(cm:np.array) -> tuple:
  '''Confusion Matrix to Metrics

  Args:
      cm: confusion matrix of shape (2,2)
  '''
  (tn, fp), (fn, tp) = cm
  accuracy = (tp+tn)/(tp+tn+fn+fp)
  precision = tp/(tp+fp)
  recall = tp/(tp+fn)
  f1 = 2*(precision*recall)/(precision+recall)

  return (accuracy, precision, recall, f1)
