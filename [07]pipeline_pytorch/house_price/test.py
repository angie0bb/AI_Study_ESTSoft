import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchmetrics

from sklearn.model_selection import StratifiedKFold
from nn import ANN
from torch.utils.data import TensorDataset
from tqdm.auto import tqdm
from tqdm.auto import trange
from train import train_one_epoch
from validate import validate_one_epoch
import pandas as pd
import numpy as np
# apply parameters  

def test(cfg, model_path: str):
    """test: run test set 
    cfg: config.py
    """
    import numpy as np
    import pandas as pd
    from torch.utils.data.dataset import TensorDataset
    from nn import ANN
    from tqdm.auto import trange

    train_params = cfg.get("train_params")
    device = torch.device(train_params.get("device"))

    # get preprocessed X, y csv files
    files = cfg.get("files")
    X = torch.tensor(pd.read_csv(files.get('test_X_csv'), index_col=0).to_numpy(dtype=np.float32))
    y = torch.tensor(pd.read_csv(files.get('test_y_csv'), index_col=0).to_numpy(dtype=np.float32))

    # Model
    Model = cfg.get("model")
    model_params = cfg.get("model_params")
    model_params["input_dim"] = X.shape[-1] # 여기에 자동으로 input_dim 넣어준다
    model = Model(**model_params).to(device)
    model.load_state_dict(torch.load(model_path))  ## train 시켜두었던 모델 꺼내기

    model.eval()  # put to evaluation mode
    metric = train_params.get("metric")
    with torch.inference_mode():
        pred = model.forward(X)
        tst_rmse = metric(pred, y).item()
        df_metric = pd.DataFrame([tst_rmse], index = ["RMSE"])
        pred_np = pd.DataFrame(pred.detach().numpy())

    return df_metric, pred_np

        # plt.figure(figsize=(8, 6))
        # sns.heatmap(df_metrics, annot=True, fmt=".2f", cmap="Blues")
        # plt.title("Model Performance")
        # plt.tight_layout()
        # plt.show()


def get_args_parser(add_help=True):
  import argparse
  
  parser = argparse.ArgumentParser(description="Test", add_help=add_help)
  parser.add_argument("-c", "--config", default="./config.py", type=str, help="configuration file")

  return parser

if __name__ == "__main__":
  args = get_args_parser().parse_args()
  exec(open(args.config).read())
  res = test(config, model_path="model.pth")
  print(res)
  # results = kfold_validate(config)
  # print(results)