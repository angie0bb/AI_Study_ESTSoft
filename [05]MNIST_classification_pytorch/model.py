import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self): #  define the layers of the network
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512), # ax+b, 2의 배수
            nn.ReLU(), # 활성화함수의 역할: 이전 층에서 계산된 512 input을 ReLU로 통과시켜서 다음 레이어로 전달
            nn.Linear(512, 512), # 층을 더 늘린다고 장점만 있는 건 아님. 층이 깊어질수록 과적합이 발생할 확률이 커짐
            # 512 -> 노드 개수는 중간에 늘려도 되고,,,그때그떄 다름
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x): # specify how data will pass through the network
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader): # enumerate: 인덱스를 붙혀서 뽑아와준다
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()  # backpropagate -> 계산된 gradient 를 가지고 parameter를 업데이트하는 부분 (model.update)
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(x) # len(x)를 batch_size로 대신해도됨
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")