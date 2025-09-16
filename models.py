import torch
import torch.nn as nn

class SimpleNeuraNetwork(nn.Module):
  def __init__(self):
    super().__init__()
    self.flatten = nn.Flatten()
    self.fc1 = nn.Sequential(
      nn.Linear(in_features= 3 * 32 *32, out_features= 256 ),
      nn.ReLU()
    )
    self.fc2 = nn.Sequential(
      nn.Linear(in_features= 256, out_features= 512 ),
      nn.ReLU()
    )
    self.fc3 = nn.Sequential(
      nn.Linear(in_features= 256, out_features= 1024 ),
      nn.ReLU()
    )
    

  def forward(self,x):
    x = self.flatten(x)
    x = self.fc1(x)
    x = self.fc2(x)
    x = self.fc3(x)
    
    return x

if __name__ == '__main__':
  model = SimpleNeuraNetwork()
  input_data = torch.rand(8, 3, 32, 32)
  result = model(input_data)
  print(result)