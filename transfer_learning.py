import torch
from torchvision.models import resnet50, ResNet50_Weights

# model = resnet50(weights=ResNet50_Weights.DEFAULT)

# image = torch.rand(2, 3, 224, 224)

# output = model(image)

# print(output.shape)


class MyResNet(nn.Module):
  def __init__(self, num_classes=10):
    super().__init__()
    self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
    self.fc1 = nn.Linear()