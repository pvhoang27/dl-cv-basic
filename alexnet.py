# alexnet_227.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    """
    AlexNet (2012) cho input 227x227.
    Sơ đồ kích thước:
      227x227x3
      -> Conv1 96 @ 11x11 s=4, no pad        -> 55x55x96
      -> MaxPool 3x3 s=2                      -> 27x27x96
      -> Conv2 256 @ 5x5 s=1, pad=2 (groups=2)-> 27x27x256
      -> MaxPool 3x3 s=2                      -> 13x13x256
      -> Conv3 384 @ 3x3 s=1, pad=1           -> 13x13x384
      -> Conv4 384 @ 3x3 s=1, pad=1 (g=2)     -> 13x13x384
      -> Conv5 256 @ 3x3 s=1, pad=1 (g=2)     -> 13x13x256
      -> MaxPool 3x3 s=2                      -> 6x6x256
      -> FC: 9216 -> 4096 -> 4096 -> num_classes
    """
    def __init__(self, num_classes: int = 1000, use_groups: bool = True, p_dropout: float = 0.5):
        super().__init__()
        g = 2 if use_groups else 1

        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),  # 227 -> 55
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                   # 55 -> 27

            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2, groups=g),  # 27 -> 27
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                   # 27 -> 13

            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),             # 13 -> 13
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1, groups=g),   # 13 -> 13
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1, groups=g),   # 13 -> 13
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=3, stride=2)                   # 13 -> 6
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=p_dropout),
            nn.Linear(256 * 6 * 6, 4096),  # 9216 -> 4096
            nn.ReLU(inplace=True),
            nn.Dropout(p=p_dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
            # Softmax NÊN dùng trong loss (CrossEntropyLoss đã gồm log-softmax),
            # không đặt Softmax ở đây để tránh sai khi huấn luyện.
        )

        self._init_weights()

    def _init_weights(self):
        # Khởi tạo theo He (Kaiming) cho conv và xavier cho linear
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  # B x 9216
        x = self.classifier(x)
        return x

# Demo chạy thử để kiểm tra kích thước
if __name__ == "__main__":
    model = AlexNet(num_classes=1000, use_groups=True)
    x = torch.randn(2, 3, 227, 227)  # batch=2
    y = model(x)
    print("Output shape:", y.shape)  # -> (2, 1000)
