from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

if __name__ == '__main__':
    # Tải dữ liệu huấn luyện từ bộ dữ liệu CIFAR10
    training_data = Animals(
        root="./cifar",
        train=True,
        download=True,
        transform=ToTensor()
    )

    # Tải dữ liệu kiểm thử
    test_data = CIFAR10(
        root="./cifar",
        train=False,
        download=True,
        transform=ToTensor()
    )