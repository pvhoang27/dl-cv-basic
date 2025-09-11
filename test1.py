from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

if __name__ == '__main__':
    # Tải dữ liệu huấn luyện từ bộ dữ liệu CIFAR10
    training_data = CIFAR10(
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

    # Tạo DataLoader cho tập huấn luyện
    train_dataloader = DataLoader(
        dataset=training_data,
        batch_size=16,
        num_workers=8,
        drop_last=True,
        shuffle=True
    )

    # Tạo DataLoader cho tập kiểm thử
    test_dataloader = DataLoader(
        dataset=test_data,
        batch_size=4,
        num_workers=8,
        drop_last=False,
        shuffle=False
    )

    # Lặp qua một batch dữ liệu để kiểm tra
    for images, labels in train_dataloader:
        print(images.shape, labels.shape)
        # Vòng lặp sẽ chỉ chạy 1 lần rồi dừng lại
        break