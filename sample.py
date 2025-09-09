from torchvision.datasets import ImageFolder

dataset = ImageFolder(root='path/to/your/images')

image, label = dataset.__getitem__(0)

print(label)

image.show()

