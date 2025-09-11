import

class AnimalDataset(Dataset):
    def __init__(self, root, train=True):
        self.root = root
        if train:
            mode = 'train'
        else:
            mode = 'test'
        self.root =os.path.join(self.root, mode)

    def __len__(self):
        pass

    def __getitem__(self, idx):
        image = None
        label = None
        return image, label



if __name__ == '__main__':
    # dataset = AnimalDataset(root="data/animals", train=True)
    # image, label = dataset.__getitem__(0)
    # print("Image shape:", image.shape)
    # print("Label:", label)
    root = "data/animals"
    datae-set = AnimalDataset(root=root, train=True)
    print(dataset.root)