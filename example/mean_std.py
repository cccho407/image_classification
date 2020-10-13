from torchvision import datasets, models, transforms

transform = transforms.Compose([
    transforms.ToTensor()
])
dataset = datasets.ImageFolder("example/train/")
mean = dataset.train_data.mean(axis=(0,1,2))
std = dataset.train_data.std(axis=(0,1,2))
mean = mean / 255
std = std / 255
print(mean)
print(std)