import os
from torchvision import transforms

path = 'D:\\Datasets\\Computer Vision\\asl_alphabet\\asl_alphabet_train\\asl_alphabet_train'
num_classes = len(os.listdir(path))

img_size = 224
batch_size = 32
epochs = 2
transform = transforms.Compose([
    transforms.ToTensor()
])