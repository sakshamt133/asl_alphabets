from torch.utils.data import Dataset
import os
import cv2 as cv


class Alphabets(Dataset):
    def __init__(self, path, transform):
        super(Alphabets, self).__init__()
        self.path = path
        self.transform = transform
        self.labels = os.listdir(path)
        self.data = []
        self.make_data()

    def make_data(self):
        for label in self.labels:
            for label_path in os.listdir(os.path.join(self.path, label)):
                temp = []
                small_path = os.path.join(self.path, label)
                complete_path = os.path.join(small_path, label_path)

                img = cv.imread(complete_path)
                img = cv.resize(img, (200, 200))
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                img = self.transform(img)

                temp.append(img)
                temp.append(label)
                self.data.append(temp)

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)
