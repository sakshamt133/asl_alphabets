import train_data
import torch
import utils
from torchvision.models import resnet18

model = resnet18(pretrained=False)
model.fc.out_features = utils.num_classes
model = model.double()
opti = torch.optim.Adam(model.parameters(), lr=0.001)
lossFun = torch.nn.CrossEntropyLoss()

if __name__ == '__main__':
    for epoch in range(utils.epochs):
        for batch in train_data.train_batch:
            imgs, labels = batch
            output = model(imgs.double())
            loss = lossFun(output, labels)
            print(f'for epoch {epoch} loss is {loss}')
            loss.backward()
            opti.step()
            opti.zero_grad()
