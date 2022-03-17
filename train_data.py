from dataset import Alphabets
import utils
from torch.utils.data import DataLoader

dataset = Alphabets(utils.path, utils.transform)

train_batch = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=8)
