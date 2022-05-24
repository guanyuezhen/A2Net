from models.model import BaseNet
import torch
from time import time
from tqdm import tqdm

print("loaded all packages")

model = BaseNet(3, 1).cuda().eval()

# We use pretrained model as the model weights
model.load_state_dict(torch.load("./tools/results_LEVIR_iter_40000_lr_0.0001/best_model.pth"))

batch_size = 28
x = torch.randn(batch_size, 3, 256, 256).cuda()
y = torch.randn(batch_size, 3, 256, 256).cuda()

######################################
#### PyTorch Test ####################
######################################
for i in tqdm(range(50)):
    # warm up
    p = model(x, y)
    # p = p + 1

total_t = 0
for i in tqdm(range(100)):
    start = time()
    p = model(x, y)
    # p = p + 1  # replace torch.cuda.synchronize()
    total_t += time() - start

print("FPS", 100 / total_t * batch_size)
print('PyTorch batchsize=%d speed test completed' % batch_size)

torch.cuda.empty_cache()
