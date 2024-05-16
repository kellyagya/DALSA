import sys
import os
import torch
import yaml
sys.path.append('../../../../')
from models.DALSA import DALSA
from utils import util
from trainers.eval import meta_test


with open('../../../../config.yml', 'r') as f:
    temp = yaml.safe_load(f)
data_path = os.path.abspath(temp['data_path'])

test_path = os.path.join(data_path,'dogs/test_pre')
model_path = '/root/autodl-san/SANet/experiments/dogs/DALSA/Conv-4/model_Conv-4.pth'

gpu = 0
torch.cuda.set_device(gpu)

model = DALSA(resnet=False, noise=False)
model.cuda()
model.load_state_dict(torch.load(model_path,map_location=util.get_device_map(gpu)),strict=True)
model.eval()

with torch.no_grad():
    # for way in [30, 25, 20, 15, 10, 5]:
        way = 5
        for shot in [1, 5]:
            mean,interval = meta_test(data_path=test_path,
                                    model=model,
                                    way=way,
                                    shot=shot,
                                    pre=True,
                                    transform_type=None,
                                    trial=10000)
            print('%d-way-%d-shot acc: %.3f\t%.3f'%(way,shot,mean,interval))