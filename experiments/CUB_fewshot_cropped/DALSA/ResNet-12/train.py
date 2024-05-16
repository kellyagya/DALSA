import os
import sys
import torch
import yaml
from functools import partial
sys.path.append('../../../../')
from trainers import trainer, dalsa_train
from datasets import dataloaders
from models.DALSA import DALSA


args = trainer.train_parser()
with open('../../../../config.yml', 'r') as f:
    temp = yaml.safe_load(f)
data_path = os.path.abspath(temp['data_path'])
fewshot_path = os.path.join(data_path,'CUB_fewshot_cropped')

pm = trainer.Path_Manager(fewshot_path=fewshot_path,args=args)

train_way = args.train_way  # 15
shots = [args.train_shot, args.train_query_shot]  # [15, 5]
train_loader = dataloaders.meta_train_dataloader(data_path=pm.train,
                                                way=train_way,
                                                shots=shots,
                                                transform_type=args.train_transform_type)

model = DALSA(way=train_way,
            shots=[args.train_shot, args.train_query_shot],
            resnet=args.resnet, noise=args.noise)

train_func = partial(dalsa_train.default_train,train_loader=train_loader)

tm = trainer.Train_Manager(args,path_manager=pm,train_func=train_func)

tm.train(model)

tm.evaluate(model)