import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbones import Conv_4, ResNet
from .backbones import CSAM
from .backbones import SSAM


class DALSA(nn.Module):
    
    def __init__(self, way=None, shots=None, resnet=False, noise=False):
        
        super().__init__()
        

        self.resolution = 5*5
        if resnet:
            self.num_channel = 640
            self.feature_extractor = ResNet.resnet12()
            self.dim = self.num_channel*5*5
            
        else:
            self.num_channel = 64
            self.feature_extractor = Conv_4.BackBone(self.num_channel)            
            self.dim = self.num_channel*5*5

        self.ssam = SSAM.SSAM(hidden_size=self.num_channel, inner_size=self.num_channel, num_patch=self.resolution, drop_prob=0.1)

        # self.args = args
        self.resnet = resnet
        self.noise = noise
        self.CSAM = CSAM.CSAM(self.resnet, self.noise)
        self.shots = shots
        self.way = way


        self.scale = nn.Parameter(torch.FloatTensor([1.0]),requires_grad=True)

        self.w1 = nn.Parameter(torch.FloatTensor([0.5]),requires_grad=True)
        self.w2 = nn.Parameter(torch.FloatTensor([0.5]),requires_grad=True)
            

    def get_feature_vector(self,inp):

        # batch_size = inp.size(0)
        feature_map = self.feature_extractor(inp)

        return feature_map
    

    def get_neg_l2_dist(self,inp,way,shot,query_shot):

        # shape
        feature_map = self.get_feature_vector(inp)
        _, d, h, w = feature_map.shape
        m = h * w

        support = feature_map[:way * shot].view(way, shot, d, m)  # [15, 5, 640, 25]
        query = feature_map[way * shot:].view(1, -1, d, m)  # [1, 225, 640, 25]
        query_num = query.shape[1]

        w_spt, w_qry = self.CSAM(support, query.transpose(0, 1))
        support = w_spt.view(w_spt.size(0), w_spt.size(1), w_spt.size(2), h, w).permute(0, 2, 1, 3, 4).contiguous()
        query = w_qry.view(w_qry.size(0), w_qry.size(1), h, w)
        
        sq_similarity, qs_similarity = self.ssam(support, query)

        l2_dist = self.w1*sq_similarity + self.w2*qs_similarity
        
        return l2_dist



    
    def meta_test(self,inp,way,shot,query_shot):

        neg_l2_dist = self.get_neg_l2_dist(inp=inp,
                                        way=way,
                                        shot=shot,
                                        query_shot=query_shot)

        # 返回的是每个输入图像的预测标签，即最相似的训练样本的类别
        _,max_index = torch.max(neg_l2_dist,1)

        return max_index


    def forward(self,inp):

        logits = self.get_neg_l2_dist(inp=inp,
                                        way=self.way,
                                        shot=self.shots[0],
                                        query_shot=self.shots[1])
        logits = logits/self.dim*self.scale

        log_prediction = F.log_softmax(logits,dim=1)

        return log_prediction

if __name__ == '__main__':
    # model = resnet12()
    data = torch.randn(15, 3, 84, 84)
    # x = model(data)
    # print(x.size())
    # print(x.shape)
    model = DALSA(way=5,
                  shots=[1, 5],
                  resnet=True, noise=True)
    output = model(data)
    print(output)
    print(output.size())

    max_index = model.meta_test(data, way=5, shot=1, query_shot=5)
    print(max_index)
    print(max_index.size())

