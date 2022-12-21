from collections import OrderedDict
from os.path import join
import pdb

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from myexp.amil import SNN_Block, Attn_Net_Gated



class ABC(nn.Module):
    def __init__(self, fusion='concat', 
                omic_sizes=[100, 200, 300, 400, 500, 600], 
                n_classes=4,
                model_size_omic: str='small', 
                dropout=0.25):
        super(ABC, self).__init__()
        self.fusion = fusion
        self.omic_sizes = omic_sizes
        self.n_classes = n_classes
        self.size_dict_WSI = [2048, 1024, 512, 256]
        self.size_dict_omic = {'small': [512, 512, 256], 'big': [1024, 1024, 1024, 256]}
        
        # PART 1
        ### FC Layer over WSI bag
        size = self.size_dict_WSI
        fc1 = [nn.Linear(size[0], size[1]), nn.ReLU()]      # 2048 -> 1024
        fc1.append(nn.Dropout(0.25))
        fc2 = [nn.Linear(size[1], size[2]), nn.ReLU()]      # 1024 -> 512
        fc2.append(nn.Dropout(0.25))
        self.wsi_net = nn.Sequential(*fc1, *fc2)

        ### Constructing Genomic SNN
        hidden = self.size_dict_omic[model_size_omic]
        sig_networks = []
        for input_dim in omic_sizes:
            # GNN for each embedding
            fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]   # input -> 512
            fc_omic.append(SNN_Block(dim1=hidden[0], dim2=hidden[1], dropout=0.25)) # 512 -> 512
            sig_networks.append(nn.Sequential(*fc_omic))
        self.sig_networks = nn.ModuleList(sig_networks)


        # PART2
        ### Multihead Attention
        self.coattn = CrossAttention(hidden_size=512)
        # N, 512 -> 6, 512


        # PART 3
        ### Deep Sets Architecture Construction
        path_attention_net = Attn_Net_Gated(L=size[2], D=size[3], dropout=dropout, n_classes=1) 
        self.path_attention_net = path_attention_net  # A: 6*1, h: 1*512
        self.path_rho = nn.Sequential(*[nn.Linear(size[2], size[3]), 
                                    nn.ReLU(), 
                                    nn.Dropout(dropout)])   # 1, 256

        ### Constructing Genomic SNN
        omic_attention_net = Attn_Net_Gated(L=hidden[1], D=hidden[2], dropout=dropout, n_classes=1) 
        self.omic_attention_net = omic_attention_net  # A: 6*1, h: 1*256
        self.omic_rho = nn.Sequential(*[nn.Linear(hidden[1], hidden[2]), 
                                    nn.ReLU(), 
                                    nn.Dropout(dropout)])   # 1,256

        
        # PART 5
        ### Fusion Layer
        if self.fusion == 'concat':
            self.mm = nn.Sequential(*[nn.Linear(256*2, size[3]), 
                                    nn.ReLU(), 
                                    nn.Linear(size[3], size[3]), 
                                    nn.ReLU()])
        else:
            self.mm = None
        
        ### Classifier                        )
        self.classifier = nn.Linear(size[3], n_classes)
        # self.softmax = nn.Softmax(dim=-1)


    def forward(self, **kwargs):
        x_path = kwargs['x_path']
        x_omic = [kwargs['x_omic%d' % i] for i in range(1,7)]

        # Bag-level representation
        h_path_bag = self.wsi_net(x_path) ### wsi: 2048 -> 512
        h_omic = [self.sig_networks[idx].forward(sig_feat) for idx, sig_feat in enumerate(x_omic)] ### omic: input -> 512
        h_omic_bag = torch.stack(h_omic) ### 6*512

        # Coattn
        # Gbag, Hcoattn <- Q, K, V
        h_path_coattn, A_coattn = self.coattn(h_omic_bag, h_path_bag, h_path_bag)
        # print('Coattn', h_path_coattn.shape, A_coattn.shape, h_omic_bag.shape)
        # torch.Size([1, 6, 512]) torch.Size([1, 6, M]) torch.Size([6, 512])
        # h_path_coattn: 1*6*512,  A_coattn: 6*N

        # Attention MIL
        # print('Attention MIL:')
        h_path_coattn = h_path_coattn.squeeze() # 6*512
        A1, h_path = self.path_attention_net(h_path_coattn)  # A: 6*1, h_path: 1*512
        # print('path0', h_path.shape)
        A1 = torch.transpose(A1, 1, 0)    # 1*6
        A1 = F.softmax(A1, dim=1) 
        h_path = torch.mm(A1, h_path)    # 1 * 512
        # print('path1', A1.shape, h_path.shape)
        h_path = self.path_rho(h_path).squeeze()   #  1 * 256
        # print('path2', h_path.shape)

        # Apply AttentionMIL for Genomic too
        h_omic = h_omic_bag.squeeze()   # 6*512
        A2, h_omic = self.omic_attention_net(h_omic)  # A: 6*1, h_omic: 1*256
        # print('omic0', h_omic.shape)
        A2 = torch.transpose(A2, 1, 0)    # 1*6
        A2 = F.softmax(A2, dim=1) 
        h_omic = torch.mm(A2, h_omic)    # 1 * 512
        # print('omic1', h_omic.shape)
        # print(h_omic.shape, A2.shape)      # torch.Size([1, 512]) torch.Size([1, 6])
        h_omic = self.omic_rho(h_omic).squeeze()   #  1 * 256
        # print('omic2', h_omic.shape)


        # Fusion
        if self.fusion == 'concat':
            h = self.mm(torch.cat([h_path, h_omic], axis=0))


        logits  = self.classifier(h).unsqueeze(0) # logits needs to be a [1 x classes] vector 
        return logits, A_coattn



class CrossAttention(nn.Module):
    def __init__(self, hidden_size):
        super(CrossAttention, self).__init__()

        self.hidden_size = hidden_size

        self.Q = nn.Linear(hidden_size, hidden_size)
        self.K = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=1)
        self.scaling_factor = torch.rsqrt(torch.tensor(self.hidden_size, dtype= torch.float))

    def forward(self, queries, keys, values):
        """The forward pass of the scaled dot attention mechanism.

        Arguments:
            queries: The current decoder hidden state, 2D or 3D tensor. (N x hidden_size)
            keys: The encoder hidden states for each step of the input sequence. (M x hidden_size)
            values: The encoder hidden states for each step of the input sequence. (M x hidden_size)

        Returns:
            context: weighted average of the values (batch_size x k x hidden_size)
            attention_weights: Normalized attention weights for each encoder hidden state. (batch_size x seq_len x (k))

            The output must be a softmax weighting over the seq_len annotations.
        """
        # print(queries.shape, keys.shape)
        # torch.Size([6, 512]) torch.Size([300, 512])
        q = self.Q(queries).unsqueeze(0) 
        k = self.K(keys).unsqueeze(0) 
        v = self.V(values).unsqueeze(0) 
        # print(q.shape, k.shape, v.shape)
        #torch.Size([1, 6, 512]) torch.Size([1, 300, 512]) torch.Size([1, 300, 512])
        unnormalized_attention = self.scaling_factor * torch.bmm(q, k.transpose(1,2)) 
        attention_weights = self.softmax(unnormalized_attention) 
        # print(attention_weights.shape)
        # torch.Size([1, 6, 300])
        context = torch.bmm(attention_weights, v)
        # print(context.shape)
        # torch.Size([6, 1, 512]) torch.Size([1, 1, 6, M])

        return context, attention_weights