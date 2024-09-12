import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.sparse as sp

import stHGC
from utils import Transfer_pytorch_Data

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
cudnn.deterministic = True
cudnn.benchmark = True
import torch.nn.functional as F
# from loss import *

from process import *
from torch_geometric.transforms import ToSparseTensor
import  torch
import numpy as np
import pandas as pd
import sklearn
import scipy.sparse as sp

def stHGC_train(adata, hidden_dims=[512, 30], n_epochs=1000,lr=0.0009,key_added='stHGC',
                  gradient_clipping=5., weight_decay=0.0001, verbose=True,
                  random_seed=2020, save_loss=False, save_reconstrction=False,
                  device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):

    # # seed_everything()
    seed = random_seed
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    datatype = '10X'


    adata.X = sp.csr_matrix(adata.X)

    if 'highly_variable' in adata.var.columns:
        adata_Vars = adata[:, adata.var['highly_variable']]
    else:
        adata_Vars = adata



    if verbose:
        print('Size of Input: ', adata_Vars.shape)
    if 'hybrid_graph' not in adata.uns.keys():
        raise ValueError("Haybrid_graph is not existed! Run Cal_Spatial_Net first!")

    if 'label_CSL' not in adata.obsm.keys():
        add_contrastive_label(adata)
    if 'feat' not in adata.obsm.keys():
        get_feature(adata)
    if 'adj' not in adata.obsm.keys():
        if datatype in ['Stereo', 'Slide']:
            construct_interaction_KNN(adata)
        else:
            construct_interaction(adata)

    data = Transfer_pytorch_Data(adata_Vars)

    model = stHGC.stHGC(hidden_dims=[data.x.shape[1]] + hidden_dims).to(device)

    class SimpleClass:
        def __init__(self, adata, device):
            self.adata = adata
            self.device = device
            self.loss_CSL = nn.BCEWithLogitsLoss()
            self.features = torch.FloatTensor(self.adata.obsm['feat'].copy()).to(self.device)
            self.features_a = torch.FloatTensor(self.adata.obsm['feat_a'].copy()).to(self.device)
            self.label_CSL = torch.FloatTensor(self.adata.obsm['label_CSL']).to(self.device)
            # self.adj = self.adata.obsm['adj_t']

    simple_instance = SimpleClass(adata, device)

    data = data.to(device)
    data = ToSparseTensor()(data)  ##transfer data to sparse data which can ensure the reproducibility when seed fixed
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


    for epochs in tqdm(range(1, n_epochs + 1)):
        model.train()
        optimizer.zero_grad()

        z, out, ret, ret_a ,emb = model(data.x, data.adj_t,simple_instance.features_a)#jia ru kong jian

        reconstruction_loss = F.mse_loss(data.x, out)

        lamda = 10
        mu = 10
        grmma =1


        loss_sl_1 = simple_instance.loss_CSL(ret, simple_instance.label_CSL)
        loss_sl_2 = simple_instance.loss_CSL(ret_a, simple_instance.label_CSL)
        sadj, graph_nei, graph_neg = spatial_construct_graph1(adata, radius=150)

        reg_loss = regularization(emb, graph_nei, graph_neg)

        loss = lamda*reconstruction_loss + mu *  (loss_sl_1 + loss_sl_2)+ grmma * reg_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
        optimizer.step()

    model.eval()

    z, out, ret, ret_a ,emb= model(data.x, data.adj_t, simple_instance.features_a)

    stHGC_rep = z.to('cpu').detach().numpy()
    adata.obsm[key_added] = stHGC_rep

    if save_reconstrction:

        ReX = out.to('cpu').detach().numpy()
        ReX[ReX < 0] = 0
        adata.layers['stHGC_ReX'] = ReX

    return adata