
import sklearn.neighbors
import torch
import ot

from scipy.spatial.distance import cosine
import numpy as np
import pandas as pd
import sklearn.neighbors
from torch_geometric.data import Data
import scipy.sparse as sp
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors
from scipy.sparse import coo_matrix

def Batch_Data(adata, num_batch_x, num_batch_y, spatial_key=['X', 'Y'], plot_Stats=False):
    Sp_df = adata.obs.loc[:, spatial_key].copy()
    Sp_df = np.array(Sp_df)
    batch_x_coor = [np.percentile(Sp_df[:, 0], (1 / num_batch_x) * x * 100) for x in range(num_batch_x + 1)]
    batch_y_coor = [np.percentile(Sp_df[:, 1], (1 / num_batch_y) * x * 100) for x in range(num_batch_y + 1)]

    Batch_list = []
    for it_x in range(num_batch_x):
        for it_y in range(num_batch_y):
            min_x = batch_x_coor[it_x]
            max_x = batch_x_coor[it_x + 1]
            min_y = batch_y_coor[it_y]
            max_y = batch_y_coor[it_y + 1]
            temp_adata = adata.copy()
            temp_adata = temp_adata[temp_adata.obs[spatial_key[0]].map(lambda x: min_x <= x <= max_x)]
            temp_adata = temp_adata[temp_adata.obs[spatial_key[1]].map(lambda y: min_y <= y <= max_y)]
            Batch_list.append(temp_adata)
    if plot_Stats:
        f, ax = plt.subplots(figsize=(1, 3))
        plot_df = pd.DataFrame([x.shape[0] for x in Batch_list], columns=['#spot/batch'])
        sns.boxplot(y='#spot/batch', data=plot_df, ax=ax)
        sns.stripplot(y='#spot/batch', data=plot_df, ax=ax, color='red', size=5)
    return Batch_list
def cal_hybrid_graph(adata, rad_cutoff=None, k_cutoff=None, model='Radius', alpha=0.8, sigma=1.0, verbose=True):


    assert model in ['Radius', 'KNN']
    assert 0 <= alpha <= 1
    if verbose:
        print('------Calculating hybrid graph...')
    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']

    expr = adata.X if isinstance(adata.X, np.ndarray) else adata.X.toarray()

    nbrs_euclidean = NearestNeighbors(radius=rad_cutoff if model == 'Radius' else None,
                                      n_neighbors=k_cutoff + 1 if model == 'KNN' else None).fit(coor)
    distances_euclidean, indices_euclidean = nbrs_euclidean.radius_neighbors(
        coor) if model == 'Radius' else nbrs_euclidean.kneighbors(coor)

    KNN_list_euclidean = []
    for it in range(indices_euclidean.shape[0]):
        euclidean_distances = distances_euclidean[it]
        euclidean_similarities = gaussian_kernel(euclidean_distances, sigma)
        KNN_list_euclidean.append(
            pd.DataFrame(zip([it] * len(indices_euclidean[it]), indices_euclidean[it], euclidean_similarities)))

    KNN_df_euclidean = pd.concat(KNN_list_euclidean)
    KNN_df_euclidean.columns = ['Cell1', 'Cell2', 'Similarity_euclidean']

    KNN_list_expr = []
    for it in range(indices_euclidean.shape[0]):
        expr_sims = [(1 - cosine(expr[it], expr[i]) + 1) / 2 for i in indices_euclidean[it]]  # 余弦相似度转换为[0, 1]
        KNN_list_expr.append(pd.DataFrame(zip([it] * len(indices_euclidean[it]), indices_euclidean[it], expr_sims)))

    KNN_df_expr = pd.concat(KNN_list_expr)
    KNN_df_expr.columns = ['Cell1', 'Cell2', 'Similarity_expr']

    KNN_df = pd.merge(KNN_df_euclidean, KNN_df_expr, on=['Cell1', 'Cell2'])
    beta = 1 - alpha
    KNN_df['Similarity'] = alpha * KNN_df['Similarity_euclidean'] + beta * KNN_df['Similarity_expr']

    hybrid_graph = KNN_df[['Cell1', 'Cell2', 'Similarity']].copy()
    hybrid_graph = hybrid_graph.loc[hybrid_graph['Similarity'] > 0,]
    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
    hybrid_graph['Cell1'] = hybrid_graph['Cell1'].map(id_cell_trans)
    hybrid_graph['Cell2'] = hybrid_graph['Cell2'].map(id_cell_trans)
    if verbose:
        print('The graph contains %d edges, %d cells.' % (hybrid_graph.shape[0], adata.n_obs))
        print('%.4f neighbors per cell on average.' % (hybrid_graph.shape[0] / adata.n_obs))

    adata.uns['hybrid_graph'] = hybrid_graph

    return adata


def Transfer_pytorch_Data(adata):
    G_df = adata.uns['hybrid_graph'].copy()
    cells = np.array(adata.obs_names)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)
    G = coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(adata.n_obs, adata.n_obs))
    G = G + coo_matrix(np.eye(G.shape[0]))

    edgeList = np.nonzero(G)
    if isinstance(adata.X, np.ndarray):
        data = Data(edge_index=torch.LongTensor(np.array([edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.X))
    else:
        data = Data(edge_index=torch.LongTensor(np.array([edgeList[0], edgeList[1]])),
                    x=torch.FloatTensor(adata.X.todense()))
    return data
def gaussian_kernel(distance, sigma=1.0):
    return np.exp(-distance ** 2 / (2 * sigma ** 2))

def stats_hybrid_graph(adata):
    import matplotlib.pyplot as plt
    Num_edge = adata.uns['hybrid_graph']['Cell1'].shape[0]
    Mean_edge = Num_edge/adata.shape[0]
    plot_df = pd.value_counts(pd.value_counts(adata.uns['hybrid_graph']['Cell1']))
    plot_df = plot_df/adata.shape[0]


#EEE', 'VII', 'VVV'
def mclust_R(adata, num_cluster,rad_cutoff , modelNames='EEE', used_obsm='stHGC', random_seed=2020, refinement=False):

    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']
    a = adata.obsm[used_obsm]#
    b = rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm])

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    # res = list(res)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')

    if refinement:
        new_type = refine_label(adata, rad_cutoff, key='mclust')
        adata.obs['mclust'] = new_type

    return adata

def refine_label(adata, rad_cutoff=150, key='label'):
    n_neigh = rad_cutoff
    new_type = []
    old_type = adata.obs[key].values

    # calculate distance
    position = adata.obsm['spatial']
    distance = ot.dist(position, position, metric='euclidean')

    n_cell = distance.shape[0]

    for i in range(n_cell):
        vec = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh + 1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)

    new_type = [str(i) for i in list(new_type)]

    return new_type