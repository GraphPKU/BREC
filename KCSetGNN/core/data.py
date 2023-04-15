import torch
import networkx as nx
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.data import Data
from torch_geometric.utils import to_undirected
from core.data_utils.data_pna import GraphPropertyDataset

import pickle, os, numpy as np
class PlanarSATPairsDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super(PlanarSATPairsDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["GRAPHSAT_new.pkl"]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        pass  

    def process(self):
        # Read data into huge `Data` list.
        data_list = pickle.load(open(os.path.join(self.root, "raw/GRAPHSAT_new.pkl"), "rb"))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class SRDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(SRDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["sr251256.g6"]  #sr251256  sr351668

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list. 
        dataset = nx.read_graph6(self.raw_paths[0])
        data_list = []
        for i,datum in enumerate(dataset):
            x = torch.ones(datum.number_of_nodes(),1)
            edge_index = to_undirected(torch.tensor(list(datum.edges())).transpose(1,0))            
            data_list.append(Data(edge_index=edge_index, x=x, y=0))
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


import scipy.io as sio
from scipy.special import comb
import numpy as np

class GraphCountDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(GraphCountDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        a=sio.loadmat(self.raw_paths[0])
        self.train_idx = torch.from_numpy(a['train_idx'][0])
        self.val_idx = torch.from_numpy(a['val_idx'][0]) 
        self.test_idx = torch.from_numpy(a['test_idx'][0]) 

    @property
    def raw_file_names(self):
        return ["randomgraph.mat"]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list. 
        b=self.processed_paths[0]       
        a=sio.loadmat(self.raw_paths[0]) #'subgraphcount/randomgraph.mat')
        # list of adjacency matrix
        A=a['A'][0]
        # list of output
        Y=a['F']

        data_list = []
        for i in range(len(A)):
            a=A[i]
            A2=a.dot(a)
            A3=A2.dot(a)
            tri=np.trace(A3)/6
            tailed=((np.diag(A3)/2)*(a.sum(0)-2)).sum()
            cyc4=1/8*(np.trace(A3.dot(a))+np.trace(A2)-2*A2.sum())
            cus= a.dot(np.diag(np.exp(-a.dot(a).sum(1)))).dot(a).sum()

            deg=a.sum(0)
            star=0
            for j in range(a.shape[0]):
                star+=comb(int(deg[j]),3)

            expy=torch.tensor([[tri,tailed,star,cyc4,cus]])

            E=np.where(A[i]>0)
            edge_index=torch.Tensor(np.vstack((E[0],E[1]))).type(torch.int64)
            x=torch.ones(A[i].shape[0],1).long() # change to category
            #y=torch.tensor(Y[i:i+1,:])            
            data_list.append(Data(edge_index=edge_index, x=x, y=expy))
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

from core.data_utils.cfi import create_cfi, create_grohe_cfi
from torch_geometric.utils import from_networkx

class CFI(InMemoryDataset):
    def __init__(self, root='data', kmin=3, kmax=10, transform=None, pre_transform=None, grohe=False):
        self.kmax = kmax
        self.kmin = kmin
        self.grohe = grohe
        super().__init__(f'{root}/CFI/{kmin}-{kmax}-grohe[{grohe}]', transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["G_{k}.gpickle" for k in range(self.kmin, self.kmax+1)] + ["H_{k}.gpickle" for k in range(self.kmin, self.kmax+1)]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        for k in range(self.kmin, self.kmax+1):
            G, H = create_grohe_cfi(k) if self.grohe else create_cfi(k)
            nx.write_gpickle(G, f"{self.raw_dir}/G_{k}.gpickle")
            nx.write_gpickle(H, f"{self.raw_dir}/H_{k}.gpickle")
    
    def process(self):
        data_list = []
        for k in range(self.kmin, self.kmax+1):
            G = nx.read_gpickle(f"{self.raw_dir}/G_{k}.gpickle")
            G = from_networkx(G, ['x'])
            H = nx.read_gpickle(f"{self.raw_dir}/H_{k}.gpickle")
            H = from_networkx(H, ['x'])
            G.y = torch.tensor([0])
            H.y = torch.tensor([1])
            G.k = torch.tensor([k])
            H.k = torch.tensor([k])
            data_list.append(G)
            data_list.append(H)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])