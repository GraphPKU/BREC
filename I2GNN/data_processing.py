import numpy as np
import torch
import csv
# from rdkit import Chem
import torch
from networkx import read_graph6
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.data import DataLoader
import utils
import os
import random
import shutil
from itertools import repeat
# from k_gnn import GraphConv, DataLoader, avg_pool
# from k_gnn import TwoMalkin, ConnectedThreeMalkin
import os
import os.path as osp
import sys
from typing import Callable, List, Optional
import torch.nn.functional as F
from torch_scatter import scatter
from tqdm import tqdm
from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)

from torch_geometric.io import read_tu_data
import scipy.io as scio
from utils import k_hop_subgraph, subgraph_to_subgraph2
import networkx as nx

class pygdataset(InMemoryDataset):
    def __init__(self, url=None, dataname='mols', root='data', processed_name='processed', homo=True,
                 transform=None, pre_transform=None, pre_filter=None):
        self.url = url
        self.root = root
        self.dataname = dataname
        self.transform = transform
        self.homo = homo
        self.pre_filter = pre_filter
        self.pre_transform = pre_transform
        self.raw = os.path.join(root, dataname)
        self.processed = os.path.join(root, dataname, processed_name)
        super(pygdataset, self).__init__(root=root, transform=transform, pre_transform=pre_transform,
                                            pre_filter=pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.x_dim = self.data.x.size(-1)
        self.y_dim = self.data.y.size(-1)
        self.e_dim = torch.max(self.data.edge_attr).item() + 1

    @property
    def raw_dir(self):
        name = 'raw'
        return os.path.join(self.root, self.dataname, name)

    @property
    def processed_dir(self):
        return self.processed
    @property
    def raw_file_names(self):
        names = ["data"]
        return ['{}_{}.npy'.format(name, self.dataname) for name in names]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def adj2data(self, d):
        # x: (n, d), A: (e, n, n)
        x, A, y = d['x'], d['A'], d['y']
        x = torch.tensor(x)
        if self.homo:
            x = torch.ones_like(x)
        assert x.size(0) == A.shape[-1]
        begin, end = np.where(np.sum(A, axis=0) == 1.)
        edge_index = torch.tensor(np.array([begin, end]))
        edge_attr = torch.argmax(torch.tensor(A[:, begin, end].T), dim=-1)
        # y = torch.tensor(np.concatenate((y[1], y[-1])))
        y = torch.tensor(y[-1])
        y = y.view([1, len(y)])
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    @staticmethod
    def wrap2data(d):
        # x: (n, d), A: (e, n, n)
        x, A, y = d['x'], d['A'], d['y']
        x = torch.tensor(x)
        begin, end = np.where(np.sum(A, axis=0) == 1.)
        edge_index = torch.tensor(np.array([begin, end]))
        edge_attr = torch.argmax(torch.tensor(A[:, begin, end].T), dim=-1)
        y = torch.tensor(y[-1:])
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    def process(self):
        # process npy data into pyg.Data
        print('Processing data from ' + self.raw_dir + '...')
        raw_data = np.load(os.path.join(self.raw_dir, "data_" + self.dataname + ".npy"), allow_pickle=True)
        data_list = [self.adj2data(d) for d in raw_data]
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            temp = []
            for i, data in enumerate(data_list):
                if i % 100 == 0:
                    print(i)
                temp.append(self.pre_transform(data))
            data_list = temp
            # data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])



class dataset_count(InMemoryDataset):
    def __init__(self, url=None, dataname='counting', root='data', processed_name='processed',
                 transform=None, pre_transform=None, pre_filter=None):
        self.url = url
        self.root = root
        self.dataname = dataname
        self.transform = transform
        self.pre_filter = pre_filter
        self.pre_transform = pre_transform
        self.raw = os.path.join(root, dataname)
        self.processed = os.path.join(root, dataname, processed_name)
        super(dataset_count, self).__init__(root=root, transform=transform, pre_transform=pre_transform,
                                         pre_filter=pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.y_dim = self.data.y.size(-1)
        # self.e_dim = torch.max(self.data.edge_attr).item() + 1

    @property
    def raw_dir(self):
        name = 'raw'
        return os.path.join(self.root, self.dataname, name)

    @property
    def processed_dir(self):
        return self.processed
    @property
    def raw_file_names(self):
        names = ["data"]
        return ['{}.npy'.format(name) for name in names]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def adj2data(self, d):
        # x: (n, d), A: (e, n, n)
        A, y = d['A'], d['y']
        begin, end = np.where(np.sum(A, axis=0) == 1.)
        edge_index = torch.tensor(np.array([begin, end]))
        # edge_attr = torch.argmax(torch.tensor(A[:, begin, end].T), dim=-1)
        # y = torch.tensor(np.concatenate((y[1], y[-1])))
        # y = torch.tensor(y[-1])
        # y = y.view([1, len(y)])

        # sanity check
        assert np.min(begin) == 0
        num_nodes = np.max(begin) + 1
        if y.ndim == 1:
            y = y.reshape([1, -1])
        return Data(edge_index=edge_index, y=torch.tensor(y), num_nodes=torch.tensor([num_nodes]))

    @staticmethod
    def wrap2data(d):
        # x: (n, d), A: (e, n, n)
        x, A, y = d['x'], d['A'], d['y']
        x = torch.tensor(x)
        begin, end = np.where(np.sum(A, axis=0) == 1.)
        edge_index = torch.tensor(np.array([begin, end]))
        edge_attr = torch.argmax(torch.tensor(A[:, begin, end].T), dim=-1)
        y = torch.tensor(y[-1:])
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    def process(self):
        # process npy data into pyg.Data
        print('Processing data from ' + self.raw_dir + '...')
        raw_data = np.load(self.raw_paths[0], allow_pickle=True)
        data_list = [self.adj2data(d) for d in raw_data]
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            temp = []
            for i, data in enumerate(data_list):
                if i % 100 == 0:
                    print('Pre-processing %d/%d' % (i, len(data_list)))
                temp.append(self.pre_transform(data))
            data_list = temp
            # data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])



class dataset_random_graph(InMemoryDataset):
    def __init__(self, url=None, dataname='count_cycle', root='data', processed_name='processed', split='train',
                 transform=None, pre_transform=None, pre_filter=None):
        self.url = url
        self.root = root
        self.dataname = dataname
        self.transform = transform
        self.pre_filter = pre_filter
        self.pre_transform = pre_transform
        self.raw = os.path.join(root, dataname)
        self.processed = os.path.join(root, dataname, processed_name)
        super(dataset_random_graph, self).__init__(root=root, transform=transform, pre_transform=pre_transform,
                                            pre_filter=pre_filter)
        split_id = 0 if split == 'train' else 1 if split == 'val' else 2
        self.data, self.slices = torch.load(self.processed_paths[split_id])
        self.y_dim = self.data.y.size(-1)
        # self.e_dim = torch.max(self.data.edge_attr).item() + 1

    @property
    def raw_dir(self):
        name = 'raw'
        return os.path.join(self.root, self.dataname, name)

    @property
    def processed_dir(self):
        return self.processed
    @property
    def raw_file_names(self):
        names = ["data"]
        return ['{}.mat'.format(name) for name in names]

    @property
    def processed_file_names(self):
        return ['data_tr.pt', 'data_val.pt', 'data_te.pt']

    def adj2data(self, A, y):
        # x: (n, d), A: (e, n, n)
        # begin, end = np.where(np.sum(A, axis=0) == 1.)
        begin, end = np.where(A == 1.)
        edge_index = torch.tensor(np.array([begin, end]))
        # edge_attr = torch.argmax(torch.tensor(A[:, begin, end].T), dim=-1)
        # y = torch.tensor(np.concatenate((y[1], y[-1])))
        # y = torch.tensor(y[-1])
        # y = y.view([1, len(y)])

        # sanity check
        # assert np.min(begin) == 0
        num_nodes = A.shape[0]
        if y.ndim == 1:
            y = y.reshape([1, -1])
        return Data(edge_index=edge_index, y=torch.tensor(y), num_nodes=torch.tensor([num_nodes]))

    @staticmethod
    def wrap2data(d):
        # x: (n, d), A: (e, n, n)
        x, A, y = d['x'], d['A'], d['y']
        x = torch.tensor(x)
        begin, end = np.where(np.sum(A, axis=0) == 1.)
        edge_index = torch.tensor(np.array([begin, end]))
        edge_attr = torch.argmax(torch.tensor(A[:, begin, end].T), dim=-1)
        y = torch.tensor(y[-1:])
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    def process(self):
        # process npy data into pyg.Data
        print('Processing data from ' + self.raw_dir + '...')
        raw_data = scio.loadmat(self.raw_paths[0])
        if raw_data['F'].shape[0] == 1:
            data_list_all = [[self.adj2data(raw_data['A'][0][i], raw_data['F'][0][i]) for i in idx]
                             for idx in [raw_data['train_idx'][0], raw_data['val_idx'][0], raw_data['test_idx'][0]]]
        else:
            data_list_all = [[self.adj2data(A, y) for A, y in zip(raw_data['A'][0][idx][0], raw_data['F'][idx][0])]
                        for idx in [raw_data['train_idx'], raw_data['val_idx'], raw_data['test_idx']]]
        for save_path, data_list in zip(self.processed_paths, data_list_all):
            print('pre-transforming for data at'+save_path)
            if self.pre_filter is not None:
                data_list = [data for data in data_list if self.pre_filter(data)]
            if self.pre_transform is not None:
                temp = []
                for i, data in enumerate(data_list):
                    if i % 100 == 0:
                        print('Pre-processing %d/%d' % (i, len(data_list)))
                    temp.append(self.pre_transform(data))
                data_list = temp
                # data_list = [self.pre_transform(data) for data in data_list]
            data, slices = self.collate(data_list)
            torch.save((data, slices), save_path)



class dataset_sr25(InMemoryDataset):
    def __init__(self, url=None, dataname='sr25', root='data', processed_name='processed',
                 transform=None, pre_transform=None, pre_filter=None):
        self.url = url
        self.root = root
        self.dataname = dataname
        self.transform = transform
        self.pre_filter = pre_filter
        self.pre_transform = pre_transform
        self.raw = os.path.join(root, dataname)
        self.processed = os.path.join(root, dataname, processed_name)
        super(dataset_sr25, self).__init__(root=root, transform=transform, pre_transform=pre_transform,
                                                   pre_filter=pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        # self.e_dim = torch.max(self.data.edge_attr).item() + 1

    @property
    def raw_dir(self):
        name = 'raw'
        return os.path.join(self.root, self.dataname, name)

    @property
    def processed_dir(self):
        return self.processed
    @property
    def raw_file_names(self):
        names = ["data"]
        return ['{}.g6'.format(name) for name in names]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def nx2data(self, d):
        # x: (n, d), A: (e, n, n)
        # begin, end = np.where(np.sum(A, axis=0) == 1.)
        edge_index = np.array(list(d.edges)).T
        edge_index = np.concatenate([edge_index, np.array([edge_index[-1], edge_index[0]])], axis=-1)
        edge_index = torch.tensor(edge_index).long()
        return Data(edge_index=edge_index, num_nodes=torch.max(edge_index).item()+1)

    @staticmethod
    def wrap2data(d):
        # x: (n, d), A: (e, n, n)
        x, A, y = d['x'], d['A'], d['y']
        x = torch.tensor(x)
        begin, end = np.where(np.sum(A, axis=0) == 1.)
        edge_index = torch.tensor(np.array([begin, end]))
        edge_attr = torch.argmax(torch.tensor(A[:, begin, end].T), dim=-1)
        y = torch.tensor(y[-1:])
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    def process(self):
        # process npy data into pyg.Data
        print('Processing data from ' + self.raw_dir + '...')
        raw_data = read_graph6(self.raw_paths[0])
        data_list = [self.nx2data(d) for d in raw_data]
        print('pre-transforming for data at'+self.processed_paths[0])
        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])




class dataset_airport(InMemoryDataset):
    def __init__(self, url=None, dataset='brazil', root='data', processed_name='processed',
                 transform=None, pre_filter=None, h=3, node_label='spd', use_rd=True, model='Nested2_k1_GNN'):
        self.url = url
        self.root = root
        self.dataname = os.path.join('airport', dataset)
        self.transform = transform
        self.pre_filter = pre_filter
        # self.pre_transform = pre_transform
        self.h = h
        self.node_label = node_label
        self.use_rd = use_rd
        self.model = model
        self.raw = os.path.join(root, self.dataname)
        self.processed = os.path.join(root, self.dataname, processed_name)
        super(dataset_airport, self).__init__(root=root, transform=transform, pre_filter=pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        # self.e_dim = torch.max(self.data.edge_attr).item() + 1

    @property
    def raw_dir(self):
        return self.raw

    @property
    def processed_dir(self):
        return self.processed
    @property
    def raw_file_names(self):
        names = ["edges", "labels"]
        return ['{}.txt'.format(name) for name in names]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def node2data(self, edge_index, node, degree):
        # x: (n, d), A: (e, n, n)
        # begin, end = np.where(np.sum(A, axis=0) == 1.)
        ind, y = node
        nodes_, edge_index_, edge_mask_, z_, relabel = \
            k_hop_subgraph(ind, self.h, edge_index=edge_index, relabel_nodes=True, node_label=self.node_label)
        assert len(nodes_) == torch.max(edge_index_).item() + 1
        d = np.array(degree(np.array(relabel)))[:, 1]
        return Data(edge_index=edge_index_, num_nodes=len(nodes_), x=torch.tensor(d).float(),z=z_, y=y)

    @staticmethod
    def read_label(f_path):
        fin_labels = open(f_path)
        labels = []
        node_id_mapping = dict()
        for new_id, line in enumerate(fin_labels.readlines()):
            old_id, label = line.strip().split()
            labels.append(int(label))
            node_id_mapping[old_id] = new_id
        fin_labels.close()
        return labels, node_id_mapping

    @staticmethod
    def read_edges(f_path, node_id_mapping):
        edges = []
        fin_edges = open(f_path)
        for line in fin_edges.readlines():
            node1, node2 = line.strip().split()[:2]
            edges.append([node_id_mapping[node1], node_id_mapping[node2]])
        fin_edges.close()
        return edges

    def process(self):
        # process npy data into pyg.Data
        print('Processing data from ' + self.raw_dir + '...')
        edges, labels = [], []
        labels, node_id_mapping = self.read_label(self.raw_paths[1])
        edges = self.read_edges(self.raw_paths[0], node_id_mapping)
        G = nx.Graph(edges)
        degree = G.degree
        edge_index = torch.tensor(edges).long().t().contiguous()
        edge_index = torch.cat([edge_index, edge_index[[1, 0], :]], dim=-1)
        data_list = [self.node2data(edge_index, node, degree) for node in enumerate(labels)]
        if self.model == 'Nested2_k1_GNN':
            print('pre-transforming for data at'+self.processed_paths[0])
            data_list_new = []
            for i, d in enumerate(data_list):
                if i % 100 == 0:
                    print('Preprocessing: %d/%d'%(i, len(data_list)))
                d_new = subgraph_to_subgraph2(d, self.h, use_rd=self.use_rd)
                d_new.y = d.y
                data_list_new.append(d_new)
            data_list = data_list_new
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])



def load_raw_csv(data_path):
    data = []
    with open(data_path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            data.append(row)
    return data


def write_csv(data, save_path):
    # first data into row
    data_save = {}
    for k in data[0].keys():
        data_save[k] = [data[0][k]]
    for i in range(1, len(data)):
        for k in data[0].keys():
            data_save[k].append(data[i][k])

    with open(save_path, mode='w', newline="") as outfile:
        writer = csv.writer(outfile)
        # pass the dictionary keys to writerow
        # function to frame the columns of the csv file
        writer.writerow(data_save.keys())

        # make use of writerows function to append
        # the remaining values to the corresponding
        # columns using zip function.
        writer.writerows(zip(*data_save.values()))


def create_one_hot_label(d, max_num_rings):
    # please manually define this function replying on the labels you want
    num_labels = 2 + (1 + max_num_rings) + 2 # 1-bit for HAS RING, 1-bit for HAS tricycles
    labels = []
    # if has ring
    flag = [1., 0] if d['has_rings'] == 'True' else [0, 1.]
    labels.append(np.array(flag).astype(np.float32))
    # how many rings
    flag = np.eye(max_num_rings + 1)[int(d['nring'])]
    labels.append(flag.astype(np.float32))
    # if has 3-ring
    # flag = [1., 0] if int(d['natom_in_3_rings']) > 0 else [0, 1.]
    # mol = Chem.MolFromSmiles(Chem.CanonSmiles(d['smiles']))
    # flag = utils.detect_triple_ring(mol)
    flag = [1., 0] if d['has_triple_ring'] == 'True' else [0, 1.]
    labels.append(np.array(flag).astype(np.float32))

    return labels


def smi2graph(smi, node_voc, edge_voc):
    # transform smiles into node features x and edge features A using vocabularies node_voc and edge_voc
    mol = Chem.MolFromSmiles(Chem.CanonSmiles(smi))
    num_atoms = mol.GetNumAtoms()
    num_node_type = len(node_voc)
    num_edge_type = len(edge_voc)
    x = np.zeros([num_atoms, num_node_type])
    A = np.zeros([num_edge_type, num_atoms, num_atoms])
    for i, atom in enumerate(mol.GetAtoms()):
        x[i, node_voc[atom.GetAtomicNum()]] = 1.
    for edge in mol.GetBonds():
        begin_idx = edge.GetBeginAtomIdx()
        end_idx = edge.GetEndAtomIdx()
        bond_type = edge.GetBondType()
        A[edge_voc[bond_type], begin_idx, end_idx] = 1.
        A[edge_voc[bond_type], end_idx, begin_idx] = 1.
    return x, A


def data_preprocessing(raw_data):
    # data_preprocessing: 1. create two dictionary for label mapping; 2. create a preprocessed data file
    processed_data = []
    label_dict = {}

    # create type <-> index mapping
    print('Vocabulary generation...')
    max_num_rings = 0
    node_attr_set = set()
    edge_attr_set = set()
    for d in raw_data:
        mol = Chem.MolFromSmiles(Chem.CanonSmiles(d['smiles']))
        for atom in mol.GetAtoms():
            node_attr_set.add(atom.GetAtomicNum())
        for edge in mol.GetBonds():
            edge_attr_set.add(edge.GetBondType())
        if int(d['nring']) > max_num_rings:
            max_num_rings = int(d['nring'])

    num_node_type = len(node_attr_set)
    num_edge_type = len(edge_attr_set)
    node_voc = {}
    edge_voc = {}
    for i, node_type in enumerate(node_attr_set):
        node_voc[node_type] = i
    for i, edge_type in enumerate(edge_attr_set):
        edge_voc[edge_type] = i
    print('Vocabulary generation done!')

    # create one hot features and labels
    print('Features generation...')
    num_samples = len(raw_data)
    for i, d in enumerate(raw_data):
        if i % 500 == 0:
            print('\r' + 'Generation process: %d/%d' % (i, num_samples), end="")
        # add processed data point
        x, A = smi2graph(d['smiles'], node_voc, edge_voc)
        processed_dp = {}
        processed_dp['smiles'] = d['smiles']
        processed_dp['x'] = x.astype(np.float32)
        processed_dp['A'] = A.astype(np.float32)
        processed_dp['num_nodes'] = x.shape[0]
        processed_dp['y'] = create_one_hot_label(d, max_num_rings)
        processed_data.append(processed_dp)
    print('\nFeatures generation done!')
    voc = {}
    voc['node_voc'] = node_voc
    voc['edge_voc'] = edge_voc
    return processed_data, voc


class graph_dataset(torch.utils.data.Dataset):
    def __init__(self, graphs, homo=False):
        # raw data is a list of smiles and other labels
        self.graphs = graphs
        self.max_num_atoms = 0
        self.num_samples = len(graphs)
        for g in self.graphs:
            num_atoms = g['num_nodes']
            if num_atoms > self.max_num_atoms:
                self.max_num_atoms = num_atoms
        self.homo = homo

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        # pad to max num of nodes
        x, A, y = self.graphs[item]['x'], self.graphs[item]['A'], self.graphs[item]['y']
        if self.homo:
            x = np.zeros_like(x)
            x[:, 0] = 1.
        x = np.pad(x, ((0, self.max_num_atoms - x.shape[0]), (0, 0)))
        A = np.pad(A, ((0, 0), (0, self.max_num_atoms - A.shape[1]), (0, self.max_num_atoms - A.shape[2])))
        return {'x': x, 'A': A, 'y': y, 'num_nodes': self.graphs[item]['num_nodes'], 'node_mask': np.sum(x, axis=-1, keepdims=True)}




HAR2EV = 27.211386246
KCALMOL2EV = 0.04336414

conversion = torch.tensor([
    1., 1., HAR2EV, HAR2EV, HAR2EV, 1., HAR2EV, HAR2EV, HAR2EV, HAR2EV, HAR2EV,
    1., KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, 1., 1., 1.
])

atomrefs = {
    6: [0., 0., 0., 0., 0.],
    7: [
        -13.61312172, -1029.86312267, -1485.30251237, -2042.61123593,
        -2713.48485589
    ],
    8: [
        -13.5745904, -1029.82456413, -1485.26398105, -2042.5727046,
        -2713.44632457
    ],
    9: [
        -13.54887564, -1029.79887659, -1485.2382935, -2042.54701705,
        -2713.42063702
    ],
    10: [
        -13.90303183, -1030.25891228, -1485.71166277, -2043.01812778,
        -2713.88796536
    ],
    11: [0., 0., 0., 0., 0.],
}


class QM9(InMemoryDataset):
    r"""The QM9 dataset from the `"MoleculeNet: A Benchmark for Molecular
    Machine Learning" <https://arxiv.org/abs/1703.00564>`_ paper, consisting of
    about 130,000 molecules with 19 regression targets.
    Each molecule includes complete spatial information for the single low
    energy conformation of the atoms in the molecule.
    In addition, we provide the atom features from the `"Neural Message
    Passing for Quantum Chemistry" <https://arxiv.org/abs/1704.01212>`_ paper.

    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | Target | Property                         | Description                                                                       | Unit                                        |
    +========+==================================+===================================================================================+=============================================+
    | 0      | :math:`\mu`                      | Dipole moment                                                                     | :math:`\textrm{D}`                          |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 1      | :math:`\alpha`                   | Isotropic polarizability                                                          | :math:`{a_0}^3`                             |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 2      | :math:`\epsilon_{\textrm{HOMO}}` | Highest occupied molecular orbital energy                                         | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 3      | :math:`\epsilon_{\textrm{LUMO}}` | Lowest unoccupied molecular orbital energy                                        | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 4      | :math:`\Delta \epsilon`          | Gap between :math:`\epsilon_{\textrm{HOMO}}` and :math:`\epsilon_{\textrm{LUMO}}` | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 5      | :math:`\langle R^2 \rangle`      | Electronic spatial extent                                                         | :math:`{a_0}^2`                             |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 6      | :math:`\textrm{ZPVE}`            | Zero point vibrational energy                                                     | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 7      | :math:`U_0`                      | Internal energy at 0K                                                             | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 8      | :math:`U`                        | Internal energy at 298.15K                                                        | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 9      | :math:`H`                        | Enthalpy at 298.15K                                                               | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 10     | :math:`G`                        | Free energy at 298.15K                                                            | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 11     | :math:`c_{\textrm{v}}`           | Heat capavity at 298.15K                                                          | :math:`\frac{\textrm{cal}}{\textrm{mol K}}` |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 12     | :math:`U_0^{\textrm{ATOM}}`      | Atomization energy at 0K                                                          | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 13     | :math:`U^{\textrm{ATOM}}`        | Atomization energy at 298.15K                                                     | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 14     | :math:`H^{\textrm{ATOM}}`        | Atomization enthalpy at 298.15K                                                   | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 15     | :math:`G^{\textrm{ATOM}}`        | Atomization free energy at 298.15K                                                | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 16     | :math:`A`                        | Rotational constant                                                               | :math:`\textrm{GHz}`                        |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 17     | :math:`B`                        | Rotational constant                                                               | :math:`\textrm{GHz}`                        |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 18     | :math:`C`                        | Rotational constant                                                               | :math:`\textrm{GHz}`                        |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+

    Args:
        root (string): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)

    Stats:
        .. list-table::
            :widths: 10 10 10 10 10
            :header-rows: 1

            * - #graphs
              - #nodes
              - #edges
              - #features
              - #tasks
            * - 130,831
              - ~18.0
              - ~37.3
              - 11
              - 19
    """  # noqa: E501

    raw_url = ('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/'
               'molnet_publish/qm9.zip')
    raw_url2 = 'https://ndownloader.figshare.com/files/3195404'
    processed_url = 'https://data.pyg.org/datasets/qm9_v3.zip'

    def __init__(self, root: str, processed_name: str = 'processed', transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        self.processed_name = processed_name
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def mean(self, target: int) -> float:
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return float(y[:, target].mean())

    def std(self, target: int) -> float:
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return float(y[:, target].std())

    def atomref(self, target) -> Optional[torch.Tensor]:
        if target in atomrefs:
            out = torch.zeros(100)
            out[torch.tensor([1, 6, 7, 8, 9])] = torch.tensor(atomrefs[target])
            return out.view(-1, 1)
        return None

    @property
    def raw_file_names(self) -> List[str]:
        try:
            import rdkit  # noqa
            return ['gdb9.sdf', 'gdb9.sdf.csv', 'uncharacterized.txt']
        except ImportError:
            return ['qm9_v3.pt']

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.processed_name)

    @property
    def processed_file_names(self) -> str:
        return 'data_v3.pt'


    def download(self):
        try:
            import rdkit  # noqa
            file_path = download_url(self.raw_url, self.raw_dir)
            extract_zip(file_path, self.raw_dir)
            os.unlink(file_path)

            file_path = download_url(self.raw_url2, self.raw_dir)
            os.rename(osp.join(self.raw_dir, '3195404'),
                      osp.join(self.raw_dir, 'uncharacterized.txt'))
        except ImportError:
            path = download_url(self.processed_url, self.raw_dir)
            extract_zip(path, self.raw_dir)
            os.unlink(path)

    def process(self):
        try:
            import rdkit
            from rdkit import Chem, RDLogger
            from rdkit.Chem.rdchem import BondType as BT
            from rdkit.Chem.rdchem import HybridizationType
            RDLogger.DisableLog('rdApp.*')

        except ImportError:
            rdkit = None

        if rdkit is None:
            print(("Using a pre-processed version of the dataset. Please "
                   "install 'rdkit' to alternatively process the raw data."),
                  file=sys.stderr)

            data_list = torch.load(self.raw_paths[0])
            data_list = [Data(**data_dict) for data_dict in data_list]

            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]

            torch.save(self.collate(data_list), self.processed_paths[0])
            return

        types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

        with open(self.raw_paths[1], 'r') as f:
            target = f.read().split('\n')[1:-1]
            target = [[float(x) for x in line.split(',')[1:20]]
                      for line in target]
            target = torch.tensor(target, dtype=torch.float)
            target = torch.cat([target[:, 3:], target[:, :3]], dim=-1)
            target = target * conversion.view(1, -1)

        with open(self.raw_paths[2], 'r') as f:
            skip = [int(x.split()[0]) - 1 for x in f.read().split('\n')[9:-2]]

        suppl = Chem.SDMolSupplier(self.raw_paths[0], removeHs=False,
                                   sanitize=False)

        data_list = []
        for i, mol in enumerate(tqdm(suppl)):
            if i in skip:
                continue

            N = mol.GetNumAtoms()

            conf = mol.GetConformer()
            pos = conf.GetPositions()
            pos = torch.tensor(pos, dtype=torch.float)

            type_idx = []
            atomic_number = []
            aromatic = []
            sp = []
            sp2 = []
            sp3 = []
            num_hs = []
            for atom in mol.GetAtoms():
                type_idx.append(types[atom.GetSymbol()])
                atomic_number.append(atom.GetAtomicNum())
                aromatic.append(1 if atom.GetIsAromatic() else 0)
                hybridization = atom.GetHybridization()
                sp.append(1 if hybridization == HybridizationType.SP else 0)
                sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
                sp3.append(1 if hybridization == HybridizationType.SP3 else 0)

            z = torch.tensor(atomic_number, dtype=torch.long)

            row, col, edge_type = [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start, end]
                col += [end, start]
                edge_type += 2 * [bonds[bond.GetBondType()]]

            edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_type = torch.tensor(edge_type, dtype=torch.long)
            edge_attr = F.one_hot(edge_type,
                                  num_classes=len(bonds)).to(torch.float)

            perm = (edge_index[0] * N + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_type = edge_type[perm]
            edge_attr = edge_attr[perm]

            row, col = edge_index
            hs = (z == 1).to(torch.float)
            num_hs = scatter(hs[row], col, dim_size=N).tolist()

            x1 = F.one_hot(torch.tensor(type_idx), num_classes=len(types))
            x2 = torch.tensor([atomic_number, aromatic, sp, sp2, sp3, num_hs],
                              dtype=torch.float).t().contiguous()
            x = torch.cat([x1.to(torch.float), x2], dim=-1)

            y = target[i].unsqueeze(0)
            name = mol.GetProp('_Name')

            # data = Data(x=x, z=z, pos=pos, edge_index=edge_index,
            #            edge_attr=edge_attr, y=y, name=name, idx=i)
            data = Data(x=torch.tensor(type_idx), pos=pos, edge_index=edge_index, edge_attr=edge_attr, y=y, name=name)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])


class TUDataset(InMemoryDataset):
    r"""A variety of graph kernel benchmark datasets, *.e.g.* "IMDB-BINARY",
    "REDDIT-BINARY" or "PROTEINS", collected from the `TU Dortmund University
    <https://chrsmrrs.github.io/datasets>`_.
    In addition, this dataset wrapper provides `cleaned dataset versions
    <https://github.com/nd7141/graph_datasets>`_ as motivated by the
    `"Understanding Isomorphism Bias in Graph Data Sets"
    <https://arxiv.org/abs/1910.12091>`_ paper, containing only non-isomorphic
    graphs.

    .. note::
        Some datasets may not come with any node labels.
        You can then either make use of the argument :obj:`use_node_attr`
        to load additional continuous node attributes (if present) or provide
        synthetic node features using transforms such as
        like :class:`torch_geometric.transforms.Constant` or
        :class:`torch_geometric.transforms.OneHotDegree`.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The `name
            <https://chrsmrrs.github.io/datasets/docs/datasets/>`_ of the
            dataset.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
        use_node_attr (bool, optional): If :obj:`True`, the dataset will
            contain additional continuous node attributes (if present).
            (default: :obj:`False`)
        use_edge_attr (bool, optional): If :obj:`True`, the dataset will
            contain additional continuous edge attributes (if present).
            (default: :obj:`False`)
        cleaned (bool, optional): If :obj:`True`, the dataset will
            contain only non-isomorphic graphs. (default: :obj:`False`)

    Stats:
        .. list-table::
            :widths: 20 10 10 10 10 10
            :header-rows: 1

            * - Name
              - #graphs
              - #nodes
              - #edges
              - #features
              - #classes
            * - MUTAG
              - 188
              - ~17.9
              - ~39.6
              - 7
              - 2
            * - ENZYMES
              - 600
              - ~32.6
              - ~124.3
              - 3
              - 6
            * - PROTEINS
              - 1,113
              - ~39.1
              - ~145.6
              - 3
              - 2
            * - COLLAB
              - 5,000
              - ~74.5
              - ~4914.4
              - 0
              - 3
            * - IMDB-BINARY
              - 1,000
              - ~19.8
              - ~193.1
              - 0
              - 2
            * - REDDIT-BINARY
              - 2,000
              - ~429.6
              - ~995.5
              - 0
              - 2
            * - ...
              -
              -
              -
              -
              -
    """

    url = 'https://www.chrsmrrs.com/graphkerneldatasets'
    cleaned_url = ('https://raw.githubusercontent.com/nd7141/'
                   'graph_datasets/master/datasets')

    def __init__(self, root: str, name: str, processed_name:str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 use_node_attr: bool = False, use_edge_attr: bool = False,
                 cleaned: bool = False):
        self.name = name
        self.cleaned = cleaned
        self.processed_name = processed_name
        super().__init__(root, transform, pre_transform, pre_filter)

        out = torch.load(self.processed_paths[0])
        if not isinstance(out, tuple) and len(out) != 3:
            raise RuntimeError(
                "The 'data' object was created by an older version of PyG. "
                "If this error occurred while loading an already existing "
                "dataset, remove the 'processed/' directory in the dataset's "
                "root folder and try again.")
        self.data, self.slices, self.sizes = out

        if self.data.x is not None and not use_node_attr:
            num_node_attributes = self.num_node_attributes
            self.data.x = self.data.x[:, num_node_attributes:]
        if self.data.edge_attr is not None and not use_edge_attr:
            num_edge_attributes = self.num_edge_attributes
            self.data.edge_attr = self.data.edge_attr[:, num_edge_attributes:]

    @property
    def raw_dir(self) -> str:
        name = f'raw{"_cleaned" if self.cleaned else ""}'
        return osp.join(self.root, self.name, name)

    @property
    def processed_dir(self) -> str:
        # name = f'processed{"_cleaned" if self.cleaned else ""}'
        name = self.processed_name + ("_cleaned" if self.cleaned else "")
        return osp.join(self.root, self.name, name)

    @property
    def num_node_labels(self) -> int:
        return self.sizes['num_node_labels']

    @property
    def num_node_attributes(self) -> int:
        return self.sizes['num_node_attributes']

    @property
    def num_edge_labels(self) -> int:
        return self.sizes['num_edge_labels']

    @property
    def num_edge_attributes(self) -> int:
        return self.sizes['num_edge_attributes']

    @property
    def raw_file_names(self) -> List[str]:
        names = ['A', 'graph_indicator']
        return [f'{self.name}_{name}.txt' for name in names]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        url = self.cleaned_url if self.cleaned else self.url
        folder = osp.join(self.root, self.name)
        path = download_url(f'{url}/{self.name}.zip', folder)
        extract_zip(path, folder)
        os.unlink(path)
        shutil.rmtree(self.raw_dir)
        os.rename(osp.join(folder, self.name), self.raw_dir)

    def process(self):
        self.data, self.slices = read_tu_data(self.raw_dir, self.name)
        if self.data.edge_attr == None:
            sizes = {'num_node_labels': self.num_features, 'num_node_attributes':self.num_features,
                 'num_edge_labels': 0,
                 'num_edge_attributes': 0}
        else:
            sizes = {'num_node_labels': self.num_features, 'num_node_attributes':self.num_features,
                      'num_edge_labels': self.data.edge_attr.size(-1),
                      'num_edge_attributes': self.data.edge_attr.size(-1)}
        if self.pre_filter is not None or self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]

            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]

            self.data, self.slices = self.collate(data_list)
            self._data_list = None  # Reset cache.

        torch.save((self.data, self.slices, sizes), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name}({len(self)})'



import pickle
class Chembl(InMemoryDataset):

    def __init__(self, root: str, processed_name: str = 'processed', transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        self.processed_name = processed_name
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ['chembl.pkl']

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.processed_name)

    @property
    def processed_file_names(self) -> str:
        return 'data_processed.pt'

    def process(self):
        try:
            import rdkit
            from rdkit import Chem, RDLogger
            from rdkit.Chem.rdchem import BondType as BT
            from rdkit.Chem.rdchem import HybridizationType
            RDLogger.DisableLog('rdApp.*')

        except ImportError:
            rdkit = None

        with open(self.raw_paths[0], 'rb') as f:
            smiles_list = pickle.load(f)


        data_list = []
        for i, sm in enumerate(smiles_list):
            if i % 500 == 0:
                print('Pre-processing: %d/%d' %(i, len(smiles_list)))
            mol = Chem.MolFromSmiles(sm)
            N = mol.GetNumAtoms()

            # x
            x = torch.zeros([N, ], dtype=torch.long)

            # edge
            row, col, edge_type = [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start, end]
                col += [end, start]
                edge_type += 2 * [1]

            edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_type = torch.tensor(edge_type, dtype=torch.long)
            edge_attr = torch.reshape(edge_type, [-1, 1]).to(torch.float)

            perm = (edge_index[0] * N + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_type = edge_type[perm]
            edge_attr = edge_attr[perm]

            data = Data(x=x, edge_index=edge_index,
                        edge_attr=edge_attr, name=sm)

            # calculate rings
            size_list = [3, 4, 5, 6, 7]
            ssr = Chem.GetSymmSSSR(mol)
            ssr = [list(s) for s in ssr]
            n_kring_graph = np.zeros([1, len(size_list)], dtype=np.int)
            n_kring_node = np.zeros((N, len(size_list)), dtype=np.int)
            for ring in ssr:
                size = len(ring)
                if size not in size_list:
                    continue
                # node level
                for atom in ring:
                    n_kring_node[atom, size_list.index(size)] += 1
                # graph level
                n_kring_graph[0, size_list.index(size)] += 1
            n_kring_graph = torch.tensor(n_kring_graph, dtype=torch.int)
            n_kring_node = torch.tensor(n_kring_node, dtype=torch.int)
            data.n_kring_graph = n_kring_graph
            # data.n_kring_node = n_kring_node
            data.y = n_kring_node.float()

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])