# This program is the pipeline for testing expressiveness.
# It includes 4 stages:
#   1. pre-calculation;
#   2. dataset construction;
#   3. model construction;
#   4. evaluation


import numpy as np
import torch
import torch_geometric
import torch_geometric.loader
from loguru import logger
import time
from BRECDataset_v3 import BRECDataset
from tqdm import tqdm
import os
from torch_geometric.nn.norm import BatchNorm, LayerNorm
from torch.nn import CosineEmbeddingLoss


import argparse
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from torch_geometric.utils import degree
from torch_geometric.nn import GCNConv, GINConv, global_add_pool
import torch_geometric.transforms as T

# from k_gnn import GraphConv, max_pool
# from k_gnn import TwoMalkin, ConnectedThreeMalkin
from dataloader import (
    DataLoader,
    DataListLoader,
)  # use a custom dataloader to handle subgraphs
from utils import create_subgraphs


# torch_geometric.seed_everything(2022)
NUM_RELABEL = 32
P_NORM = 2
OUTPUT_DIM = 16
EPSILON_MATRIX = 1e-7
EPSILON_CMP = 1e-6
SAMPLE_NUM = 400
EPOCH = 20
MARGIN = 0.0
LEARNING_RATE = 1e-5
THRESHOLD = 72.34
BATCH_SIZE = 32
WEIGHT_DECAY = 1e-4
LOSS_THRESHOLD = 0.01

global_var = globals().copy()
HYPERPARAM_DICT = dict()
for k, v in global_var.items():
    if isinstance(v, int) or isinstance(v, float):
        HYPERPARAM_DICT[k] = v

# part_dict: {graph generation type, range}
part_dict = {
    "Basic": (0, 60),
    "Regular": (60, 160),
    "Extension": (160, 260),
    "CFI": (260, 360),
    "4-Vertex_Condition": (360, 380),
    "Distance_Regular": (380, 400),
}
parser = argparse.ArgumentParser(description="Nested GNN for EXP/CEXP datasets")
parser.add_argument("--model", type=str, default="GIN")  # Base GNN used, GIN or GCN
parser.add_argument(
    "--h",
    type=int,
    default=1,
    help="largest height of rooted subgraphs to simulate",
)
parser.add_argument("--layers", type=int, default=4)  # Number of GNN layers
parser.add_argument("--width", type=int, default=32)  # Dimensionality of GNN embeddings
parser.add_argument("--node_label", type=str, default="hop")
parser.add_argument("--P_NORM", type=str, default="2")
parser.add_argument("--EPOCH", type=int, default=30)
parser.add_argument("--LEARNING_RATE", type=float, default=1e-2)
parser.add_argument("--BATCH_SIZE", type=int, default=32)
parser.add_argument("--WEIGHT_DECAY", type=float, default=1e-2)
parser.add_argument("--OUTPUT_DIM", type=int, default=16)
parser.add_argument("--SEED", type=int, default=2022)
parser.add_argument("--THRESHOLD", type=float, default=THRESHOLD)
parser.add_argument("--MARGIN", type=float, default=MARGIN)
parser.add_argument("--LOSS_THRESHOLD", type=float, default=LOSS_THRESHOLD)


args = parser.parse_args()
P_NORM = 2 if args.P_NORM == "2" else torch.inf
EPOCH = args.EPOCH
LEARNING_RATE = args.LEARNING_RATE
BATCH_SIZE = args.BATCH_SIZE
WEIGHT_DECAY = args.WEIGHT_DECAY
OUTPUT_DIM = args.OUTPUT_DIM
SEED = args.SEED
THRESHOLD = args.THRESHOLD
MARGIN = args.MARGIN
LOSS_THRESHOLD = args.LOSS_THRESHOLD
torch_geometric.seed_everything(SEED)
torch.backends.cudnn.deterministic = True
# torch.use_deterministic_algorithms(True)


class BasicGCN(torch.nn.Module):
    def __init__(self, num_layers, hidden):
        super(BasicGCN, self).__init__()
        self.conv1 = GCNConv(1, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
        self.lin1 = torch.nn.Linear(hidden, hidden)
        # self.lin2 = Linear(hidden, dataset.num_classes)
        self.lin2 = Linear(hidden, 1)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        edge_index = data.edge_index
        if "x" in data:
            x = data.x
        else:
            x = torch.ones([data.num_nodes, 1]).to(edge_index.device)
        x = self.conv1(x, edge_index)
        for conv in self.convs:
            x = conv(x, edge_index)
        x = global_add_pool(x, batch=None)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x
        # return F.log_softmax(x, dim=1)

    def __repr__(self):
        return self.__class__.__name__


class NestedGCN(torch.nn.Module):
    def __init__(self, num_layers, hidden):
        super(NestedGCN, self).__init__()
        # self.conv1 = GCNConv(dataset.num_features, hidden)
        self.conv1 = GCNConv(1, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
        self.lin1 = torch.nn.Linear(hidden, hidden)
        # self.lin2 = Linear(hidden, dataset.num_classes)
        self.lin2 = Linear(hidden, OUTPUT_DIM)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        # edge_index, batch = data.edge_index, data.batch
        edge_index = data.edge_index
        if "x" in data:
            x = data.x
        else:
            x = torch.ones([data.num_nodes, 1]).to(edge_index.device)
        x = self.conv1(x, edge_index)
        for conv in self.convs:
            x = conv(x, edge_index)

        x = global_add_pool(x, data.node_to_subgraph)
        x = global_add_pool(x, data.subgraph_to_graph)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        # return x
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


# NGNN model
# add one hot optiton for initial encoding
class NestedGIN(torch.nn.Module):
    def __init__(self, num_layers, hidden, one_hot=False, num_hop=1):
        super(NestedGIN, self).__init__()
        self.one_hot = one_hot
        self.num_hop = num_hop + 1
        if self.one_hot:
            # print("one_hot")
            initial_lin = Linear(self.num_hop, hidden)
        else:
            initial_lin = Linear(1, hidden)
        self.conv1 = GINConv(
            Sequential(
                initial_lin,
                ReLU(),
                Linear(hidden, hidden),
                ReLU(),
            ),
            train_eps=False,
        )
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GINConv(
                    Sequential(
                        Linear(hidden, hidden),
                        ReLU(),
                        Linear(hidden, hidden),
                        ReLU(),
                    ),
                    train_eps=False,
                )
            )
        self.lin1 = torch.nn.Linear(hidden, hidden)
        self.lin2 = Linear(hidden, OUTPUT_DIM)
        self.bn1 = BN(num_features=hidden, momentum=1.0, affine=False)
        self.bn2 = BN(num_features=OUTPUT_DIM, momentum=1.0, affine=False)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.bn1.reset_parameters()
        self.bn2.reset_parameters()
        # self.ln.reset_parameters()

    def forward(self, data):
        # edge_index, batch = data.edge_index, data.batch
        edge_index = data.edge_index
        if "z" in data:
            if self.one_hot:
                x = F.one_hot(data.z, num_classes=self.num_hop).squeeze().float()
            else:
                x = data.z.float()
        else:
            x = data.x
        # if "x" in data:
        #     x = data.x
        # elif "z" in data:
        #     if self.one_hot:
        #         x = F.one_hot(data.z, num_classes=self.num_hop).squeeze().float()
        #     else:
        #         x = data.z.float()
        # else:
        #     x = torch.ones([data.num_nodes, 1]).to(edge_index.device)
        x = self.conv1(x, edge_index)

        for conv in self.convs:
            x = conv(x, edge_index)

        x = global_add_pool(x, data.node_to_subgraph)
        x = global_add_pool(x, data.subgraph_to_graph)

        # print(x)
        # x = self.bn1(x)
        # print(x)
        x = F.relu(self.lin1(x))
        # x = F.dropout(x, p=0.1, training=self.training)
        x = self.lin2(x)
        x = self.bn2(x)
        return x
        # return F.softmax(x, dim=-1)
        # return F.log_softmax(x, dim=-1)


# Stage 1: pre calculation
# Here is for some calculation without data. e.g. generating all the k-substructures
def pre_calculation(*args, **kwargs):
    time_start = time.process_time()

    # Do something

    time_end = time.process_time()
    time_cost = round(time_end - time_start, 2)
    logger.info(f"pre-calculation time cost: {time_cost}")


# Stage 2: dataset construction
# Here is for dataset construction, including data processing
def get_dataset(name, pre_transform, device):
    time_start = time.process_time()

    # Do something
    def node_feature_transform(data):
        if "x" not in data:
            data.x = torch.ones([data.num_nodes, 1], dtype=torch.long)
        return data
    dataset = BRECDataset(name=name, pre_transform=pre_transform, transform=node_feature_transform)

    time_end = time.process_time()
    time_cost = round(time_end - time_start, 2)
    logger.info(f"dataset construction time cost: {time_cost}")

    return dataset


# Stage 3: model construction
# Here is for model construction.
def get_model(args, device):
    time_start = time.process_time()

    # Do something
    if args.model == "GIN":
        model = NestedGIN(
            args.layers,
            args.width,
            not (args.node_label == "no"),
            args.h,
        ).to(device)
    elif args.model == "GCN":
        model = NestedGCN(args.layers, args.width).to(device)
    elif args.model == "BasicGCN":
        model = BasicGCN(args.layers, args.width).to(device)
    else:
        raise NotImplementedError("model type not supported")
    model.reset_parameters()

    time_end = time.process_time()
    time_cost = round(time_end - time_start, 2)
    logger.info(f"model construction time cost: {time_cost}")
    return model


# Stage 4: evaluation
# Here is for evaluation.
def evaluation(dataset, model, path, device, args):
    def T2_calculation(dataset, log_flag=False):
        with torch.no_grad():
            loader = DataLoader(dataset, batch_size=BATCH_SIZE)
            pred_0_list = []
            pred_1_list = []
            for data in loader:
                pred = model(data.to(device)).detach()
                pred_0_list.extend(pred[0::2])
                pred_1_list.extend(pred[1::2])
            X = torch.cat([x.reshape(1, -1) for x in pred_0_list], dim=0).T
            Y = torch.cat([x.reshape(1, -1) for x in pred_1_list], dim=0).T
            if log_flag:
                logger.info(f"X_mean = {torch.mean(X, dim=1)}")
                logger.info(f"Y_mean = {torch.mean(Y, dim=1)}")
            D = X - Y
            D_mean = torch.mean(D, dim=1).reshape(-1, 1)
            S = torch.cov(D)
            inv_S = torch.linalg.pinv(S)
            return torch.mm(torch.mm(D_mean.T, inv_S), D_mean)

    time_start = time.process_time()

    # Do something
    cnt = 0
    correct_list = []
    fail_in_reliability = 0
    loss_func = CosineEmbeddingLoss(margin=MARGIN)

    for part_name, part_range in part_dict.items():
        logger.info(f"{part_name} part starting ---")

        cnt_part = 0
        fail_in_reliability_part = 0
        start = time.process_time()

        for id in tqdm(range(part_range[0], part_range[1])):
            logger.info(f"ID: {id}")
            optimizer = torch.optim.Adam(
                model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
            dataset_traintest = dataset[
                id * NUM_RELABEL * 2 : (id + 1) * NUM_RELABEL * 2
            ]
            dataset_reliability = dataset[
                (id + SAMPLE_NUM)
                * NUM_RELABEL
                * 2 : (id + SAMPLE_NUM + 1)
                * NUM_RELABEL
                * 2
            ]
            model.reset_parameters()
            model.train()
            for _ in range(EPOCH):
                traintest_loader = DataLoader(dataset_traintest, batch_size=BATCH_SIZE)
                loss_all = 0
                for data in traintest_loader:
                    optimizer.zero_grad()
                    pred = model(data.to(device))
                    loss = loss_func(
                        pred[0::2],
                        pred[1::2],
                        torch.tensor([-1] * (len(pred) // 2)).to(device),
                    )
                    loss.backward()
                    optimizer.step()
                    loss_all += len(pred) / 2 * loss.item()
                loss_all /= NUM_RELABEL
                logger.info(f"Loss: {loss_all}")
                if loss_all < LOSS_THRESHOLD:
                    logger.info("Early Stop Here")
                    break
                scheduler.step(loss_all)

            model.eval()
            T_square_traintest = T2_calculation(dataset_traintest, True)
            T_square_reliability = T2_calculation(dataset_reliability, True)

            isomorphic_flag = False
            reliability_flag = False
            if T_square_traintest > THRESHOLD and not torch.isclose(
                T_square_traintest, T_square_reliability, atol=EPSILON_CMP
            ):
                isomorphic_flag = True
            if T_square_reliability < THRESHOLD:
                reliability_flag = True

            if isomorphic_flag:
                cnt += 1
                cnt_part += 1
                correct_list.append(id)
                logger.info(f"Correct num in current part: {cnt_part}")
            if not reliability_flag:
                fail_in_reliability += 1
                fail_in_reliability_part += 1
            logger.info(f"isomorphic: {isomorphic_flag} {T_square_traintest}")
            logger.info(f"reliability: {reliability_flag} {T_square_reliability}")

        end = time.process_time()
        time_cost_part = round(end - start, 2)

        logger.info(
            f"{part_name} part costs time {time_cost_part}; Correct in {cnt_part} / {part_range[1] - part_range[0]}"
        )
        logger.info(
            f"Fail in reliability: {fail_in_reliability_part} / {part_range[1] - part_range[0]}"
        )

    time_end = time.process_time()
    time_cost = round(time_end - time_start, 2)
    logger.info(f"evaluation time cost: {time_cost}")

    Acc = round(cnt / SAMPLE_NUM, 2)
    logger.info(f"Correct in {cnt} / {SAMPLE_NUM}, Acc = {Acc}")

    logger.info(f"Fail in reliability: {fail_in_reliability} / {SAMPLE_NUM}")
    logger.info(correct_list)

    logger.add(f"{path}/result_show.txt", format="{message}", encoding="utf-8")
    logger.info(
        "Real_correct\tCorrect\tFail\th\tlayers\twidth\tOUTPUT_DIM\tBATCH_SIZE\tLEARNING_RATE\tWEIGHT_DECAY\tSEED"
    )
    logger.info(
        f"{cnt-fail_in_reliability}\t{cnt}\t{fail_in_reliability}\t{args.h}\t{args.layers}\t{args.width}\t{OUTPUT_DIM}\t{BATCH_SIZE}\t{LEARNING_RATE}\t{WEIGHT_DECAY}\t{SEED}"
    )


def main():
    # Command Line Arguments
    LAYERS = args.layers
    WIDTH = args.width

    MODEL = f"Nested{args.model}-"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    pre_transform = None
    if args.h is not None:

        def pre_transform(g):
            return create_subgraphs(
                g,
                args.h,
                node_label=args.node_label,
                use_rd=False,
                subgraph_pretransform=None,
            )

    OUT_PATH = "result_BREC"
    NAME = f"{MODEL}h={args.h}_layer={LAYERS}_hidden={WIDTH}_{args.node_label}"
    DATASET_NAME = f"h={args.h}_{args.node_label}"
    path = os.path.join(OUT_PATH, NAME)
    os.makedirs(path, exist_ok=True)

    logger.remove(handler_id=None)
    LOG_NAME = os.path.join(path, "log.txt")
    logger.add(LOG_NAME, rotation="5MB")

    logger.info(HYPERPARAM_DICT)
    logger.info(args)

    pre_calculation()
    dataset = get_dataset(name=DATASET_NAME, pre_transform=pre_transform, device=device)
    model = get_model(args, device)
    evaluation(dataset, model, OUT_PATH, device, args)


if __name__ == "__main__":
    main()
