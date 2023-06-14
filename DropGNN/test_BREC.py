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
from torch.nn import CosineEmbeddingLoss
import argparse

from torch import nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader, Data
from torch_geometric.utils import degree
from torch_geometric.utils.convert import from_networkx
from torch_geometric.nn import GINConv, GINEConv, global_add_pool
import torch_geometric.transforms as T


NUM_RELABEL = 32
P_NORM = 2
OUTPUT_DIM = 16
EPSILON_MATRIX = 1e-7
EPSILON_CMP = 1e-6
SAMPLE_NUM = 400
EPOCH = 50
MARGIN = 0.0
LEARNING_RATE = 1e-3
THRESHOLD = 5.0
BATCH_SIZE = 16
WEIGHT_DECAY = 1e-5
LOSS_THRESHOLD = 0.00
SEED = 2023

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
parser = argparse.ArgumentParser(description="BREC Test")

parser.add_argument("--P_NORM", type=str, default="2")
parser.add_argument("--EPOCH", type=int, default=EPOCH)
parser.add_argument("--LEARNING_RATE", type=float, default=LEARNING_RATE)
parser.add_argument("--BATCH_SIZE", type=int, default=BATCH_SIZE)
parser.add_argument("--WEIGHT_DECAY", type=float, default=WEIGHT_DECAY)
parser.add_argument("--OUTPUT_DIM", type=int, default=OUTPUT_DIM)
parser.add_argument("--SEED", type=int, default=SEED)
parser.add_argument("--THRESHOLD", type=float, default=THRESHOLD)
parser.add_argument("--MARGIN", type=float, default=MARGIN)
parser.add_argument("--LOSS_THRESHOLD", type=float, default=LOSS_THRESHOLD)
parser.add_argument("--device", type=int, default=0)
parser.add_argument(
    "--augmentation",
    type=str,
    default="none",
    help="Options are ['none', 'ports', 'ids', 'random', 'dropout']",
)
parser.add_argument("--prob", type=int, default=-1)
parser.add_argument("--num_runs", type=int, default=50)
parser.add_argument(
    "--num_layers", type=int, default=4
)  # 9 layers were used for skipcircles dataset
parser.add_argument("--use_aux_loss", action="store_true", default=False, help='Not Supported Now!')
parser.add_argument("--hidden_units", type=int, default=32)

# General settings.
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
def get_dataset(name, device):
    time_start = time.process_time()

    # Do something
    def makefeatures(data):
        data.x = torch.ones((data.num_nodes, 1))
        data.id = torch.tensor(
            np.random.permutation(np.arange(data.num_nodes))
        ).unsqueeze(1)
        return data

    def addports(data):
        data.ports = torch.zeros(data.num_edges, 1)
        degs = degree(
            data.edge_index[0], data.num_nodes, dtype=torch.long
        )  # out degree of all nodes
        for n in range(data.num_nodes):
            deg = degs[n]
            ports = np.random.permutation(int(deg))
            for i, neighbor in enumerate(data.edge_index[1][data.edge_index[0] == n]):
                nb = int(neighbor)
                data.ports[
                    torch.logical_and(
                        data.edge_index[0] == n, data.edge_index[1] == nb
                    ),
                    0,
                ] = float(ports[i])
        return data

    pre_transform = T.Compose([makefeatures, addports])

    dataset = BRECDataset(name=name, pre_transform=pre_transform)
    time_end = time.process_time()
    time_cost = round(time_end - time_start, 2)
    logger.info(f"dataset construction time cost: {time_cost}")

    return dataset


# Stage 3: model construction
# Here is for model construction.
def get_model(args, num_nodes, num_features, device):
    time_start = time.process_time()
    # Do something

    n = num_nodes
    gamma = n
    p_opt = 2 * 1 / (1 + gamma)
    if args.prob >= 0:
        p = args.prob
    else:
        p = p_opt
    if args.num_runs > 0:
        num_runs = args.num_runs
    else:
        num_runs = gamma

    graph_classification = True
    num_features = num_features
    Conv = GINConv
    if args.augmentation == "ports":
        Conv = GINEConv
    elif args.augmentation == "ids":
        num_features += 1
    elif args.augmentation == "random":
        num_features += 1
    use_aux_loss = args.use_aux_loss

    class GIN(nn.Module):
        def __init__(self):
            super(GIN, self).__init__()

            dim = args.hidden_units

            self.num_layers = args.num_layers

            self.convs = nn.ModuleList()
            self.bns = nn.ModuleList()
            self.fcs = nn.ModuleList()

            self.convs.append(
                Conv(
                    nn.Sequential(
                        nn.Linear(num_features, dim),
                        nn.BatchNorm1d(dim),
                        nn.ReLU(),
                        nn.Linear(dim, dim),
                    )
                )
            )
            self.bns.append(nn.BatchNorm1d(dim))
            self.fcs.append(nn.Linear(num_features, OUTPUT_DIM))
            self.fcs.append(nn.Linear(dim, OUTPUT_DIM))

            for i in range(self.num_layers - 1):
                self.convs.append(
                    Conv(
                        nn.Sequential(
                            nn.Linear(dim, dim),
                            nn.BatchNorm1d(dim),
                            nn.ReLU(),
                            nn.Linear(dim, dim),
                        )
                    )
                )
                self.bns.append(nn.BatchNorm1d(dim))
                self.fcs.append(nn.Linear(dim, OUTPUT_DIM))

        def reset_parameters(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    m.reset_parameters()
                elif isinstance(m, Conv):
                    m.reset_parameters()
                elif isinstance(m, nn.BatchNorm1d):
                    m.reset_parameters()

        def forward(self, data):
            x = data.x
            edge_index = data.edge_index
            batch = data.batch

            if args.augmentation == "ids":
                x = torch.cat([x, data.id.float()], dim=1)
            elif args.augmentation == "random":
                x = torch.cat(
                    [x, torch.randint(0, 100, (x.size(0), 1), device=x.device) / 100.0],
                    dim=1,
                )

            outs = [x]
            for i in range(self.num_layers):
                if args.augmentation == "ports":
                    x = self.convs[i](x, edge_index, data.ports.expand(-1, x.size(-1)))
                else:
                    x = self.convs[i](x, edge_index)
                x = self.bns[i](x)
                x = F.relu(x)
                outs.append(x)

            out = None
            for i, x in enumerate(outs):
                if graph_classification:
                    x = global_add_pool(x, batch)
                x = self.fcs[i](x)  # No dropout for these experiments
                if out is None:
                    out = x
                else:
                    out += x
            return out
            # return F.log_softmax(out, dim=-1), 0

    class DropGIN(nn.Module):
        def __init__(self):
            super(DropGIN, self).__init__()

            dim = args.hidden_units

            self.num_layers = args.num_layers

            self.convs = nn.ModuleList()
            self.bns = nn.ModuleList()
            self.fcs = nn.ModuleList()

            self.convs.append(
                Conv(
                    nn.Sequential(
                        nn.Linear(num_features, dim),
                        nn.BatchNorm1d(dim),
                        nn.ReLU(),
                        nn.Linear(dim, dim),
                    )
                )
            )
            self.bns.append(nn.BatchNorm1d(dim))
            self.fcs.append(nn.Linear(num_features, OUTPUT_DIM))
            self.fcs.append(nn.Linear(dim, OUTPUT_DIM))

            for i in range(self.num_layers - 1):
                self.convs.append(
                    Conv(
                        nn.Sequential(
                            nn.Linear(dim, dim),
                            nn.BatchNorm1d(dim),
                            nn.ReLU(),
                            nn.Linear(dim, dim),
                        )
                    )
                )
                self.bns.append(nn.BatchNorm1d(dim))
                self.fcs.append(nn.Linear(dim, OUTPUT_DIM))

            if use_aux_loss:
                self.aux_fcs = nn.ModuleList()
                self.aux_fcs.append(nn.Linear(num_features, OUTPUT_DIM))
                for i in range(self.num_layers):
                    self.aux_fcs.append(nn.Linear(dim, OUTPUT_DIM))

        def reset_parameters(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    m.reset_parameters()
                elif isinstance(m, Conv):
                    m.reset_parameters()
                elif isinstance(m, nn.BatchNorm1d):
                    m.reset_parameters()

        def forward(self, data):
            x = data.x
            edge_index = data.edge_index
            batch = data.batch

            # Do runs in paralel, by repeating the graphs in the batch
            x = x.unsqueeze(0).expand(num_runs, -1, -1).clone()
            drop = torch.bernoulli(
                torch.ones([x.size(0), x.size(1)], device=x.device) * p
            ).bool()
            x[drop] = 0.0
            del drop
            outs = [x]
            x = x.view(-1, x.size(-1))
            run_edge_index = edge_index.repeat(1, num_runs) + torch.arange(
                num_runs, device=edge_index.device
            ).repeat_interleave(edge_index.size(1)) * (edge_index.max() + 1)
            for i in range(self.num_layers):
                x = self.convs[i](x, run_edge_index)
                x = self.bns[i](x)
                x = F.relu(x)
                outs.append(x.view(num_runs, -1, x.size(-1)))
            del run_edge_index

            out = None
            for i, x in enumerate(outs):
                x = x.mean(dim=0)
                if graph_classification:
                    x = global_add_pool(x, batch)
                x = self.fcs[i](x)  # No dropout layer in these experiments
                if out is None:
                    out = x
                else:
                    out += x

            if use_aux_loss:
                aux_out = torch.zeros(
                    num_runs, out.size(0), out.size(1), device=out.device
                )
                run_batch = batch.repeat(num_runs) + torch.arange(
                    num_runs, device=edge_index.device
                ).repeat_interleave(batch.size(0)) * (batch.max() + 1)
                for i, x in enumerate(outs):
                    if graph_classification:
                        x = x.view(-1, x.size(-1))
                        x = global_add_pool(x, run_batch)
                    x = x.view(num_runs, -1, x.size(-1))
                    x = self.aux_fcs[i](x)  # No dropout layer in these experiments
                    aux_out += x

                return out, aux_out
                # return F.log_softmax(out, dim=-1), F.log_softmax(aux_out, dim=-1)
            else:
                return out
                # return F.log_softmax(out, dim=-1), 0

    if args.augmentation == "dropout":
        model = DropGIN().to(device)
    else:
        model = GIN().to(device)
        use_aux_loss = False

    time_end = time.process_time()
    time_cost = round(time_end - time_start, 2)
    logger.info(f"model construction time cost: {time_cost}")
    return model


# Stage 4: evaluation
# Here is for evaluation.
def evaluation(dataset, path, device, args):
    """
    When testing on BREC, even on the same graph, the output embedding may be different,
    because numerical precision problem occur on large graphs, and even the same graph is permuted.
    However, if you want to test on some simple graphs without permutation outputting the exact same embedding,
    some modification is needed to avoid computing the inverse matrix of a zero matrix.
    """
    # If you want to test on some simple graphs without permutation outputting the exact same embedding, please use S_epsilon.
    # S_epsilon = torch.diag(
    #     torch.full(size=(OUTPUT_DIM, 1), fill_value=EPSILON_MATRIX).reshape(-1)
    # ).to(device)
    def T2_calculation(dataset, log_flag=False):
        with torch.no_grad():
            loader = torch_geometric.loader.DataLoader(dataset, batch_size=BATCH_SIZE)
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
            # If you want to test on some simple graphs without permutation outputting the exact same embedding, please use inv_S with S_epsilon.
            # inv_S = torch.linalg.pinv(S + S_epsilon)
            return torch.mm(torch.mm(D_mean.T, inv_S), D_mean)

    time_start = time.process_time()

    # Do something
    num_nodes_list = np.load('num_node.npy', allow_pickle=True)
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
            model = get_model(args, num_nodes_list[id], 1, device)
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
            model.train()
            for _ in range(EPOCH):
                traintest_loader = torch_geometric.loader.DataLoader(
                    dataset_traintest, batch_size=BATCH_SIZE
                )
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
        "Real_correct\tCorrect\tFail\tnum_layers\thidden_units\tnum_runs\tOUTPUT_DIM\tBATCH_SIZE\tLEARNING_RATE\tWEIGHT_DECAY\tSEED"
    )
    logger.info(
        f"{cnt-fail_in_reliability}\t{cnt}\t{fail_in_reliability}\t{args.num_layers}\t{args.hidden_units}\t{args.num_runs}\t{OUTPUT_DIM}\t{BATCH_SIZE}\t{LEARNING_RATE}\t{WEIGHT_DECAY}\t{SEED}"
    )


def main():
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    OUT_PATH = "result_BREC"
    NAME = args.augmentation
    DATASET_NAME = "no_param"
    path = os.path.join(OUT_PATH, NAME)
    os.makedirs(path, exist_ok=True)

    logger.remove(handler_id=None)
    LOG_NAME = os.path.join(path, "log.txt")
    logger.add(LOG_NAME, rotation="5MB")

    logger.info(args)

    pre_calculation()
    dataset = get_dataset(name=DATASET_NAME, device=device)
    # model = get_model(args, device)
    evaluation(dataset, OUT_PATH, device, args)


if __name__ == "__main__":
    main()
