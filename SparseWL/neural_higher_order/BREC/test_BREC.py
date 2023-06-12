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
from BRECDataset_v3_3wl import BRECDataset
from tqdm import tqdm
import os
from torch.nn import CosineEmbeddingLoss
import argparse

from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, Set2Set
from torch_geometric.data import InMemoryDataset, Data, DataLoader
import torch.nn.functional as F

NUM_RELABEL = 32
P_NORM = 2
OUTPUT_DIM = 16
EPSILON_MATRIX = 1e-7
EPSILON_CMP = 1e-6
SAMPLE_NUM = 270
EPOCH = 20
MARGIN = 0.0
LEARNING_RATE = 1e-4
THRESHOLD = 72.34
BATCH_SIZE = 16
WEIGHT_DECAY = 1e-4
LOSS_THRESHOLD = 0.2
SEED = 2023

global_var = globals().copy()
HYPERPARAM_DICT = dict()
for k, v in global_var.items():
    if isinstance(v, int) or isinstance(v, float):
        HYPERPARAM_DICT[k] = v

# part_dict: {graph generation type, range}
part_dict = {
    "Basic": (0, 60),
    "Regular": (60, 110),
    "Extension": (110, 210),
    "CFI": (210, 270),
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
parser.add_argument("--device", type=int, default=2)
parser.add_argument("--dim", type=int, default=16)

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
    class MyData(Data):
        def __inc__(self, key, value, *args, **kwargs):
            return self.num_nodes if key in ["edge_index_1", "edge_index_2"] else 0

    class MyTransform(object):
        def __call__(self, data):
            new_data = MyData()
            for key, item in data:
                new_data[key] = item
            return new_data

    time_start = time.process_time()

    # Do something
    dataset = BRECDataset(name=name, transform=MyTransform())

    time_end = time.process_time()
    time_cost = round(time_end - time_start, 2)
    logger.info(f"dataset construction time cost: {time_cost}")

    return dataset


class NetGIN(torch.nn.Module):
    def __init__(self, dim):
        super(NetGIN, self).__init__()

        num_features = 3

        nn1_1 = Sequential(
            Linear(num_features, dim),
            torch.nn.BatchNorm1d(dim),
            ReLU(),
            Linear(dim, dim),
            torch.nn.BatchNorm1d(dim),
            ReLU(),
        )
        nn1_2 = Sequential(
            Linear(num_features, dim),
            torch.nn.BatchNorm1d(dim),
            ReLU(),
            Linear(dim, dim),
            torch.nn.BatchNorm1d(dim),
            ReLU(),
        )
        self.conv1_1 = GINConv(nn1_1, train_eps=True)
        self.conv1_2 = GINConv(nn1_2, train_eps=True)
        self.mlp_1 = Sequential(
            Linear(2 * dim, dim),
            torch.nn.BatchNorm1d(dim),
            ReLU(),
            Linear(dim, dim),
            torch.nn.BatchNorm1d(dim),
            ReLU(),
        )

        nn2_1 = Sequential(
            Linear(dim, dim),
            torch.nn.BatchNorm1d(dim),
            ReLU(),
            Linear(dim, dim),
            torch.nn.BatchNorm1d(dim),
            ReLU(),
        )
        nn2_2 = Sequential(
            Linear(dim, dim),
            torch.nn.BatchNorm1d(dim),
            ReLU(),
            Linear(dim, dim),
            torch.nn.BatchNorm1d(dim),
            ReLU(),
        )
        self.conv2_1 = GINConv(nn2_1, train_eps=True)
        self.conv2_2 = GINConv(nn2_2, train_eps=True)
        self.mlp_2 = Sequential(
            Linear(2 * dim, dim),
            torch.nn.BatchNorm1d(dim),
            ReLU(),
            Linear(dim, dim),
            torch.nn.BatchNorm1d(dim),
            ReLU(),
        )

        nn3_1 = Sequential(
            Linear(dim, dim),
            torch.nn.BatchNorm1d(dim),
            ReLU(),
            Linear(dim, dim),
            torch.nn.BatchNorm1d(dim),
            ReLU(),
        )
        nn3_2 = Sequential(
            Linear(dim, dim),
            torch.nn.BatchNorm1d(dim),
            ReLU(),
            Linear(dim, dim),
            torch.nn.BatchNorm1d(dim),
            ReLU(),
        )
        self.conv3_1 = GINConv(nn3_1, train_eps=True)
        self.conv3_2 = GINConv(nn3_2, train_eps=True)
        self.mlp_3 = Sequential(
            Linear(2 * dim, dim),
            torch.nn.BatchNorm1d(dim),
            ReLU(),
            Linear(dim, dim),
            torch.nn.BatchNorm1d(dim),
            ReLU(),
        )

        nn4_1 = Sequential(
            Linear(dim, dim),
            torch.nn.BatchNorm1d(dim),
            ReLU(),
            Linear(dim, dim),
            torch.nn.BatchNorm1d(dim),
            ReLU(),
        )
        nn4_2 = Sequential(
            Linear(dim, dim),
            torch.nn.BatchNorm1d(dim),
            ReLU(),
            Linear(dim, dim),
            torch.nn.BatchNorm1d(dim),
            ReLU(),
        )
        self.conv4_1 = GINConv(nn4_1, train_eps=True)
        self.conv4_2 = GINConv(nn4_2, train_eps=True)
        self.mlp_4 = Sequential(
            Linear(2 * dim, dim),
            torch.nn.BatchNorm1d(dim),
            ReLU(),
            Linear(dim, dim),
            torch.nn.BatchNorm1d(dim),
            ReLU(),
        )

        nn5_1 = Sequential(
            Linear(dim, dim),
            torch.nn.BatchNorm1d(dim),
            ReLU(),
            Linear(dim, dim),
            torch.nn.BatchNorm1d(dim),
            ReLU(),
        )
        nn5_2 = Sequential(
            Linear(dim, dim),
            torch.nn.BatchNorm1d(dim),
            ReLU(),
            Linear(dim, dim),
            torch.nn.BatchNorm1d(dim),
            ReLU(),
        )
        self.conv5_1 = GINConv(nn5_1, train_eps=True)
        self.conv5_2 = GINConv(nn5_2, train_eps=True)
        self.mlp_5 = Sequential(
            Linear(2 * dim, dim),
            torch.nn.BatchNorm1d(dim),
            ReLU(),
            Linear(dim, dim),
            torch.nn.BatchNorm1d(dim),
            ReLU(),
        )

        nn6_1 = Sequential(
            Linear(dim, dim),
            torch.nn.BatchNorm1d(dim),
            ReLU(),
            Linear(dim, dim),
            torch.nn.BatchNorm1d(dim),
            ReLU(),
        )
        nn6_2 = Sequential(
            Linear(dim, dim),
            torch.nn.BatchNorm1d(dim),
            ReLU(),
            Linear(dim, dim),
            torch.nn.BatchNorm1d(dim),
            ReLU(),
        )
        self.conv6_1 = GINConv(nn6_1, train_eps=True)
        self.conv6_2 = GINConv(nn6_2, train_eps=True)
        self.mlp_6 = Sequential(
            Linear(2 * dim, dim),
            torch.nn.BatchNorm1d(dim),
            ReLU(),
            Linear(dim, dim),
            torch.nn.BatchNorm1d(dim),
            ReLU(),
        )
        self.set2set = Set2Set(1 * dim, processing_steps=6)
        self.fc1 = Linear(2 * dim, dim)
        self.fc4 = Linear(dim, OUTPUT_DIM)

    def forward(self, data):
        x = data.x

        x_1 = F.relu(self.conv1_1(x, data.edge_index_1))
        x_2 = F.relu(self.conv1_2(x, data.edge_index_2))
        x_1_r = self.mlp_1(torch.cat([x_1, x_2], dim=-1))

        x_1 = F.relu(self.conv2_1(x_1_r, data.edge_index_1))
        x_2 = F.relu(self.conv2_2(x_1_r, data.edge_index_2))
        x_2_r = self.mlp_2(torch.cat([x_1, x_2], dim=-1))

        x_1 = F.relu(self.conv3_1(x_2_r, data.edge_index_1))
        x_2 = F.relu(self.conv3_2(x_2_r, data.edge_index_2))
        x_3_r = self.mlp_3(torch.cat([x_1, x_2], dim=-1))

        x_1 = F.relu(self.conv4_1(x_3_r, data.edge_index_1))
        x_2 = F.relu(self.conv4_2(x_3_r, data.edge_index_2))
        x_4_r = self.mlp_4(torch.cat([x_1, x_2], dim=-1))

        x_1 = F.relu(self.conv5_1(x_4_r, data.edge_index_1))
        x_2 = F.relu(self.conv5_2(x_4_r, data.edge_index_2))
        x_5_r = self.mlp_5(torch.cat([x_1, x_2], dim=-1))

        x_1 = F.relu(self.conv6_1(x_5_r, data.edge_index_1))
        x_2 = F.relu(self.conv6_2(x_5_r, data.edge_index_2))
        x_6_r = self.mlp_6(torch.cat([x_1, x_2], dim=-1))

        x = x_6_r

        x = self.set2set(x, data.batch)

        x = F.relu(self.fc1(x))
        x = self.fc4(x)
        return x


# Stage 3: model construction
# Here is for model construction.
def get_model(args, device):
    time_start = time.process_time()

    # Do something
    model = NetGIN(args.dim).to(device)

    time_end = time.process_time()
    time_cost = round(time_end - time_start, 2)
    logger.info(f"model construction time cost: {time_cost}")
    return model


# Stage 4: evaluation
# Here is for evaluation.
def evaluation(dataset, model, path, device, args):
    # If you want to test on some simple graphs without permutation outputting the exact same embedding, please use S_epsilon.
    # S_epsilon = torch.diag(
    #     torch.full(size=(OUTPUT_DIM, 1), fill_value=EPSILON_MATRIX).reshape(-1)
    # ).to(device)
    def cov(X):
        D = X.shape[-1]
        mean = torch.mean(X, dim=-1).unsqueeze(-1)
        X = X - mean
        return torch.matmul(X, X.T) / (D - 1)
        # return 1 / (D - 1) * X @ X.transpose(-1, -2)

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
            S = cov(D)
            inv_S = torch.linalg.pinv(S)
            # If you want to test on some simple graphs without permutation outputting the exact same embedding, please use inv_S with S_epsilon.
            # inv_S = torch.linalg.pinv(S + S_epsilon)
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
            model = get_model(args, device)
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
        "Real_correct\tCorrect\tFail\tdim\tOUTPUT_DIM\tBATCH_SIZE\tLEARNING_RATE\tWEIGHT_DECAY\tSEED"
    )
    logger.info(
        f"{cnt-fail_in_reliability}\t{cnt}\t{fail_in_reliability}\t{args.dim}\t{OUTPUT_DIM}\t{BATCH_SIZE}\t{LEARNING_RATE}\t{WEIGHT_DECAY}\t{SEED}"
    )


def main():
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    OUT_PATH = "result_BREC"
    NAME = f"dlgnn_{args.dim}"
    DATASET_NAME = "no_param"
    path = os.path.join(OUT_PATH, NAME)
    os.makedirs(path, exist_ok=True)

    logger.remove(handler_id=None)
    LOG_NAME = os.path.join(path, "log.txt")
    logger.add(LOG_NAME, rotation="5MB")

    logger.info(args)

    pre_calculation()
    dataset = get_dataset(name=DATASET_NAME, device=device)
    model = get_model(args, device)
    evaluation(dataset, model, OUT_PATH, device, args)


if __name__ == "__main__":
    main()
