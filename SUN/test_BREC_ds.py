# This program is the pipeline for testing expressiveness.
# It includes 4 stages:
#   1. pre-calculation;
#   2. dataset construction;
#   3. model construction;
#   4. evaluation


import numpy as np
import torch
import torch_geometric
from torch_geometric.data import DataLoader
from loguru import logger
import time
from BRECDataset_v3 import BRECDataset
from tqdm import tqdm
import os
import argparse
import torch.nn.functional as F
from torch.nn import CosineEmbeddingLoss

from json import dumps
from utils import (
    get_model,
)
from data import (
    policy2transform,
)


NUM_RELABEL = 32
P_NORM = 2
OUTPUT_DIM = 16
EPSILON_MATRIX = 1e-8
EPSILON_CMP = 1e-6
SAMPLE_NUM = 400
EPOCH = 30
MARGIN = 0.0
LEARNING_RATE = 1e-4
THRESHOLD = 72.34
BATCH_SIZE = 16
WEIGHT_DECAY = 1e-5
LOSS_THRESHOLD = 0.0


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


parser = argparse.ArgumentParser("arguments for training and testing")

parser.add_argument("--P_NORM", type=str, default="2")
parser.add_argument("--EPOCH", type=int, default=EPOCH)
parser.add_argument("--LEARNING_RATE", type=float, default=LEARNING_RATE)
parser.add_argument("--BATCH_SIZE", type=int, default=BATCH_SIZE)
parser.add_argument("--WEIGHT_DECAY", type=float, default=WEIGHT_DECAY)
parser.add_argument("--OUTPUT_DIM", type=int, default=16)
parser.add_argument("--SEED", type=int, default=2022)
parser.add_argument("--THRESHOLD", type=float, default=THRESHOLD)
parser.add_argument("--MARGIN", type=float, default=MARGIN)
parser.add_argument("--LOSS_THRESHOLD", type=float, default=LOSS_THRESHOLD)

parser.add_argument(
    "--device", type=int, default=1, help="which gpu to use if any (default: 1)"
)
parser.add_argument(
    "--gnn_type",
    type=str,
    help="Type of convolution {gin, originalgin, zincgin, graphconv}",
)
parser.add_argument(
    "--random_ratio",
    type=float,
    default=0.0,
    help="Number of random features, > 0 only for RNI",
)
parser.add_argument("--model", type=str, help="Type of model {deepsets, dss, gnn}")
parser.add_argument(
    "--drop_ratio", type=float, default=0.5, help="dropout ratio (default: 0.5)"
)
parser.add_argument(
    "--num_layer",
    type=int,
    default=5,
    help="number of GNN message passing layers (default: 5)",
)
parser.add_argument(
    "--channels",
    type=str,
    default="64-64",
    help='String with dimension of each DS layer, separated by "-"'
    "(considered only if args.model is deepsets)",
)
parser.add_argument(
    "--emb_dim",
    type=int,
    default=32,
    help="dimensionality of hidden units in GNNs (default: 32)",
)
parser.add_argument(
    "--jk",
    type=str,
    default="last",
    help="JK strategy, either last or concat (default: last)",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=1,
    help="input batch size for training (default: 1)",
)
parser.add_argument(
    "--learning_rate",
    type=float,
    default=0.01,
    help="learning rate for training (default: 0.01)",
)
parser.add_argument(
    "--decay_rate",
    type=float,
    default=0.5,
    help="decay rate for training (default: 0.5)",
)
parser.add_argument(
    "--decay_step",
    type=int,
    default=50,
    help="decay step for training (default: 50)",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=100,
    help="number of epochs to train (default: 100)",
)
parser.add_argument(
    "--num_workers", type=int, default=0, help="number of workers (default: 0)"
)
parser.add_argument(
    "--dataset",
    type=str,
    default="BREC",
    help="dataset name (default: BREC)",
)
parser.add_argument(
    "--data_dir",
    type=str,
    default="dataset/",
    help="directory where to store the data (default: dataset/)",
)
parser.add_argument(
    "--policy",
    type=str,
    default="edge_deleted",
    help="Subgraph selection policy in {edge_deleted, node_deleted, ego_nets}"
    " (default: edge_deleted)",
)
parser.add_argument(
    "--num_hops",
    type=int,
    default=2,  # FIXME in configs
    help="Depth of the ego net if policy is ego_nets (default: 2)",
)
parser.add_argument(
    "--seed", type=int, default=2022, help="random seed (default: 2022)"
)
parser.add_argument(
    "--fraction",
    type=float,
    default=1.0,
    help="Fraction of subsampled subgraphs (1.0 means full bag aka no sampling)",
)
parser.add_argument("--patience", type=int, default=20, help="patience (default: 20)")
parser.add_argument(
    "--task_idx",
    type=int,
    default=-1,
    help="Task idx for Counting substracture task",
)
parser.add_argument(
    "--use_transpose",
    type=str2bool,
    default=False,
    help="Whether to use transpose in SUN",
)
parser.add_argument(
    "--use_residual",
    type=str2bool,
    default=False,
    help="Whether to use residual in SUN",
)
parser.add_argument(
    "--use_cosine",
    type=str2bool,
    default=False,
    help="Whether to use cosine in SGD",
)
parser.add_argument(
    "--optimizer", type=str, default="adam", help="Optimizer, default Adam"
)
parser.add_argument(
    "--asam_rho", type=float, default=0.5, help="Rho parameter for asam."
)
parser.add_argument("--test", action="store_true", help="quick test")

parser.add_argument(
    "--filename", type=str, default="", help="filename to output result (default: )"
)
parser.add_argument(
    "--add_bn", type=str2bool, default=True, help="Whether to use batchnorm in SUN"
)
parser.add_argument(
    "--use_readout",
    type=str2bool,
    default=True,
    help="Whether to use subgraph readout in SUN",
)
parser.add_argument(
    "--use_mlp",
    type=str2bool,
    default=True,
    help="Whether to use mlps (instead of linears) in SUN",
)
parser.add_argument(
    "--subgraph_readout",
    type=str,
    default="sum",
    help="Subgraph readout, default sum",
)
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
torch.cuda.manual_seed_all(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
# torch.use_deterministic_algorithms(True)

# part_dict: {graph generation type, range}
part_dict = {
    "Basic": (0, 60),
    "Regular": (60, 110),
    "Extension": (160, 260),
    "CFI": (260, 320),
    # "4-Vertex_Condition": (360, 380),
    # "Distance_Regular": (380, 400),
}


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
def get_dataset(args, dataset_name):
    time_start = time.process_time()

    # Do something

    dataset = BRECDataset(
        name=dataset_name,
        pre_transform=policy2transform(policy=args.policy, num_hops=args.num_hops),
    )

    time_end = time.process_time()
    time_cost = round(time_end - time_start, 2)
    logger.info(f"dataset construction time cost: {time_cost}")

    return dataset


# Stage 3: model construction
# Here is for model construction.
def get_model_brec(args, device):
    time_start = time.process_time()

    # Do something
    model = get_model(args, in_dim=1, out_dim=OUTPUT_DIM, device=device)

    time_end = time.process_time()
    time_cost = round(time_end - time_start, 2)
    logger.info(f"model construction time cost: {time_cost}")
    return model


# Stage 4: evaluation
# Here is for evaluation.
def evaluation(dataset, model, path, device, args):
    def cov(X):
        D = X.shape[-1]
        mean = torch.mean(X, dim=-1).unsqueeze(-1)
        X = X - mean
        # return torch.matmul(X, X.T) / (D - 1)
        return 1 / (D - 1) * X @ X.transpose(-1, -2)

    S_epsilon = torch.diag(
        torch.full(size=(OUTPUT_DIM, 1), fill_value=EPSILON_MATRIX).reshape(-1)
    )

    def T2_calculation(dataset, log_flag=False):
        with torch.no_grad():
            loader_iter = iter(
                DataLoader(
                    dataset,
                    batch_size=1,
                    num_workers=args.num_workers,
                    follow_batch=["subgraph_idx"],
                )
            )
            pred_0_list = []
            pred_1_list = []
            for i in range(NUM_RELABEL):
                g_0 = next(loader_iter).to(device)
                g_1 = next(loader_iter).to(device)
                pred_0 = model(g_0.to(device)).detach()
                pred_1 = model(g_1.to(device)).detach()
                pred_0_list.append(pred_0)
                pred_1_list.append(pred_1)
            X = torch.cat([x for x in pred_0_list], dim=0).T
            Y = torch.cat([x for x in pred_1_list], dim=0).T
            if log_flag:
                logger.info(f"X_mean = {torch.mean(X, dim=1)}")
                logger.info(f"Y_mean = {torch.mean(Y, dim=1)}")
            D = (X - Y).cpu()
            # D = (X - Y)
            D_mean = torch.mean(D, dim=1).reshape(-1, 1)
            # S = torch.nan_to_num(cov(D))
            S = cov(D)
            # inv_S = torch.linalg.pinv(S)
            inv_S = torch.inverse(S + S_epsilon)
            return torch.abs(torch.mm(torch.mm(D_mean.T, inv_S), D_mean))

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
            logger.info(f"ID: {id - part_range[0]}")
            model = get_model_brec(args, device)
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
                traintest_loader = DataLoader(
                    dataset_traintest,
                    batch_size=BATCH_SIZE,
                    num_workers=args.num_workers,
                    follow_batch=["subgraph_idx"],
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
                correct_list.append(id - part_range[0])
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
        "Real_correct\tCorrect\tFail\tModel\tNum_hops\tNum_layer\tEmb_dim\tChannels\tJk\tGnn_type\tPolicy\tOUTPUT_DIM\tBATCH_SIZE\tLEARNING_RATE\tWEIGHT_DECAY\tTHRESHOLD\tMARGIN\tLOSS_THRESHOLD\tEPOCH\tSEED"
    )
    logger.info(
        f"{cnt-fail_in_reliability}\t{cnt}\t{fail_in_reliability}\t{args.model}\t{args.num_hops}\t{args.num_layer}\t{args.emb_dim}\t{args.channels}\t{args.jk}\t{args.gnn_type}\t{args.policy}"
        f"\t{OUTPUT_DIM}\t{BATCH_SIZE}\t{LEARNING_RATE}\t{WEIGHT_DECAY}\t{THRESHOLD}\t{MARGIN}\t{LOSS_THRESHOLD}\t{EPOCH}\t{SEED}"
    )


def main():
    args.channels = list(map(int, args.channels.split("-")))
    if args.channels[0] == 0:
        # Used to get NestedGNN from DS
        args.channels = []
    device = (
        torch.device("cuda:" + str(args.device))
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    OUT_PATH = "result_BREC"
    NAME = f"{args.model}-{args.num_hops}_{args.num_layer}_{args.emb_dim}_{args.gnn_type}_{args.policy}"
    PATH = os.path.join(OUT_PATH, NAME)
    if args.policy in ["ego_nets", "ego_nets_plus", "nested"]:
        DATASET_NAME = os.path.join(args.policy, str(args.num_hops))
    else:
        DATASET_NAME = "no_param"

    os.makedirs(PATH, exist_ok=True)
    LOG_NAME = os.path.join(PATH, "log.txt")
    logger.remove(handler_id=None)
    logger.add(LOG_NAME, rotation="5MB")

    logger.info(f"Args: {dumps(vars(args), indent=4, sort_keys=True)}")
    pre_calculation()
    dataset = get_dataset(args, dataset_name=DATASET_NAME)
    model = get_model_brec(args, device)
    evaluation(dataset=dataset, model=model, device=device, path=OUT_PATH, args=args)


if __name__ == "__main__":
    main()
