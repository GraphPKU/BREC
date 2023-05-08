# This program is the pipeline for testing expressiveness.
# It includes 4 stages:
#   1. pre-calculation;
#   2. dataset construction;
#   3. model construction;
#   4. evaluation


import torch_geometric
import torch_geometric.loader
from loguru import logger
import time
from BRECDataset_v3 import BRECDataset
from tqdm import tqdm
import os
import argparse
from torch.nn import CosineEmbeddingLoss


import numpy as np
import torch
from json import dumps
import torch_geometric.transforms as T
from torch.optim import Adam
from torch_geometric.loader import DataListLoader, DataLoader
from torch_geometric.nn import DataParallel
from torch_geometric.seed import seed_everything
import torch.nn.functional as F
import train_utils
from data_utils import extract_multi_hop_neighbors, post_transform, resistance_distance
from layers.input_encoder import LinearEncoder, EmbeddingEncoder
from layers.layer_utils import make_gnn_layer
from models.GraphClassification import GraphClassification
from models.model_utils import make_GNN
from torch.utils.data import ConcatDataset


NUM_RELABEL = 32
P_NORM = 2
OUTPUT_DIM = 16
EPSILON_MATRIX = 1e-7
EPSILON_CMP = 1e-6
SAMPLE_NUM = 400
EPOCH = 20
MARGIN = 0.0
LEARNING_RATE = 1e-4
THRESHOLD = 72.34
BATCH_SIZE = 16
WEIGHT_DECAY = 1e-5
LOSS_THRESHOLD = 0.3

global_var = globals().copy()
HYPERPARAM_DICT = dict()
for k, v in global_var.items():
    if isinstance(v, int) or isinstance(v, float):
        HYPERPARAM_DICT[k] = v

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
    "--save_dir",
    type=str,
    default="./save",
    help="Base directory for saving information.",
)
parser.add_argument(
    "--seed", type=int, default=2022, help="Random seed for reproducibility."
)
parser.add_argument("--dataset_name", type=str, default="BREC", help="name of dataset")
parser.add_argument(
    "--drop_prob",
    type=float,
    default=0.0,
    help="Probability of zeroing an activation in dropout layers.",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=1,
    help="Batch size per GPU. Scales automatically when \
                        multiple GPUs are available.",
)
parser.add_argument("--num_workers", type=int, default=0, help="number of worker.")
parser.add_argument(
    "--load_path",
    type=str,
    default=None,
    help="Path to load as a model checkpoint.",
)
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
parser.add_argument("--l2_wd", type=float, default=3e-6, help="L2 weight decay.")
parser.add_argument(
    "--kernel",
    type=str,
    default="spd",
    choices=("gd", "spd"),
    help="the kernel used for K-hop computation",
)
parser.add_argument("--num_epochs", type=int, default=200, help="Number of epochs.")
parser.add_argument(
    "--max_grad_norm",
    type=float,
    default=5.0,
    help="Maximum gradient norm for gradient clipping.",
)
parser.add_argument(
    "--hidden_size", type=int, default=32, help="hidden size of the model"
)
parser.add_argument(
    "--model_name",
    type=str,
    default="KPGIN",
    choices=("KPGCN", "KPGIN", "KPGraphSAGE", "KPGINPlus"),
    help="Base GNN model",
)
parser.add_argument("--K", type=int, default=4, help="number of hop to consider")
parser.add_argument(
    "--max_pe_num",
    type=int,
    default=1000,
    help="Maximum number of path encoding. Must be equal to or greater than 1",
)
parser.add_argument(
    "--max_edge_type",
    type=int,
    default=1,
    help="Maximum number of type of edge to consider in peripheral edge information",
)
parser.add_argument(
    "--max_edge_count",
    type=int,
    default=1000,
    help="Maximum count per edge type in peripheral edge information",
)
parser.add_argument(
    "--max_hop_num",
    type=int,
    default=4,
    help="Maximum number of hop to consider in peripheral configuration information",
)
parser.add_argument(
    "--max_distance_count",
    type=int,
    default=1000,
    help="Maximum count per hop in peripheral configuration information",
)
parser.add_argument(
    "--wo_peripheral_edge",
    action="store_true",
    help="remove peripheral edge information from model",
)
parser.add_argument(
    "--wo_peripheral_configuration",
    action="store_true",
    help="remove peripheral node configuration from model",
)
parser.add_argument(
    "--wo_path_encoding",
    action="store_true",
    help="remove path encoding from model",
)
parser.add_argument(
    "--wo_edge_feature", action="store_true", help="remove edge feature from model"
)
parser.add_argument(
    "--num_hop1_edge", type=int, default=1, help="Number of edge type in hop 1"
)
parser.add_argument(
    "--num_layer", type=int, default=4, help="Number of layer for feature encoder"
)
parser.add_argument(
    "--JK",
    type=str,
    default="last",
    choices=("sum", "max", "mean", "attention", "last"),
    help="Jumping knowledge method",
)
parser.add_argument(
    "--residual",
    action="store_true",
    help="Whether to use residual connection between each layer",
)
parser.add_argument(
    "--use_rd",
    action="store_true",
    help="Whether to add resistance distance feature to model",
)
parser.add_argument(
    "--virtual_node",
    action="store_true",
    help="Whether add virtual node information in each layer",
)
parser.add_argument("--eps", type=float, default=0.0, help="Initital epsilon in GIN")
parser.add_argument(
    "--train_eps", action="store_true", help="Whether the epsilon is trainable"
)
parser.add_argument(
    "--combine",
    type=str,
    default="geometric",
    choices=("attention", "geometric"),
    help="Jumping knowledge method",
)
parser.add_argument(
    "--pooling_method",
    type=str,
    default="sum",
    choices=("mean", "sum", "attention"),
    help="pooling method in graph classification",
)
parser.add_argument(
    "--norm_type",
    type=str,
    default="Batch",
    choices=("Batch", "Layer", "Instance", "GraphSize", "Pair"),
    help="normalization method in model",
)
parser.add_argument(
    "--aggr",
    type=str,
    default="add",
    help="aggregation method in GNN layer, only works in GraphSAGE",
)
parser.add_argument(
    "--split", type=int, default=10, help="number of fold in cross validation"
)
parser.add_argument(
    "--factor",
    type=float,
    default=0.5,
    help="factor in the ReduceLROnPlateau learning rate scheduler",
)
parser.add_argument(
    "--patience",
    type=int,
    default=5,
    help="patience in the ReduceLROnPlateau learning rate scheduler",
)

args = parser.parse_args()
if args.wo_path_encoding:
    args.num_hopk_edge = 1
else:
    args.num_hopk_edge = args.max_pe_num
args.parallel = False
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

# part_dict: {graph generation type, range}
part_dict = {
    "Basic": (0, 60),
    "Regular": (60, 160),
    "Extension": (160, 260),
    "CFI": (260, 360),
    "4-Vertex_Condition": (360, 380),
    "Distance_Regular": (380, 400),
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
    def node_feature_transform(data):
        if "x" not in data:
            data.x = torch.ones([data.num_nodes, 1], dtype=torch.long)
        return data

    def multihop_transform(g):
        return extract_multi_hop_neighbors(
            g,
            args.K,
            args.max_pe_num,
            args.max_hop_num,
            args.max_edge_type,
            args.max_edge_count,
            args.max_distance_count,
            args.kernel,
        )

    if args.use_rd:
        rd_feature = resistance_distance
    else:

        def rd_feature(g):
            return g

    transform = post_transform(args.wo_path_encoding, args.wo_edge_feature)

    dataset = BRECDataset(
        name=dataset_name,
        pre_transform=T.Compose(
            [node_feature_transform, multihop_transform, rd_feature]
        ),
        transform=transform,
    )
    dataset.data.x = dataset.data.x.long()
    # dataset.data.x = Variable(dataset.data.x.long(), requires_grad=True)

    time_end = time.process_time()
    time_cost = round(time_end - time_start, 2)
    logger.info(f"dataset construction time cost: {time_cost}")

    return dataset


# Stage 3: model construction
# Here is for model construction.
def get_model(args):
    time_start = time.process_time()

    # Do something
    layer = make_gnn_layer(args)
    init_emb = EmbeddingEncoder(args.input_size, args.hidden_size)
    GNNModel = make_GNN(args)
    gnn = GNNModel(
        num_layer=args.num_layer,
        gnn_layer=layer,
        JK=args.JK,
        norm_type=args.norm_type,
        init_emb=init_emb,
        residual=args.residual,
        virtual_node=args.virtual_node,
        use_rd=args.use_rd,
        num_hop1_edge=args.num_hop1_edge,
        max_edge_count=args.max_edge_count,
        max_hop_num=args.max_hop_num,
        max_distance_count=args.max_distance_count,
        wo_peripheral_edge=args.wo_peripheral_edge,
        wo_peripheral_configuration=args.wo_peripheral_configuration,
        drop_prob=args.drop_prob,
    )

    model = GraphClassification(
        embedding_model=gnn,
        pooling_method=args.pooling_method,
        output_size=args.output_size,
    ).to(args.device)

    model.reset_parameters()
    if args.parallel:
        model = DataParallel(model, args.gpu_ids)

    time_end = time.process_time()
    time_cost = round(time_end - time_start, 2)
    logger.info(f"model construction time cost: {time_cost}")
    return model


# Stage 4: evaluation
# Here is for evaluation.
def evaluation(dataset, model, path, device):
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
            logger.info(f"ID: {id - part_range[0]}")
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
        "Real_correct\tCorrect\tFail\tK\tkernel\tlayers\thidden\tOUTPUT_DIM\tBATCH_SIZE\tLEARNING_RATE\tWEIGHT_DECAY\tTHRESHOLD\tMARGIN\tLOSS_THRESHOLD\tEPOCH\tSEED"
    )
    logger.info(
        f"{cnt-fail_in_reliability}\t{cnt}\t{fail_in_reliability}\t{args.K}\t{args.kernel}\t{args.num_layer}\t{args.hidden_size}"
        f"\t{OUTPUT_DIM}\t{BATCH_SIZE}\t{LEARNING_RATE}\t{WEIGHT_DECAY}\t{THRESHOLD}\t{MARGIN}\t{LOSS_THRESHOLD}\t{EPOCH}\t{SEED}"
    )


def main():

    args.name = (
        args.model_name
        + "_"
        + args.kernel
        + "_"
        + str(args.K)
        + "_"
        + str(args.wo_peripheral_edge)
        + "_"
        + str(args.wo_peripheral_configuration)
        + "_"
        + str(args.wo_path_encoding)
        + "_"
        + str(args.wo_edge_feature)
    )

    OUT_PATH = "result_BREC"
    NAME = args.name
    PATH = os.path.join(OUT_PATH, NAME)
    DATASET_NAME = str(args.K) + str(args.kernel)
    os.makedirs(PATH, exist_ok=True)
    LOG_NAME = os.path.join(PATH, "log.txt")
    logger.remove(handler_id=None)
    logger.add(LOG_NAME)

    args.input_size = 2
    args.output_size = OUTPUT_DIM

    logger.info(f"Args: {dumps(vars(args), indent=4, sort_keys=True)}")
    args.device, args.gpu_ids = train_utils.get_available_devices()
    pre_calculation()
    dataset = get_dataset(args, dataset_name=DATASET_NAME)
    model = get_model(args)
    evaluation(dataset=dataset, model=model, device=args.device, path=OUT_PATH)


if __name__ == "__main__":
    main()
