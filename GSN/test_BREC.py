## ---- imports -----

import argparse
import utils_parsing as parse
import os

import random
import copy
import json

import torch
import torch.nn.functional as F
import numpy as np

from torch.nn import CosineEmbeddingLoss
from torch_geometric.data import DataLoader
from torch_geometric.data import Data


from utils import process_arguments, prepare_dataset
from utils_data_prep import separate_data, separate_data_given_split
from utils_encoding import encode

from train_test_funcs import (
    train,
    test_isomorphism,
    test,
    test_ogb,
    setup_optimization,
    resume_training,
)

from models_graph_classification_mlp import MLPSubstructures
from models_graph_classification import GNNSubstructures
from models_graph_classification_ogb_original import GNN_OGB

from ogb.graphproppred import Evaluator

from loguru import logger
import time
from tqdm import tqdm

NUM_RELABEL = 32
P_NORM = 2
OUTPUT_DIM = 16
EPSILON_MATRIX = 1e-7
EPSILON_CMP = 1e-6
SAMPLE_NUM = 260
EPOCH = 20
MARGIN = 0.0
LEARNING_RATE = 1e-4
THRESHOLD = 72.34
BATCH_SIZE = 16
WEIGHT_DECAY = 1e-5
LOSS_THRESHOLD = 0.1
part_dict = {
    "Basic": (0, 60),
    "Regular": (60, 160),
    "Extension": (160, 260),
}


def get_model(
    num_features,
    encoder_ids,
    d_id,
    num_edge_features,
    d_in_node_encoder,
    d_in_edge_encoder,
    encoder_degrees,
    d_degree,
    args,
):
    return GNNSubstructures(
        in_features=num_features,
        out_features=OUTPUT_DIM,
        encoder_ids=encoder_ids,
        d_in_id=d_id,
        in_edge_features=num_edge_features,
        d_in_node_encoder=d_in_node_encoder,
        d_in_edge_encoder=d_in_edge_encoder,
        encoder_degrees=encoder_degrees,
        d_degree=d_degree,
        **args,
    )


# Stage 4: evaluation
# Here is for evaluation.
def evaluation(
    dataset,
    path,
    device,
    num_features,
    encoder_ids,
    d_id,
    num_edge_features,
    d_in_node_encoder,
    d_in_edge_encoder,
    encoder_degrees,
    d_degree,
    args,
):
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
            model = GNNSubstructures(
                in_features=num_features,
                out_features=OUTPUT_DIM,
                encoder_ids=encoder_ids,
                d_in_id=d_id,
                in_edge_features=num_edge_features,
                d_in_node_encoder=d_in_node_encoder,
                d_in_edge_encoder=d_in_edge_encoder,
                encoder_degrees=encoder_degrees,
                d_degree=d_degree,
                **args,
            ).to(device)
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
        "Real_correct\tCorrect\tFail\tOUTPUT_DIM\tBATCH_SIZE\tLEARNING_RATE\tWEIGHT_DECAY\tSEED"
    )
    logger.info(
        f"{cnt-fail_in_reliability}\t{cnt}\t{fail_in_reliability}\t{OUTPUT_DIM}\t{BATCH_SIZE}\t{LEARNING_RATE}\t{WEIGHT_DECAY}\t{SEED}"
    )


## ---- main function -----


def main(args):

    ## ----------------------------------- argument processing

    (
        args,
        extract_ids_fn,
        count_fn,
        automorphism_fn,
        loss_fn,
        prediction_fn,
        perf_opt,
    ) = process_arguments(args)
    evaluator = Evaluator(args["dataset_name"]) if args["dataset"] == "ogb" else None

    ## ----------------------------------- infrastructure

    torch.manual_seed(args["seed"])
    torch.cuda.manual_seed(args["seed"])
    torch.cuda.manual_seed_all(args["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args["np_seed"])
    os.environ["PYTHONHASHSEED"] = str(args["seed"])
    random.seed(args["seed"])
    print("[info] Setting all random seeds {}".format(args["seed"]))

    torch.set_num_threads(args["num_threads"])
    if args["GPU"]:
        device = torch.device(
            "cuda:" + str(args["device_idx"]) if torch.cuda.is_available() else "cpu"
        )
        print(
            "[info] Training will be performed on {}".format(
                torch.cuda.get_device_name(args["device_idx"])
            )
        )
    else:
        device = torch.device("cpu")
        print("[info] Training will be performed on cpu")

    if args["wandb"]:
        import wandb

        wandb.init(
            sync_tensorboard=False,
            project=args["wandb_project"],
            reinit=False,
            config=args,
            entity=args["wandb_entity"],
        )
        print("[info] Monitoring with wandb")

    ## ----------------------------------- datasets: prepare and preprocess (count or load subgraph counts)

    path = os.path.join(args["root_folder"], args["dataset"], args["dataset_name"])
    subgraph_params = {
        "induced": args["induced"],
        "edge_list": args["custom_edge_list"],
        "directed": args["directed"],
        "directed_orbits": args["directed_orbits"],
    }
    graphs_ptg, num_classes, orbit_partition_sizes = prepare_dataset(
        path,
        args["dataset"],
        args["dataset_name"],
        args["id_scope"],
        args["id_type"],
        args["k"],
        args["regression"],
        extract_ids_fn,
        count_fn,
        automorphism_fn,
        args["multiprocessing"],
        args["num_processes"],
        **subgraph_params,
    )

    # OGB-specifics: different feature collections
    if args["dataset"] == "ogb":

        if (
            args["features_scope"] == "simple"
        ):  # only retain the top two node/edge features
            print("[info] (OGB) Using simple node and edge features")
            simple_graphs = []
            for graph in graphs_ptg:
                new_data = Data()
                for attr in graph.__iter__():
                    name, value = attr
                    setattr(new_data, name, value)
                setattr(new_data, "x", graph.x[:, :2])
                setattr(new_data, "edge_features", graph.edge_features[:, :2])
                simple_graphs.append(new_data)
            graphs_ptg = simple_graphs
        else:
            print("[info] (OGB) Using full node and edge features")

    ## ----------------------------------- node and edge feature dimensions

    if graphs_ptg[0].x.dim() == 1:
        num_features = 1
    else:
        num_features = graphs_ptg[0].num_features

    if hasattr(graphs_ptg[0], "edge_features"):
        if graphs_ptg[0].edge_features.dim() == 1:
            num_edge_features = 1
        else:
            num_edge_features = graphs_ptg[0].edge_features.shape[1]
    else:
        num_edge_features = None

    if args["dataset"] == "chemical" and args["dataset_name"] == "ZINC":
        d_in_node_encoder, d_in_edge_encoder = torch.load(
            os.path.join(path, "processed", "num_feature_types.pt")
        )
        d_in_node_encoder, d_in_edge_encoder = [d_in_node_encoder], [d_in_edge_encoder]
    else:
        d_in_node_encoder = [num_features]
        d_in_edge_encoder = [num_edge_features]

    ## ----------------------------------- encode ids and degrees (and possibly edge features)

    degree_encoding = args["degree_encoding"] if args["degree_as_tag"][0] else None
    id_encoding = args["id_encoding"] if args["id_encoding"] != "None" else None
    encoding_parameters = {
        "ids": {
            "bins": args["id_bins"],
            "strategy": args["id_strategy"],
            "range": args["id_range"],
        },
        "degree": {
            "bins": args["degree_bins"],
            "strategy": args["degree_strategy"],
            "range": args["degree_range"],
        },
    }

    print("Encoding substructure counts and degree features... ", end="")
    graphs_ptg, encoder_ids, d_id, encoder_degrees, d_degree = encode(
        graphs_ptg, id_encoding, degree_encoding, **encoding_parameters
    )
    print("Done.")

    assert args["mode"] in [
        # "isomorphism_test",
        # "train",
        # "test",
        "BREC_test"
    ], "Unknown mode. Only support BREC_test"

    ## ----------------------------------- graph isomorphism testing
    ##
    ## We use GSN with random weights, so no training is performed
    ##

    if args["mode"] == "BREC_test":
        OUT_PATH = "result_BREC"
        NAME = f'{args["model_name"]}_{args["id_type"]}_{args["induced"]}_{args["k"]}_{args["id_scope"]}_{args["msg_kind"]}_{args["num_layers"]}_{args["d_out"]}'
        path = os.path.join(OUT_PATH, NAME)
        os.makedirs(path, exist_ok=True)

        logger.remove(handler_id=None)
        LOG_NAME = os.path.join(path, "log.txt")
        logger.add(LOG_NAME, rotation="5MB")

        logger.info(args)

        evaluation(
            graphs_ptg,
            path,
            device,
            num_features,
            encoder_ids,
            d_id,
            num_edge_features,
            d_in_node_encoder,
            d_in_edge_encoder,
            encoder_degrees,
            d_degree,
            args,
        )
    else:
        print("Not supported for test_BREC!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # set seeds to ensure reproducibility
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--np_seed", type=int, default=0)

    # this specifies the folds for cross-validation
    parser.add_argument(
        "--fold_idx", type=parse.str2list2int, default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    )
    parser.add_argument("--onesplit", type=parse.str2bool, default=False)

    # set multiprocessing to true in order to do the precomputation in parallel
    parser.add_argument("--multiprocessing", type=parse.str2bool, default=False)
    parser.add_argument("--num_processes", type=int, default=64)

    ###### data loader parameters
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_threads", type=int, default=1)

    ###### these are to select the dataset:
    # - dataset can be bionformatics or social and states the class;
    # - name is for the specific problem itself
    parser.add_argument("--dataset", type=str, default="bioinformatics")
    parser.add_argument("--dataset_name", type=str, default="MUTAG")
    parser.add_argument("--split", type=str, default="given")
    parser.add_argument("--root_folder", type=str, default="./datasets")

    ######  set degree_as_tag to True to use the degree as node features;
    # set retain_features to True to keep the existing features as well;
    parser.add_argument("--degree_as_tag", type=parse.str2bool, default=False)
    parser.add_argument("--retain_features", type=parse.str2bool, default=False)

    ###### used only for ogb to reproduce the different configurations,
    # i.e. additional features (full) or not (simple), virtual node or not (vn: True)
    parser.add_argument("--features_scope", type=str, default="full")
    parser.add_argument("--vn", type=parse.str2bool, default=False)
    # denotes the aggregation used by the virtual node
    parser.add_argument("--vn_pooling", type=str, default="sum")
    parser.add_argument("--input_vn_encoder", type=str, default="one_hot_encoder")
    parser.add_argument("--d_out_vn_encoder", type=int, default=None)
    parser.add_argument("--d_out_vn", type=int, default=None)

    ###### substructure-related parameters:
    # - id_type: substructure family
    # - induced: graphlets vs motifs
    # - edge_automorphism: induced edge automorphism or line graph edge automorphism (slightly larger group than the induced edge automorphism)
    # - k: size of substructures that are used; e.g. k=3 means three nodes
    # - id_scope: local vs global --> GSN-e vs GSN-v
    parser.add_argument("--id_type", type=str, default="cycle_graph")
    parser.add_argument("--induced", type=parse.str2bool, default=False)
    parser.add_argument("--edge_automorphism", type=str, default="induced")
    parser.add_argument("--k", type=parse.str2list2int, default=[3])
    parser.add_argument("--id_scope", type=str, default="local")
    parser.add_argument(
        "--custom_edge_list", type=parse.str2ListOfListsOfLists2int, default=None
    )
    parser.add_argument("--directed", type=parse.str2bool, default=False)
    parser.add_argument("--directed_orbits", type=parse.str2bool, default=False)

    ###### encoding args: different ways to encode discrete data

    parser.add_argument("--id_encoding", type=str, default="one_hot_unique")
    parser.add_argument("--degree_encoding", type=str, default="one_hot_unique")

    # binning and minmax encoding parameters. NB: not used in our experimental evaluation
    parser.add_argument("--id_bins", type=parse.str2list2int, default=None)
    parser.add_argument("--degree_bins", type=parse.str2list2int, default=None)
    parser.add_argument("--id_strategy", type=str, default="uniform")
    parser.add_argument("--degree_strategy", type=str, default="uniform")
    parser.add_argument("--id_range", type=parse.str2list2int, default=None)
    parser.add_argument("--degree_range", type=parse.str2list2int, default=None)

    parser.add_argument("--id_embedding", type=str, default="one_hot_encoder")
    parser.add_argument("--d_out_id_embedding", type=int, default=None)
    parser.add_argument("--degree_embedding", type=str, default="one_hot_encoder")
    parser.add_argument("--d_out_degree_embedding", type=int, default=None)

    parser.add_argument("--input_node_encoder", type=str, default="None")
    parser.add_argument("--d_out_node_encoder", type=int, default=None)
    parser.add_argument("--edge_encoder", type=str, default="None")
    parser.add_argument("--d_out_edge_encoder", type=int, default=None)

    # sum or concatenate embeddings when multiple discrete features available
    parser.add_argument("--multi_embedding_aggr", type=str, default="sum")

    # only used for the GIN variant: creates a dummy variable for self loops (e.g. edge features or edge counts)
    parser.add_argument("--extend_dims", type=parse.str2bool, default=True)

    ###### model to be used and architecture parameters, in particular
    # - d_h: is the dimension for internal mlps, set to None to
    #   make it equal to d_out
    # - final_projection: is for jumping knowledge, specifying
    #   which layer is accounted for in the last model stage, if
    #   the list has only one element, that that value gets applied
    #   to all the layers
    # - jk_mlp: set it to True to use an MLP after each jk layer, otherwise a linear layer will be used

    parser.add_argument("--model_name", type=str, default="GSN_sparse")

    parser.add_argument("--random_features", type=parse.str2bool, default=False)
    parser.add_argument("--num_mlp_layers", type=int, default=2)
    parser.add_argument("--d_h", type=int, default=None)
    parser.add_argument("--activation_mlp", type=str, default="relu")
    parser.add_argument("--bn_mlp", type=parse.str2bool, default=True)

    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--d_msg", type=int, default=None)
    parser.add_argument("--d_out", type=int, default=16)
    parser.add_argument("--bn", type=parse.str2bool, default=True)
    parser.add_argument("--dropout_features", type=float, default=0)
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--train_eps", type=parse.str2bool, default=False)
    parser.add_argument("--aggr", type=str, default="add")
    parser.add_argument("--flow", type=str, default="source_to_target")

    parser.add_argument("--final_projection", type=parse.str2list2bool, default=[True])
    parser.add_argument("--jk_mlp", type=parse.str2bool, default=False)
    parser.add_argument("--residual", type=parse.str2bool, default=False)

    parser.add_argument("--readout", type=str, default="sum")

    ###### architecture variations:
    # - msg_kind: gin (extends gin with structural identifiers),
    #             general (general formulation with MLPs - eq 3,4 of the main paper)
    #             ogb (extends the architecture used in ogb with structural identifiers)
    # - inject*: passes the relevant variable to deeper layers akin to skip connections.
    #            If set to False, then the variable is used only as input to the first layer
    parser.add_argument("--msg_kind", type=str, default="general")
    parser.add_argument("--inject_ids", type=parse.str2bool, default=False)
    parser.add_argument("--inject_degrees", type=parse.str2bool, default=False)
    parser.add_argument("--inject_edge_features", type=parse.str2bool, default=True)

    ###### optimisation parameters
    parser.add_argument("--shuffle", type=parse.str2bool, default=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=300)
    parser.add_argument("--num_iters", type=int, default=None)
    parser.add_argument("--num_iters_test", type=int, default=None)
    parser.add_argument("--eval_frequency", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--regularization", type=float, default=0)
    parser.add_argument("--scheduler", type=str, default="StepLR")
    parser.add_argument("--scheduler_mode", type=str, default="min")
    parser.add_argument("--min_lr", type=float, default=0.0)
    parser.add_argument("--decay_steps", type=int, default=50)
    parser.add_argument("--decay_rate", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=20)

    ###### training parameters: task, loss, metric
    parser.add_argument("--regression", type=parse.str2bool, default=False)
    parser.add_argument("--loss_fn", type=str, default="CrossEntropyLoss")
    parser.add_argument("--prediction_fn", type=str, default="multi_class_accuracy")

    ######  folders to save results
    parser.add_argument("--results_folder", type=str, default="temp")
    parser.add_argument("--checkpoint_file", type=str, default="checkpoint")

    ######  general (mode, gpu, logging)
    parser.add_argument("--mode", type=str, default="train")

    parser.add_argument("--resume", type=parse.str2bool, default=False)
    parser.add_argument("--GPU", type=parse.str2bool, default=True)
    parser.add_argument("--device_idx", type=int, default=0)
    parser.add_argument("--wandb", type=parse.str2bool, default=True)
    parser.add_argument("--wandb_realtime", type=parse.str2bool, default=False)
    parser.add_argument("--wandb_project", type=str, default="gsn_project")
    parser.add_argument("--wandb_entity", type=str, default="anonymous")

    ######  misc
    parser.add_argument("--isomorphism_eps", type=float, default=1e-2)
    parser.add_argument("--return_scores", action="store_true")

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
    main(vars(args))
