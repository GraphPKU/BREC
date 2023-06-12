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

from typing import Tuple
import logging
import yaml
from ml_collections import ConfigDict
from sacred import Experiment
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter
from numpy import mean as np_mean
from numpy import std as np_std

from models.get_model import get_model
from training.trainer import Trainer
from data.get_data import get_data, get_pretransform, get_transform
from data.const import TASK_TYPE_DICT, CRITERION_DICT
from data.data_utils import SyncMeanTimer
from data.data_utils import AttributedDataLoader, scale_grad
from data.custom_dataloader import MYDataLoader


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
WEIGHT_DECAY = 1e-4
LOSS_THRESHOLD = 0.2
SEED = 2023

ex = Experiment()

# part_dict: {graph generation type, range}
part_dict = {
    "Basic": (0, 60),
    "Regular": (60, 160),
    "Extension": (160, 260),
    "CFI": (260, 320),
    # "4-Vertex_Condition": (360, 380),
    # "Distance_Regular": (380, 400),
}
# parser = argparse.ArgumentParser(description="BREC Test")

# parser.add_argument("--P_NORM", type=str, default="2")
# parser.add_argument("--EPOCH", type=int, default=EPOCH)
# parser.add_argument("--LEARNING_RATE", type=float, default=LEARNING_RATE)
# parser.add_argument("--BATCH_SIZE", type=int, default=BATCH_SIZE)
# parser.add_argument("--WEIGHT_DECAY", type=float, default=WEIGHT_DECAY)
# parser.add_argument("--OUTPUT_DIM", type=int, default=OUTPUT_DIM)
# parser.add_argument("--SEED", type=int, default=SEED)
# parser.add_argument("--THRESHOLD", type=float, default=THRESHOLD)
# parser.add_argument("--MARGIN", type=float, default=MARGIN)
# parser.add_argument("--LOSS_THRESHOLD", type=float, default=LOSS_THRESHOLD)
# parser.add_argument("--device", type=int, default=0)

# # General settings.
# args = parser.parse_args()

# P_NORM = 2 if args.P_NORM == "2" else torch.inf
# EPOCH = args.EPOCH
# LEARNING_RATE = args.LEARNING_RATE
# BATCH_SIZE = args.BATCH_SIZE
# WEIGHT_DECAY = args.WEIGHT_DECAY
# OUTPUT_DIM = args.OUTPUT_DIM
# SEED = args.SEED
# THRESHOLD = args.THRESHOLD
# MARGIN = args.MARGIN
# LOSS_THRESHOLD = args.LOSS_THRESHOLD
# torch_geometric.seed_everything(SEED)
# torch.backends.cudnn.deterministic = True
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
def get_dataset(name, args, device):
    time_start = time.process_time()

    # Do something
    brec_pre_transform, postfix = get_pretransform(args)
    brec_transform, sample_collator = get_transform(args)
    dataset = BRECDataset(
        name=name, pre_transform=brec_pre_transform, transform=brec_transform
    )

    time_end = time.process_time()
    time_cost = round(time_end - time_start, 2)
    logger.info(f"dataset construction time cost: {time_cost}")

    return dataset


# Stage 3: model construction
# Here is for model construction.
def get_model_brec(args, dataset, device):
    time_start = time.process_time()

    # Do something
    model, emb_model = get_model(args, dataset)
    model.to(device)
    if emb_model is not None:
        emb_model.to(device)

    time_end = time.process_time()
    time_cost = round(time_end - time_start, 2)
    logger.info(f"model construction time cost: {time_cost}")
    return model, emb_model


# Stage 4: evaluation
# Here is for evaluation.
def evaluation(dataset, model, emb_model, path, device, args):
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
                data = data.to(device)
                if emb_model is not None:
                    data, _ = trainer.emb_model_forward(data, emb_model, train=False)
                pred = model(data).detach()
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
        trainer = Trainer(
            dataset=args.dataset.lower(),
            task_type=None,
            voting=args.voting,
            max_patience=args.patience,
            criterion=None,
            device=device,
            imle_configs=args.imle_configs,
            **args.sample_configs,
        )
        start = time.process_time()

        for id in tqdm(range(part_range[0], part_range[1])):
            logger.info(f"ID: {id}")
            model.reset_parameters()
            optimizer = torch.optim.Adam(
                model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
            if emb_model is not None:
                emb_model.reset_parameters()
                optimizer_embd = torch.optim.Adam(
                    emb_model.parameters(),
                    lr=args.imle_configs.embd_lr,
                    weight_decay=args.imle_configs.reg_embd,
                )
                # scheduler_embd = None
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
            if emb_model is not None:
                emb_model.train()
                trainer.imle_scheduler.return_list = False
                trainer.imle_scheduler.perturb = False
                trainer.imle_scheduler.sample_rand = trainer.imle_sample_rand
                optimizer_embd.zero_grad()
            for _ in range(EPOCH):
                traintest_loader = torch_geometric.loader.DataLoader(
                    dataset_traintest, batch_size=BATCH_SIZE
                )
                loss_all = 0
                for batch_id, data in enumerate(traintest_loader):
                    data = data.to(device)
                    optimizer.zero_grad()
                    if emb_model is not None:
                        data, aux_loss = trainer.emb_model_forward(
                            data, emb_model, train=True
                        )
                    pred = model(data)
                    loss = loss_func(
                        pred[0::2],
                        pred[1::2],
                        torch.tensor([-1] * (len(pred) // 2)).to(device),
                    )
                    if aux_loss is not None:
                        loss += aux_loss

                    loss.backward()
                    optimizer.step()

                    if optimizer_embd is not None:
                        if (
                            batch_id % trainer.micro_batch_embd
                            == trainer.micro_batch_embd - 1
                        ) or (batch_id >= len(traintest_loader) - 1):
                            emb_model = scale_grad(
                                emb_model, (batch_id % trainer.micro_batch_embd) + 1
                            )
                            optimizer_embd.step()
                            optimizer_embd.zero_grad()

                    loss_all += len(pred) / 2 * loss.detach().item()
                loss_all /= NUM_RELABEL
                logger.info(f"Loss: {loss_all}")
                if loss_all < LOSS_THRESHOLD:
                    logger.info("Early Stop Here")
                    break
                scheduler.step(loss_all)

            if emb_model is not None:
                del trainer.imle_scheduler.graphs
                del trainer.imle_scheduler.ptr

            model.eval()
            if emb_model is not None:
                emb_model.eval()
                trainer.imle_scheduler.return_list = False
                trainer.imle_scheduler.perturb = (
                    trainer.voting > 1
                )  # only perturb when voting more than once
                trainer.imle_scheduler.sample_rand = False  # test time, always
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
        "Real_correct\tCorrect\tFail\tpolicy\tsample_k\tsample_num\thid_size\tnum_layer\tOUTPUT_DIM\tBATCH_SIZE\tEPOCH\tLEARNING_RATE\tWEIGHT_DECAY\tSEED"
    )
    logger.info(
        f"{cnt-fail_in_reliability}\t{cnt}\t{fail_in_reliability}\t{args.sample_configs.sample_policy}\t{args.sample_configs.sample_k}\t{args.sample_configs.num_subgraphs}\t{args.hid_size}\t{args.num_convlayers}"
        f"\t{OUTPUT_DIM}\t{BATCH_SIZE}\t{EPOCH}\t{LEARNING_RATE}\t{WEIGHT_DECAY}\t{SEED}"
    )


def naming(args) -> str:
    name = f"{args.dataset}_{args.model}_"

    if args.imle_configs is not None:
        name += "IMLE_"
    elif args.sample_configs.sample_with_esan:
        name += "ESAN_"
    elif args.sample_configs.num_subgraphs == 0:
        name += "normal_train_"
    else:
        name += "OnTheFly_"

    name += f"policy_{args.sample_configs.sample_policy}_"
    name += f"samplek_{args.sample_configs.sample_k}_"
    name += f"subg_{args.sample_configs.num_subgraphs}_"
    name += f"rm_node_{args.sample_configs.remove_node}_"
    name += f"fullg_{args.sample_configs.add_full_graph}_"
    try:
        name += f"auxloss_{args.imle_configs.aux_loss_weight}"
    except AttributeError:
        pass

    return name


@ex.automain
def run(fixed):
    fixed = dict(fixed)
    root_dir = (
        fixed["config_root"] if "config_root" in fixed else fixed["dataset"].lower()
    )
    with open(f"./configs/{root_dir}/common_configs.yaml", "r") as stream:
        try:
            common_configs = yaml.safe_load(stream)["common"]
            default_configs = {
                k: v for k, v in common_configs.items() if k not in fixed
            }
            fixed.update(default_configs)
        except yaml.YAMLError as exc:
            print(exc)

    args = ConfigDict(fixed)
    hparams = naming(args)

    global P_NORM, EPOCH, LEARNING_RATE, BATCH_SIZE, WEIGHT_DECAY, OUTPUT_DIM, SEED, THRESHOLD, MARGIN, LOSS_THRESHOLD
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

    device = torch.device(f"cuda:{args.DEVICE}" if torch.cuda.is_available() else "cpu")
    device = torch.device('cpu')

    OUT_PATH = "result_BREC"
    NAME = hparams
    DATASET_NAME = "no_param"
    path = os.path.join(OUT_PATH, NAME)
    os.makedirs(path, exist_ok=True)

    logger.remove(handler_id=None)
    LOG_NAME = os.path.join(path, "log.txt")
    logger.add(LOG_NAME, rotation="5MB")

    logger.info(args)

    pre_calculation()
    dataset = get_dataset(name=DATASET_NAME, args=args, device=device)

    model, emb_model = get_model_brec(args, dataset, device)
    evaluation(dataset, model, emb_model, OUT_PATH, device, args)
