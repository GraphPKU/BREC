import numpy as np
import torch
import torch_geometric
import torch_geometric.loader
from loguru import logger
import time
from brec.dataset import BRECDataset
from tqdm import tqdm
import os
from torch.nn import CosineEmbeddingLoss
import configparser

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


#    learning_rate, weight_decay, batch_size, epoch, loss_threshold):
def evaluate(dataset, model, device, log_path=None, training_config=None):
    """_summary_

    Args:
        dataset (torch_geometric.data.InMemoryDataset): Brec Dataset
        model (torch.nn.Module): Model to be tested
        device (torch.device): Torch device
        logger (loguru.logger): Logger
        log_path (String): log path
        training_config (configparser.ConfigParser): training config

    Returns:
        None: None
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

    part_dict = {
        "Basic": (0, 60),
        "Regular": (60, 160),
        "Extension": (160, 260),
        "CFI": (260, 360),
        "4-Vertex_Condition": (360, 380),
        "Distance_Regular": (380, 400),
    }
    if log_path is not None:
        logger.remove()
        logger.add(log_path)

    global LEARNING_RATE, WEIGHT_DECAY, LOSS_THRESHOLD, BATCH_SIZE, EPOCH
    if training_config is not None:
        LEARNING_RATE = (
            training_config.getfloat("training", "learning_rate")
            if training_config.has_option("training", "learning_rate")
            else LEARNING_RATE
        )
        WEIGHT_DECAY = (
            training_config.getfloat("training", "weight_decay")
            if training_config.has_option("training", "weight_decay")
            else WEIGHT_DECAY
        )
        LOSS_THRESHOLD = (
            training_config.getfloat("training", "loss_threshold")
            if training_config.has_option("training", "loss_threshold")
            else LOSS_THRESHOLD
        )
        BATCH_SIZE = (
            training_config.getint("training", "batch_size")
            if training_config.has_option("training", "batch_size")
            else BATCH_SIZE
        )
        EPOCH = (
            training_config.getint("training", "epoch")
            if training_config.has_option("training", "epoch")
            else EPOCH
        )

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
            model.reset_parameters()
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
                # logger.info(f"Loss: {loss_all}")
                if loss_all < LOSS_THRESHOLD:
                    # logger.info("Early Stop Here")
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
