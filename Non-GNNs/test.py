import numpy as np
import argparse
from tqdm import tqdm
from utils import (
    FWL_hash,
    WL_1_hash,
    WL_hash,
    Distance_WL_hash,
    Resistance_distance_WL_hash,
)
import random
import logging
import time
import os


np.random.seed(2022)
random.seed(2022)

func_dict = {
    "fwl": FWL_hash,
    "wl": WL_1_hash,
    "k-wl": WL_hash,
    "distance": Distance_WL_hash,
    "resistance": Resistance_distance_WL_hash,
}


def func_None():
    raise NotImplementedError(f"Cannot find func {args.method}")


def wl_method(method, G, k, mode=None):
    return func_dict.get(method, func_None)(G, k, mode)


part_dict = {
    "Basic": (0, 60),
    "Regular": (60, 160),
    "Extension": (160, 260),
    "CFI": (260, 360),
    "4-Vertex_Condition": (360, 380),
    "Distance_Regular": (380, 400),
    "Reliability": (400, 800),
}


parser = argparse.ArgumentParser(description="Test non-GNN methods on BREC.")
parser.add_argument("--file", type=str, default="brec_nonGNN.npy")
parser.add_argument("--wl", type=int, default="1")
parser.add_argument("--mode", type=str, default="none")
parser.add_argument("--method", type=str, default="wl")
parser.add_argument("--graph_type", type=str, default="none")
args = parser.parse_args()

G_TYPE = args.graph_type.strip()
if G_TYPE == "none":
    method_name = f"{args.wl}{args.method}_{args.mode}"
else:
    if G_TYPE in part_dict:
        method_name = f"{args.wl}{args.method}_{args.mode}_{G_TYPE}"
    else:
        raise NotImplementedError(f"{G_TYPE} do not exist!")


path = os.path.join("result", method_name)
os.makedirs(path, exist_ok=True)
os.makedirs(os.path.join(path, "part_result"), exist_ok=True)

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
logging.basicConfig(
    filename=os.path.join(path, "logging.log"),
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=DATE_FORMAT,
)
logging.info(args)


def count_distinguish_num(graph_tuple_list):
    logging.info(f"{method_name} test starting ---")
    print(f"{method_name} test starting ---")

    cnt = 0
    correct_list = []
    time_cost = 0
    DATA_NUM = (
        400 if G_TYPE == "none" else int(part_dict[G_TYPE][1] - part_dict[G_TYPE][0])
    )

    for part_name, part_range in part_dict.items():
        if not (G_TYPE == "none" or G_TYPE == part_name):
            continue

        logging.info(f"{part_name} part starting ---")

        cnt_part = 0
        correct_list_part = []
        start = time.process_time()

        for id in tqdm(range(part_range[0], part_range[1])):
            graph_tuple = graph_tuple_list[id]
            if not wl_method(
                args.method, graph_tuple[0], args.wl, args.mode
            ) == wl_method(args.method, graph_tuple[1], args.wl, args.mode):
                cnt += 1
                cnt_part += 1
                correct_list.append(id)
                correct_list_part.append(id)
                # if part_name == 'Reliability':
                #     print(id)
            else:
                logging.info(f"Wrong in {id}")

        end = time.process_time()
        time_cost_part = round(end - start, 2)
        time_cost += time_cost_part

        logging.info(
            f"{part_name} part costs time {time_cost_part}; Correct in {cnt_part} / {part_range[1] - part_range[0]}"
        )
        print(
            f"{part_name} part costs time {time_cost_part}; Correct in {cnt_part} / {part_range[1] - part_range[0]}"
        )
        np.save(os.path.join(path, "part_result", part_name), correct_list_part)

    time_cost = round(time_cost, 2)
    Acc = round(cnt / DATA_NUM, 2)

    logging.info(f"Costs time {time_cost}; Correct in {cnt} / {DATA_NUM}, Acc = {Acc}")
    print(f"Costs time {time_cost}; Correct in {cnt} / {DATA_NUM}, Acc = {Acc}")

    np.save(os.path.join(path, "result"), correct_list)

    return


def main():
    graph_tuple_list = np.load(args.file, allow_pickle=True)
    count_distinguish_num(graph_tuple_list)


if __name__ == "__main__":
    main()
