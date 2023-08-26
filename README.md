# Towards Better Evaluation of GNN Expressiveness with BREC Dataset

## About

This repository is the official implementation of the following paper: [Towards Better Evaluation of GNN Expressiveness with BREC Dataset](https://arxiv.org/abs/2304.07702).

We also provide a Pypi package for simple usage. Please refer to [Pypi package](https://pypi.org/project/brec/).

**BREC**  is a new dataset for GNN expressiveness comparison.
It addresses the limitations of previous datasets, including difficulty, granularity, and scale, by incorporating
400 pairs of various graphs in four categories (Basic, Regular, Extension, CFI).
The graphs are organized pair-wise, where each pair is tested individually to return whether a GNN can distinguish them. We propose a new evaluation method, **RPC** (Reliable Paired Comparisons), with a contrastive training framework.


## Usages

### File Structure

We first introduce the general file structure of BREC:

```bash
├── Data
    └── raw
        └── brec_v3.npy    # unprocessed BREC dataset in graph6 format
├── BRECDataset_v3.py    # BREC dataset construction file
├── test_BREC.py    # Evaluation framework file
└── test_BREC_search.py    # Run test_BREC.py with 10 seeds for the final result
```

To test on BREC, there are four steps to follow:

1. Select a model and go to the corresponding [directory](#directory).
2. [Prepare](#preparation) dataset based on selected model requirements.
3. Check *test_BREC.py* for implementation if you want to test your own GNN.
4. Run *test_BREC_search.py* for final result. Only if no failure in reliability check for all seeds is available.

### Requirements

Tested combination: Python 3.8.13 + [PyTorch 1.13.1](https://pytorch.org/get-started/previous-versions/) + [PyTorch_Geometric 2.2](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

Other required Python libraries included: numpy, networkx, loguru, etc.

For reproducing other results, please refer to the corresponding requirements for additional libraries.

### <span id="preparation">Data Preparation</span>

Data preparation requires two steps: generate the dataset and arrange it in the correct position.

#### Step 1: Data Generation

We provide zipped data file *BREC_data_all.zip*. You can unzip it for 3 data files in npy format. You can also customize dataset refering to [Customize Dataset](#customize).

For most methods, only *brec_v3.npy* is needed. More detailed requirements on datasets can refer to corresponding implementations.

#### Step 2: Data Arrangement

Replace *$.txt* with *$.npy* in the corresponding *Data/raw* directory.
For most methods, only *brec_v3.txt* is in *Data/raw* directory. Thus replacing *brec_v3.txt* with *brec_v3.npy* is enough.

### <span id="directory">Reproduce Baselines</span>

For baseline results reproduction, please refer to the respective directories:

| Baseline          | Directory                           |
| ----------------- | ----------------------------------- |
| NGNN              | NestedGNN                           |
| DS-GNN            | SUN                                 |
| DSS-GNN           | SUN                                 |
| SUN               | SUN                                 |
| PPGN              | ProvablyPowerfulGraphNetworks_torch |
| GNN-AK            | GNNAsKernel                         |
| DE+NGNN           | NestedGNN                           |
| KP-GNN            | KP-GNN                              |
| KC-SetGNN         | KCSetGNN                            |
| I$^2$-GNN         | I2GNN                               |
| GSN               | GSN                                 |
| Graphormer        | Graphormer                          |
| OSAN              | OSAN                                |
| $\delta$-LGNN(SparseWL) | SparseWL                      |
| SWL               | SWL                                 |
| DropGNN           | DropGNN                             |
| Non-GNN Baselines | Non-GNN                             |
| Your Own GNN      | Base                                |

### Test Your Own GNN

In addition to previous steps in reproducing baselines, implementing *test_BREC.py* is needed.

#### Evaluation Step

To test your GNNs, in addition to previous steps, you need to implement *test_BREC.py* with your model and run (${configs} represents corresponding config usage):

```bash
python test_BREC.py ${configs}
```

*test.py* is the pipeline for evaluation, including four stages:

```bash
1. pre-calculation;

2. dataset construction;

3. model construction;

4. evaluation
```

**Pre-calculation** aims to organize offline operations on graphs.

**Dataset construction** aims to process the dataset with specific operations. BRECDataset is implemented based on InMemoryDataset. It is recommended to use transform and pre_transform to transform the graphs.

**Model construction** aims to construct the GNN.

**Evaluation** implements RPC. With the model and dataset, it will produce the final results.

Suppose your own experiment is done by running *python main.py*. You can easily implement *test_BREC.py* with *main.py*. You can drop the training and testing pipeline in *main.py* and split the rest into corresponding stages in *test.py*.

### <span id="customize">Customize BREC Dataset</span>

Some graphs in BREC may be too difficult for some models, like strongly regular graphs that 3-WL can not distinguish.
You can discard some graphs from BREC to reduce test time.
In addition, the parameter $q$ in RPC can also be adjusted when customizing. Only the *customize* directory is required.

```bash
├── Data     # Original graph file
    └── raw
        ├── basic.npy  # Basic graphs
        ├── regular.npy  # Simple regular graphs
        ├── str.npy   # Strongly regular graphs
        ├── cfi.npy   # CFI graphs
        ├── extension.npy # Extension graphs
        ├── 4vtx.npy  # 4-vertex condition graphs
        └── dr.npy   # Distance regular graphs
├── dataset_v3.py    # Generating brec_v3.npy 
├── dataset_v3_3wl.py # Generating brec_v3_3wl.npy    
└── dataset_v3_no4v_60cfi.py  # Generating brec_v3_no4v_60cfi.npy
```

Using *brec_v3.npy* by running *python dataset_v3.py* is enough for most methods.

For customization, suppose you want to discard distance regular graphs from BREC. You need to delete *dr.npy* related codes in *dataset_v3.py*. The total pair number and the "category-id_range" dictionary should also be adjusted.

"NUM" represent $q$ in RPC, which can be adjusted for a different RPC check.

### Results Demonstration

The 400 pairs of graphs are from four categories: Basic, Regular, Extension, CFI. We further split 4-vertex condition and distance regular graphs from Regular as a separate category. The "category-id_range" dictionary is as follows:

```python
  "Basic": (0, 60),
  "Regular": (60, 160),
  "Extension": (160, 260),
  "CFI": (260, 360),
  "4-Vertex_Condition": (360, 380),
  "Distance_Regular": (380, 400),
```

You can refer to the detailed graph in *customize/Data/raw* for analysis.
