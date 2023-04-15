# KCSetGNN Reproduction

## Requirements

Please refer to [(k,c)(≤)-SetGNN](https://github.com/LingxiaoShawn/KCSetGNN)

## Usages

To reproduce best result on KCSetGNN, run:

```bash
python test_BREC_search.py
```

The configs are stored in train/configs/BREC.yaml.

Noting that KCSetGNN costs much memory for large graphs, thus we only test on 3-WL-distinguishable graphs, i.e., the upper bound of KCSetGNN (k=3, c=2). Please replace *brec_v3_3wl.txt* to *brec_v3_3wl.npy*.
