# GNN-AK Reproduction

## Requirements

Please refer to [I$^2$-GNN](https://github.com/graphpku/i2gnn)

## Usages

To reproduce best result on I$^2$-GNN, run:

```bash
python test_BREC_search.py
```

Noting that I$^2$-GNN costs much memory for large graphs, thus we do not test on 4-vertex condition graphs and the largest 40 CFI graphs. Please replace *brec_v3_no4v_60cfi.txt* to *brec_v3_no4v_60cfi.npy*.
