# GSN Reproduction

## Requirements

Please refer to [GSN](https://github.com/gbouritsas/GSN)

## Usages

To reproduce best result on GSN, run:

```bash
python test_BREC_search.py
```

To reproduce other results on GSN-v or GSN-e with specific subgraph size, please change the corresponding code in test_BREC_search.py

Noting that GSN already implement an isomorphism test framework, thus we do not use BRECDataset_v3.py, the file in g6 format is used.
