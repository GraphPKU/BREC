import os

SEED = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
for seed in SEED:
    script_base = (
        f"python test_BREC_ds.py --policy=ego_nets_plus --model=deepsets --channels=64-64 --gnn_type=originalgin --drop_ratio=0.0 --jk=last --num_hops=6 --num_layer=10 --emb_dim=32 "
        f"--BATCH_SIZE=32  --SEED={seed} --device=0"
    )
    print(script_base)
    os.system(script_base)
