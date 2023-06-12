import os

SEED = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
for seed in SEED:
    script_base = (
        f"python test_BREC_sun.py --policy=ego_nets_plus --model=sun --channels=64-64 --gnn_type=originalgin --drop_ratio=0.0 --jk=last --num_hops=6 --num_layer=9 --emb_dim=32 "
        f"--BATCH_SIZE=32 --LEARNING_RATE=1e-4 --WEIGHT_DECAY=1e-4 --SEED={seed} --device=1 --LOSS_THRESHOLD=0.01 --EPOCH 20"
    )
    print(script_base)
    os.system(script_base)
