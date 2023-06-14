import os
SEED = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
for seed in SEED:
    script_base = (
        f"python test_BREC.py --SEED={seed} --hidden_units 16 --num_layers 10 --augmentation dropout "
        f"--OUTPUT_DIM=16 --BATCH_SIZE=16 --LEARNING_RATE=0.0001 --WEIGHT_DECAY=1e-05  --device=0 --num_runs=100 --EPOCH=100"
    )
    print(script_base)
    os.system(script_base)
