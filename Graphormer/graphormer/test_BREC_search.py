import os

SEED = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
for seed in SEED:
    script_base = (
        f"python test_BREC.py --ffn_dim 80 --hidden_dim 80 --num_heads 8 --dropout_rate 0.0 --intput_dropout_rate 0.0 --n_layers 12 "
        f"--EPOCH=100 --LEARNING_RATE 2e-5 --edge_type none --WEIGHT_DECAY 0.0 --BATCH_SIZE 16 --SEED={seed}"
    )
    print(script_base)
    os.system(script_base)
