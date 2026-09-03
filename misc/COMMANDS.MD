python run_experiments.py capacity --checkpoints 100 100000 --preset fast
python run_experiments.py ablation --seeds 0 1 2 3 4 5 6 7 8 9 --preset tuned
./run_kaggle_experiments.sh --override dg_bridge_lr=0.01