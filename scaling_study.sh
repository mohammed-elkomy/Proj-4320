#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# All scaling experiment sbatch commands.
# Naming: NnNg = N nodes x N GPUs per node.
#
# Strong scaling: fixed problem (strong_scaling.ppm, pop=100).
#   Total budget = 100k gens, auto-divided by world_size per island.
#
# Weak scaling: image grows with total GPU count, 100k gens per island.
#   Multi-node configs pre-scale MAX_GENERATIONS so division yields 100k/island.
# ─────────────────────────────────────────────────────────────────────────────

# ── Strong scaling ────────────────────────────────────────────────────────────
sbatch --nodes=1 --ntasks-per-node=1 --gres=gpu:1 run.sbatch config/strong_scaling_1n1g.conf # same image
sbatch --nodes=1 --ntasks-per-node=1 --gres=gpu:2 run.sbatch config/strong_scaling_1n2g.conf # same image
sbatch --nodes=1 --ntasks-per-node=1 --gres=gpu:4 run.sbatch config/strong_scaling_1n4g.conf # same image
sbatch --nodes=1 --ntasks-per-node=1 --gres=gpu:6 run.sbatch config/strong_scaling_1n6g.conf # same image
sbatch --nodes=2 --ntasks-per-node=1 --gres=gpu:6 run.sbatch config/strong_scaling_2n6g.conf # same image
sbatch --nodes=4 --ntasks-per-node=1 --gres=gpu:6 run.sbatch config/strong_scaling_4n6g.conf # same image
sbatch --nodes=8 --ntasks-per-node=1 --gres=gpu:6 run.sbatch config/strong_scaling_8n6g.conf # same image

# ── Weak scaling ──────────────────────────────────────────────────────────────
sbatch --nodes=1 --ntasks-per-node=1 --gres=gpu:1 run.sbatch config/weak_scaling_1n1g.conf   # x1  image
sbatch --nodes=1 --ntasks-per-node=1 --gres=gpu:2 run.sbatch config/weak_scaling_1n2g.conf   # x2  image
sbatch --nodes=1 --ntasks-per-node=1 --gres=gpu:4 run.sbatch config/weak_scaling_1n4g.conf   # x4  image
sbatch --nodes=1 --ntasks-per-node=1 --gres=gpu:6 run.sbatch config/weak_scaling_1n6g.conf   # x6  image
sbatch --nodes=2 --ntasks-per-node=1 --gres=gpu:6 run.sbatch config/weak_scaling_2n6g.conf   # x12 image, 100k gens/island
sbatch --nodes=4 --ntasks-per-node=1 --gres=gpu:6 run.sbatch config/weak_scaling_4n6g.conf   # x24 image, 100k gens/island
sbatch --nodes=8 --ntasks-per-node=1 --gres=gpu:6 run.sbatch config/weak_scaling_8n6g.conf   # x48 image, 100k gens/island
