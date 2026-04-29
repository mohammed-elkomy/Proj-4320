# Triangle GA — MPI + CUDA Image Approximation

Approximates a target image using a population of semi-transparent triangles evolved
by a Genetic Algorithm.  Parallelism comes from two orthogonal sources:

- **CUDA** — batch fitness evaluation: all chromosomes in the population are rendered
  and scored in a single GPU kernel launch (one thread-block per chromosome).
  Multiple GPUs on the same node are used automatically.
- **MPI island model** — each MPI rank runs an independent GA island on its node's
  GPUs.  Every `MIGRATION_INTERVAL` generations the best chromosomes circulate in
  a ring (rank 0 → 1 → … → N−1 → 0), letting islands share discoveries without
  merging their populations.

The generation budget (`MAX_GENERATIONS`) is divided evenly across ranks so total
work stays constant for strong-scaling experiments.

## Sample output

The file `sample.ppm` is an 800×400 side-by-side image (target | GA approximation)
produced after 100 k generations with 300 triangles on a single V100 GPU.
PPM is a raw binary format; open it with any image viewer that supports it
(e.g. `display sample.ppm`, GIMP, or `convert sample.ppm sample.png` via ImageMagick).

> **Note:** GitHub does not render `.ppm` files inline. Convert to PNG to embed
> the image in a web viewer.

## Repository layout

```
app.h            — shared header (config, image, GA, CUDA interfaces)
main.c           — MPI driver: island loop, migration, I/O
ga.c             — genetic algorithm (selection, crossover, mutation)
render.cu        — CUDA batch renderer + loss kernels
render.c         — CPU reference renderer (used for benchmarking only)
util.c           — image I/O, RNG (xoshiro256**), profiler
Makefile
run.sbatch       — SLURM job template
scaling_study.sh — all sbatch commands for weak & strong scaling
config/          — ready-to-use config files (see below)
input/           — target PPM images at various scales (x1 … x48)
sample.ppm       — example output (800×400, target | rendered, side-by-side)
```

## Prerequisites

| Dependency | Purpose |
|------------|---------|
| `nvcc` (CUDA ≥ 10, `sm_70`) | GPU renderer |
| `mpicc` / Spectrum MPI | multi-node parallelism |
| `xl_r` (IBM XL runtime) | required on AiMOS POWER9 nodes |

## Building and running on AiMOS

### 1. Load modules

```bash
module load xl_r spectrum-mpi cuda
```

### 2. Build

```bash
make clean && make
```

### 3. Quick single-GPU run

```bash
mpirun -np 1 ./app.o config/default.conf
```

Output is written to `output/<RUN_PREFIX>_<timestamp>/`:

```
final_result.ppm      — side-by-side: target | best solution
progress_ppm/         — snapshots every VISUALISE_EVERY generations
run.log               — fitness curve + profiler report
hyperparams.txt       — full config used for this run
ga_checkpoint.bin     — checkpoint for resuming
```

### 4. Multi-GPU on one node

```bash
# 6 GPUs, single MPI rank — all GPUs used automatically (NUM_GPUS=0)
mpirun -np 1 ./app.o config/100k.conf
```

### 5. Multi-node via SLURM

```bash
sbatch --nodes=2 --ntasks-per-node=1 --gres=gpu:6 run.sbatch config/strong_scaling_2n6g.conf
```

`run.sbatch` builds the binary, then calls `mpirun -np $SLURM_NTASKS ./app.o $CONFIG`.
Resource flags (`--nodes`, `--gres`) override the sbatch defaults and are passed at
submission time — no need to edit the template.

---

## Configuration reference

Config files use `KEY = value` syntax; `#` starts a comment; blank lines are ignored.
All parameters are loaded at runtime — no recompilation needed.

| Key | Default | Description |
|-----|---------|-------------|
| `RUN_PREFIX` | `run` | Label prepended to the output folder name |
| `TARGET_FILE` | `input/target.ppm` | Path to the target PPM image |
| `N_TRIANGLES` | `300` | Number of triangles per chromosome |
| `POP_SIZE` | `100` | Population size |
| `DISC_COUNT` | `0` | Slots replaced per generation; 0 = `ceil(POP_SIZE × 0.75)` |
| `MAX_GENERATIONS` | `0` | Hard generation limit; 0 = run until stagnation |
| `STAGNATION_GENS` | `2000` | Stop after N generations with no improvement (ignored when `MAX_GENERATIONS > 0`) |
| `STAGNATION_RELATIVE` | `0` | 0 = absolute threshold, 1 = relative threshold |
| `STAGNATION_ABS_TOL` | `1e-7` | Min raw improvement to reset stagnation counter |
| `STAGNATION_REL_TOL` | `1e-4` | Min fractional improvement (relative mode) |
| `CROSSOVER_PROB` | `0.95` | Probability a discarded slot gets crossover vs. mutation |
| `ALPHA_INIT` | `0.15` | Fixed triangle transparency (not mutated) |
| `LOSS_TYPE` | `mse` | Loss function: `mse`, `l4`, `ssim`, `logll`, or `wmse` |
| `WMSE_POWER` | `0.5` | Exponent for area weight in `wmse` mode |
| `NUM_GPUS` | `0` | GPUs to use per rank; 0 = all visible GPUs |
| `MIGRATION_INTERVAL` | `0` | Gens between ring migrations; 0 = disabled |
| `MIGRATION_SIZE` | `1` | Chromosomes exchanged per migration event |
| `VISUALISE_EVERY` | `10000` | Save a side-by-side snapshot every N generations |

### Loss functions

| Name | Formula | Notes |
|------|---------|-------|
| `mse` | mean((p−t)²) per channel | Balanced default |
| `l4` | mean((p−t)⁴) per channel | Aggressively penalises worst pixels |
| `ssim` | 1 − mean(SSIM_R, SSIM_G, SSIM_B) | Structure-aware; uses 16×16 patches |
| `logll` | mean(−t·log(p+ε) − (1−t)·log(1−p+ε)) | Binary cross-entropy per channel |
| `wmse` | MSE × (total_area)^`WMSE_POWER` | Penalises large-triangle solutions |

---

## Reproducing the scaling study

All experiments use 300 triangles, population 100, and MSE loss.
Run every job at once with the helper script:

```bash
bash scaling_study.sh
```

Or submit individual jobs as shown below.

### Strong scaling

**Goal:** fixed total work (100 k generations), measure wall time as GPU count grows.

The `MAX_GENERATIONS` budget is automatically divided by the number of MPI ranks, so
each island always does `100000 / world_size` generations.

| Config | Nodes | GPUs/node | Total GPUs | MPI ranks | Gens/island |
|--------|-------|-----------|------------|-----------|-------------|
| `strong_scaling_1n1g.conf` | 1 | 1 | 1 | 1 | 100 000 |
| `strong_scaling_1n2g.conf` | 1 | 2 | 2 | 1 | 100 000 |
| `strong_scaling_1n4g.conf` | 1 | 4 | 4 | 1 | 100 000 |
| `strong_scaling_1n6g.conf` | 1 | 6 | 6 | 1 | 100 000 |
| `strong_scaling_2n6g.conf` | 2 | 6 | 12 | 2 | 50 000 |
| `strong_scaling_4n6g.conf` | 4 | 6 | 24 | 4 | 25 000 |
| `strong_scaling_8n6g.conf` | 8 | 6 | 48 | 8 | 12 500 |

```bash
sbatch --nodes=1 --ntasks-per-node=1 --gres=gpu:1 run.sbatch config/strong_scaling_1n1g.conf
sbatch --nodes=1 --ntasks-per-node=1 --gres=gpu:2 run.sbatch config/strong_scaling_1n2g.conf
sbatch --nodes=1 --ntasks-per-node=1 --gres=gpu:4 run.sbatch config/strong_scaling_1n4g.conf
sbatch --nodes=1 --ntasks-per-node=1 --gres=gpu:6 run.sbatch config/strong_scaling_1n6g.conf
sbatch --nodes=2 --ntasks-per-node=1 --gres=gpu:6 run.sbatch config/strong_scaling_2n6g.conf
sbatch --nodes=4 --ntasks-per-node=1 --gres=gpu:6 run.sbatch config/strong_scaling_4n6g.conf
sbatch --nodes=8 --ntasks-per-node=1 --gres=gpu:6 run.sbatch config/strong_scaling_8n6g.conf
```

All strong-scaling configs use `input/strong_scaling.ppm` as the target.

### Weak scaling

**Goal:** keep work-per-GPU constant (100 k generations/island) while growing the
image proportionally to total GPU count.

`MAX_GENERATIONS` is pre-scaled in each config so that after dividing by
`world_size` each island still runs 100 k generations.

| Config | Nodes | GPUs/node | Total GPUs | Target image | MAX_GENERATIONS |
|--------|-------|-----------|------------|--------------|-----------------|
| `weak_scaling_1n1g.conf` | 1 | 1 | 1 | `x1` | 100 000 |
| `weak_scaling_1n2g.conf` | 1 | 2 | 2 | `x2` | 100 000 |
| `weak_scaling_1n4g.conf` | 1 | 4 | 4 | `x4` | 100 000 |
| `weak_scaling_1n6g.conf` | 1 | 6 | 6 | `x6` | 100 000 |
| `weak_scaling_2n6g.conf` | 2 | 6 | 12 | `x12` | 200 000 |
| `weak_scaling_4n6g.conf` | 4 | 6 | 24 | `x24` | 400 000 |
| `weak_scaling_8n6g.conf` | 8 | 6 | 48 | `x48` | 800 000 |

```bash
sbatch --nodes=1 --ntasks-per-node=1 --gres=gpu:1 run.sbatch config/weak_scaling_1n1g.conf
sbatch --nodes=1 --ntasks-per-node=1 --gres=gpu:2 run.sbatch config/weak_scaling_1n2g.conf
sbatch --nodes=1 --ntasks-per-node=1 --gres=gpu:4 run.sbatch config/weak_scaling_1n4g.conf
sbatch --nodes=1 --ntasks-per-node=1 --gres=gpu:6 run.sbatch config/weak_scaling_1n6g.conf
sbatch --nodes=2 --ntasks-per-node=1 --gres=gpu:6 run.sbatch config/weak_scaling_2n6g.conf
sbatch --nodes=4 --ntasks-per-node=1 --gres=gpu:6 run.sbatch config/weak_scaling_4n6g.conf
sbatch --nodes=8 --ntasks-per-node=1 --gres=gpu:6 run.sbatch config/weak_scaling_8n6g.conf
```

The `xN` images in `input/` are the same base photograph scaled to N× the area of
`target_area_x1.ppm`.

---

## Multi-node migration

When using more than one MPI rank, enable ring migration to share good solutions
between islands:

```
MIGRATION_INTERVAL = 100   # exchange every 100 generations
MIGRATION_SIZE     = 1     # 1 chromosome per direction
```

Received chromosomes replace the worst slots and are immediately re-evaluated before
the next GA step.  The multi-node scaling configs already include these settings.
