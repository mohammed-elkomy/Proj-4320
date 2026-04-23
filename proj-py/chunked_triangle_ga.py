"""
Low-poly triangle approximation — pluggable solver with timing profiler.

Usage
-----
    python triangle_ga.py --solver ga         # Genetic Algorithm (default)
    python triangle_ga.py --solver annealing  # scipy dual_annealing

Timing
------
A global Profiler singleton accumulates wall-clock time for two buckets:
  - "render"   : time spent inside render_triangles()
  - "optimize" : time spent in solver logic EXCLUDING rendering
                 (selection, crossover, mutation, sorting, DA overhead)

At the end a summary table is printed showing total time, call count,
and mean time per call for each bucket.
"""

import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import dual_annealing


# ── Canvas & problem size ──────────────────────────────────────────────────────
WIDTH       = 64
HEIGHT      = 64
N_TRIANGLES = 5

N_VERTEX_GENES = N_TRIANGLES * 3 * 2
N_COLOR_GENES  = N_TRIANGLES * 4
N_GENES        = N_VERTEX_GENES + N_COLOR_GENES

# ── GA hyper-parameters ────────────────────────────────────────────────────────
POP_SIZE        = 40
ELITE_FRAC      = 0.2
MUTATION_RATE   = 0.2
MUTATION_SIGMA  = 0.05
STAGNATION_GENS = 30

# ── Dual annealing hyper-parameters ───────────────────────────────────────────
DA_MAXITER = 2000
DA_SEED    = 42


# ══════════════════════════════════════════════════════════════════════════════
#  Profiler
# ══════════════════════════════════════════════════════════════════════════════

class Profiler:
    """
    Lightweight wall-clock profiler with named buckets.

    Usage
    -----
        with PROFILER.measure("render"):
            ...  # code to time

    At the end call PROFILER.report() to print a summary table.

    Design notes
    ------------
    - Uses time.perf_counter() for highest available resolution.
    - Re-entrant calls to the same bucket are supported (times accumulate).
    - The "optimize" bucket should wrap the full solver step BEFORE calling
      render, then rendering time is subtracted automatically via the
      separate "render" bucket — so the two buckets are truly non-overlapping.
    """

    def __init__(self):
        self._totals = {}   # bucket -> total seconds
        self._counts = {}   # bucket -> number of calls

    class _Context:
        """Context manager returned by measure()."""
        def __init__(self, profiler, bucket):
            self._p      = profiler
            self._bucket = bucket
            self._start  = None

        def __enter__(self):
            self._start = time.perf_counter()
            return self

        def __exit__(self, *_):
            elapsed = time.perf_counter() - self._start
            self._p._totals[self._bucket] = self._p._totals.get(self._bucket, 0.0) + elapsed
            self._p._counts[self._bucket] = self._p._counts.get(self._bucket, 0)   + 1

    def measure(self, bucket: str):
        """Return a context manager that times the enclosed block."""
        return self._Context(self, bucket)

    def report(self):
        """Print a formatted timing summary for all recorded buckets."""
        buckets = list(self._totals.keys())
        if not buckets:
            print("Profiler: no data recorded.")
            return

        total_wall = sum(self._totals.values())

        col_w = max(len(b) for b in buckets) + 2
        header = (f"{'Bucket':<{col_w}}  {'Total (s)':>10}  "
                  f"{'Calls':>8}  {'Mean (ms)':>10}  {'% of wall':>10}")
        sep    = "-" * len(header)

        print("\n" + sep)
        print("  Timing Report")
        print(sep)
        print(header)
        print(sep)

        for bucket in buckets:
            total   = self._totals[bucket]
            calls   = self._counts[bucket]
            mean_ms = (total / calls * 1000) if calls else 0.0
            pct     = (total / total_wall * 100) if total_wall else 0.0
            print(f"{bucket:<{col_w}}  {total:>10.4f}  "
                  f"{calls:>8d}  {mean_ms:>10.4f}  {pct:>9.1f}%")

        print(sep)
        print(f"{'TOTAL':<{col_w}}  {total_wall:>10.4f}")
        print(sep + "\n")


# Global profiler instance shared by all modules
PROFILER = Profiler()


# ══════════════════════════════════════════════════════════════════════════════
#  State vector helpers
# ══════════════════════════════════════════════════════════════════════════════

def vec_to_parts(x):
    """Split unified state vector -> (vertices_flat, colors)."""
    vertices_flat = x[:N_VERTEX_GENES].astype(np.float32)
    colors        = x[N_VERTEX_GENES:].reshape(N_TRIANGLES, 4).astype(np.float32)
    return vertices_flat, colors


def parts_to_vec(vertices_flat, colors):
    """Pack vertices and colors into one flat float64 state vector."""
    return np.concatenate([vertices_flat.ravel(), colors.ravel()]).astype(np.float64)


# ══════════════════════════════════════════════════════════════════════════════
#  Rendering
# ══════════════════════════════════════════════════════════════════════════════

def render_triangles(vertices_flat, colors, width=WIDTH, height=HEIGHT):
    """
    Rasterise N triangles with ordered alpha blending.
    Time spent here is recorded under the "render" profiler bucket.

    Parameters
    ----------
    vertices_flat : (N*3*2,) float32  -- normalised [0,1] vertex coords
    colors        : (N, 4)  float32  -- RGBA per triangle in [0,1]

    Returns
    -------
    (H, W, 3) float32 RGB image in [0, 1]
    """
    with PROFILER.measure("render"):
        verts    = vertices_flat.reshape(N_TRIANGLES, 3, 2)
        verts_px = verts * np.array([width, height], dtype=np.float32)

        ys, xs = np.mgrid[0:height, 0:width]
        pixels  = np.stack([xs, ys], axis=-1).reshape(-1, 2).astype(np.float32)

        canvas = np.ones((height * width, 3), dtype=np.float32)

        for i in range(N_TRIANGLES):
            tri        = verts_px[i]
            r, g, b, a = colors[i]

            # Barycentric point-in-triangle test
            v0 = tri[2] - tri[0]
            v1 = tri[1] - tri[0]
            v2 = pixels  - tri[0]

            d00 = np.dot(v0, v0)
            d01 = np.dot(v0, v1)
            d11 = np.dot(v1, v1)
            d02 = v2 @ v0
            d12 = v2 @ v1

            denom = d00 * d11 - d01 * d01
            if abs(denom) < 1e-10:
                continue

            inv    = 1.0 / denom
            u      = (d11 * d02 - d01 * d12) * inv
            v      = (d00 * d12 - d01 * d02) * inv
            inside = (u >= 0) & (v >= 0) & (u + v <= 1)

            # Alpha blending: dst = src * alpha + dst * (1 - alpha)
            src = np.array([r, g, b], dtype=np.float32)
            canvas[inside] = src * a + canvas[inside] * (1.0 - a)

        return canvas.reshape(height, width, 3)


# ══════════════════════════════════════════════════════════════════════════════
#  Loss function  (shared by both solvers)
# ══════════════════════════════════════════════════════════════════════════════

def compute_loss(x, target):
    """
    MSE between the rendering of state x and the target image.
    Rendering time is captured inside render_triangles() itself.
    """
    h, w = target.shape[:2]
    verts, colors = vec_to_parts(x)
    rendered = render_triangles(verts, colors, width=w, height=h)
    return float(np.mean((rendered - target) ** 2))


# ══════════════════════════════════════════════════════════════════════════════
#  Ground-truth builder
# ══════════════════════════════════════════════════════════════════════════════

def build_ground_truth():
    """Render a target image from 5 hand-crafted triangles."""
    gt_vertices = np.array([
        [[0.10, 0.10], [0.50, 0.05], [0.30, 0.50]],
        [[0.50, 0.20], [0.90, 0.10], [0.80, 0.60]],
        [[0.10, 0.50], [0.40, 0.90], [0.05, 0.95]],
        [[0.40, 0.40], [0.70, 0.50], [0.55, 0.85]],
        [[0.20, 0.60], [0.60, 0.70], [0.30, 0.95]],
    ], dtype=np.float32)

    gt_colors = np.array([
        [0.9, 0.2, 0.2, 0.7],
        [0.2, 0.7, 0.3, 0.7],
        [0.2, 0.3, 0.9, 0.7],
        [0.9, 0.8, 0.1, 0.6],
        [0.7, 0.1, 0.8, 0.6],
    ], dtype=np.float32)

    gt_x       = parts_to_vec(gt_vertices.flatten(), gt_colors)
    target_img = render_triangles(gt_vertices.flatten(), gt_colors)
    return target_img, gt_x


# ══════════════════════════════════════════════════════════════════════════════
#  Solver 1 — Genetic Algorithm
# ══════════════════════════════════════════════════════════════════════════════

class GeneticAlgorithm:
    """
    Evolves a population of triangle configurations to minimise MSE vs target.

    Timing
    ------
    Each call to step() is timed under the "optimize" bucket.
    Because render_triangles() is called inside step() and records itself
    under "render", the two buckets cleanly partition the wall-clock time:

        optimize_net = optimize_total - render_total
    """

    # A solver will be called for each partition so this can stay virtually the same

    def __init__(self, target, pop_size=POP_SIZE):
        self.target   = target
        self.pop_size = pop_size
        self.n_elite  = max(1, int(pop_size * ELITE_FRAC))

        self.population = [self._random_individual() for _ in range(pop_size)]
        self.fitnesses  = [self._evaluate(x) for x in self.population]

        self.generation       = 0
        self.best_loss        = min(self.fitnesses)
        self.stagnation_count = 0
        self._done            = False

    def is_done(self):
        return self._done

    def step(self):
        """
        One generation: sort -> elitism -> select -> crossover -> mutate -> eval.
        The entire method (including internal render calls) is timed under
        the "optimize" bucket.  render_triangles() also records itself under
        "render", so:
            net optimizer overhead = optimize_total - render_total
        """
        if self._done:
            return self._stats()

        with PROFILER.measure("optimize"):
            # 1. Sort ascending by MSE
            order           = np.argsort(self.fitnesses)
            self.population = [self.population[i] for i in order]
            self.fitnesses  = [self.fitnesses[i]  for i in order]

            # 2. Elitism — keep top n_elite unchanged
            new_pop  = self.population[:self.n_elite]
            new_fits = self.fitnesses[:self.n_elite]

            # 3. Crossover + mutation to fill the rest
            while len(new_pop) < self.pop_size:
                pa, pb = self._select_parents(new_pop)
                child  = self._crossover(pa, pb)
                child  = self._mutate(child)
                new_pop.append(child)
                new_fits.append(self._evaluate(child))   # <-- render called here

            self.population = new_pop
            self.fitnesses  = new_fits
            self.generation += 1

            # 4. Stagnation check
            current_best = min(self.fitnesses)
            if current_best < self.best_loss - 1e-7:
                self.best_loss        = current_best
                self.stagnation_count = 0
            else:
                self.stagnation_count += 1

            if self.stagnation_count >= STAGNATION_GENS:
                self._done = True
                print(f"[GA] Stagnation after {self.generation} generations. Stopping.")

        return self._stats()

    # ── Individual construction ───────────────────────────────────────────────

    def _random_individual(self):
        return np.random.rand(N_GENES).astype(np.float64)

    # ── Selection ────────────────────────────────────────────────────────────

    def _select_parents(self, elite_pool):
        """Pick two distinct parents from the elite pool."""
        idx_a, idx_b = np.random.choice(len(elite_pool), size=2, replace=False)
        return elite_pool[idx_a], elite_pool[idx_b]

    # ── Crossover ────────────────────────────────────────────────────────────

    def _crossover(self, pa, pb):
        """Single-point crossover on the unified flat state vector."""
        cut = np.random.randint(1, N_GENES)
        return np.concatenate([pa[:cut], pb[cut:]])

    # ── Mutation ─────────────────────────────────────────────────────────────

    def _mutate(self, x):
        """Gaussian per-gene mutation, clamped to [0, 1]."""
        child       = x.copy()
        mask        = np.random.rand(N_GENES) < MUTATION_RATE
        child[mask] += np.random.normal(0, MUTATION_SIGMA, mask.sum())
        return np.clip(child, 0.0, 1.0)

    # ── Fitness evaluation ────────────────────────────────────────────────────

    def _evaluate(self, x):
        """Render x and return MSE vs target.  Render time goes to 'render'."""
        return compute_loss(x, self.target)

    # ── Stats ─────────────────────────────────────────────────────────────────

    def _stats(self):
        best_idx = int(np.argmin(self.fitnesses))
        return {
            "generation": self.generation,
            "best_loss":  float(min(self.fitnesses)),
            "avg_loss":   float(np.mean(self.fitnesses)),
            "best_x":     self.population[best_idx],
        }


# ══════════════════════════════════════════════════════════════════════════════
#  Solver 2 — Dual Annealing (scipy)
# ══════════════════════════════════════════════════════════════════════════════

class DualAnnealingSolver:
    """
    Wraps scipy.optimize.dual_annealing.

    Timing
    ------
    The entire dual_annealing call is wrapped in the "optimize" bucket.
    render_triangles() records itself under "render" on every objective
    evaluation, so the same accounting holds as for the GA.
    """

    PRINT_EVERY = 100

    def __init__(self, target, maxiter=DA_MAXITER, seed=DA_SEED):
        self.target  = target
        self.maxiter = maxiter
        self.seed    = seed

        self._done       = False
        self._result     = None
        self._eval_count = 0
        self._best_loss  = np.inf

    def is_done(self):
        return self._done

    def step(self):
        """Run dual_annealing to completion (blocks). Timed under 'optimize'."""
        if self._done:
            return self._stats()

        bounds = [(0.0, 1.0)] * N_GENES

        def callback(x, f, context):
            self._eval_count += 1
            if f < self._best_loss:
                self._best_loss = f
            if self._eval_count % self.PRINT_EVERY == 0:
                print(f"[DA] evals={self._eval_count:5d} | best={self._best_loss:.6f}")

        print(f"[DA] Starting dual_annealing  maxiter={self.maxiter} ...")

        # Wrap the full scipy call in the "optimize" bucket.
        # render_triangles() will also fire its own "render" timing inside.
        with PROFILER.measure("optimize"):
            self._result = dual_annealing(
                func     = lambda x: compute_loss(x, self.target),
                bounds   = bounds,
                maxiter  = self.maxiter,
                seed     = self.seed,
                callback = callback,
            )

        self._done = True
        print(f"[DA] Finished.  best_loss={self._result.fun:.6f}  "
              f"total_evals={self._result.nfev}")
        return self._stats()

    def _stats(self):
        if self._result is None:
            return {"generation": 0, "best_loss": np.inf,
                    "avg_loss": np.inf, "best_x": np.zeros(N_GENES)}
        return {
            "generation": self._result.nfev,
            "best_loss":  float(self._result.fun),
            "avg_loss":   float(self._result.fun),
            "best_x":     self._result.x,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  Visualisation
# ══════════════════════════════════════════════════════════════════════════════

def show_progress(target, label, best_loss, canvas=None, x=None, save_path=None):
    rendered = canvas if canvas is not None else render_triangles(*vec_to_parts(x))
    # verts, colors = vec_to_parts(x)
    # rendered      = render_triangles(verts, colors)

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
    axes[0].imshow(target,   vmin=0, vmax=1); axes[0].set_title("Ground Truth"); axes[0].axis("off")
    axes[1].imshow(rendered, vmin=0, vmax=1)
    axes[1].set_title(f"{label}  loss={best_loss:.5f}")
    axes[1].axis("off")

    plt.tight_layout()
    path = save_path or f"progress_{label}.png"
    plt.savefig(path, dpi=80)
    #plt.show()
    plt.close(fig)


def show_final(target, canvas, solver_name, best_loss):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(target, vmin=0, vmax=1); axes[0].set_title("Target");             axes[0].axis("off")
    axes[1].imshow(canvas, vmin=0, vmax=1); axes[1].set_title(f"{solver_name} Result"); axes[1].axis("off")
    plt.suptitle(f"Final MSE loss: {best_loss:.6f}")
    plt.tight_layout()
    plt.savefig("final_result.png", dpi=120)
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
#  Runner helpers
# ══════════════════════════════════════════════════════════════════════════════

def slice_canvas(canvas, num_chunks, overlap):
    chunk_h = HEIGHT // num_chunks
    chunk_w = WIDTH  // num_chunks

    h, w, _ = canvas.shape

    overlap_h = int(chunk_h * (overlap - 1.0))
    overlap_w = int(chunk_w * (overlap - 1.0))

    stride_h = chunk_h - overlap_h
    stride_w = chunk_w - overlap_w

    chunks = []
    row = 0
    for y_start in range(0, h, stride_h):
        col = 0
        for x_start in range(0, w, stride_w):
            y_end = min(y_start + chunk_h + overlap_h, h)
            x_end = min(x_start + chunk_w + overlap_w, w)

            data_view = canvas[y_start:y_end, x_start:x_end]

            chunks.append({
                "data": data_view,
                "grid_row": row,
                "grid_col": col,
            })

            col += 1
            if x_end >= w: break
        row += 1
        if y_end >= h: break

    return chunks

def stitch_together(solvers, chunks, num_chunks):
    chunk_h = HEIGHT // num_chunks
    chunk_w = WIDTH  // num_chunks

    canvas = np.ones((HEIGHT, WIDTH, 3), dtype=np.float32)

    for solver, chunk in zip(solvers, chunks):
        r0 = chunk["grid_row"] * chunk_h
        c0 = chunk["grid_col"] * chunk_w

        stats  = solver._stats()
        best_x = stats["best_x"]

        verts, colors = vec_to_parts(best_x)
        rendered = render_triangles(verts, colors, width=chunk_w, height=chunk_h)

        canvas[r0:r0+chunk_h, c0:c0+chunk_w] = rendered

    return canvas
        
def run_ga(target, num_chunks=1, overlap=1):
    """
    Run the Genetic Algorithm and return (best_x, best_loss).
    
    Inputs:
        target: 3d numpy ndarray - the RGB target image
        num_chunks: integer - number of chunks to split target into
        overlap: float - multiplier which induces chunk overlap

    Outputs:
        final: ga object - GA reconstructed RGB image and stats

    """

    # perform partioning and initalize solvers
    chunks = slice_canvas(target, num_chunks, overlap)
    solvers = [GeneticAlgorithm(target=chunk["data"]) for chunk in chunks]

    VISUALISE_EVERY = 20
    with PROFILER.measure("total"):
        while any(not ga.is_done() for ga in solvers): # while none of the partitions are done

            for ga in solvers:
                ga.step()

            gen = solvers[0].generation
            print(f"Generation {gen} processed")
            all_done = all(ga.is_done() for ga in solvers)
            if gen % VISUALISE_EVERY == 0 or all_done:
                canvas = stitch_together(solvers, chunks, num_chunks)

                loss = float(np.mean((canvas - target) ** 2))

                per_chunk_losses = [ga._stats()["best_loss"] for ga in solvers]
                avg_chunk_loss = float(np.mean(per_chunk_losses))

                print(f"Gen {gen:4d} | stitched_mse={loss:.6f} | avg_chunk_loss={avg_chunk_loss:.6f}")

                show_progress(target, canvas=canvas, label=f"GA_gen{gen:04d}", best_loss=loss)

    # Final stitch and stats
    canvas = stitch_together(solvers, chunks, num_chunks)
    loss = float(np.mean((canvas - target) ** 2))
    print(f"\nDone. Final stitched MSE={loss:.6f}")
    return canvas, loss


def run_dual_annealing(target):
    """Run Dual Annealing and return (best_x, best_loss)."""
    da    = DualAnnealingSolver(target=target)
    stats = da.step()
    show_progress(target, stats["best_x"],
                  label="DA_final",
                  best_loss=stats["best_loss"])
    return stats["best_x"], stats["best_loss"]


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Low-poly triangle approximation with selectable solver.")
    parser.add_argument(
        "--solver",
        choices=["ga", "annealing"],
        default="ga",
        help="Solver: 'ga' (Genetic Algorithm) or 'annealing' (Dual Annealing). Default: ga",
    )
    args = parser.parse_args()

    np.random.seed(42)

    target, gt_x = build_ground_truth()
    print(f"Target image : {target.shape}")
    print(f"State vector : length {N_GENES}  "
          f"({N_TRIANGLES} triangles x (3 vertices x 2 coords + 4 RGBA))\n")
    print(f"Solver       : {args.solver}\n")

    if args.solver == "ga":
        canvas, best_loss = run_ga(target, 2, 1.2)
        solver_label = "GA"
        show_final(target, canvas, solver_label, best_loss)
    else:
        best_x, best_loss = run_dual_annealing(target)
        solver_label = "Dual Annealing"

    # show_final(target, best_x, solver_label, best_loss)
    print(f"\nDone. Final loss={best_loss:.6f}  Saved final_result.png")

    # ── Print timing breakdown ────────────────────────────────────────────────
    # "optimize" total includes render time (render is called from within it).
    # Net optimizer overhead = optimize_total - render_total.
    # render_total   = PROFILER._totals.get("render",   0.0)
    # optimize_total = PROFILER._totals.get("optimize", 0.0)
    # net_optimize   = optimize_total - render_total

    # PROFILER._totals["optimize (net, excl. render)"] = net_optimize
    # PROFILER._counts["optimize (net, excl. render)"] = PROFILER._counts.get("optimize", 0)
    # del PROFILER._totals["optimize"]
    # del PROFILER._counts["optimize"]

    PROFILER.report()


if __name__ == "__main__":
    main()
