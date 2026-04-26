# dual-annealing

Implementation of Dual Annealing (think `scipy.optimize.dual_annealing`) in modern C++.

## Prerequisites

- `git`
- `cmake` (>= 3.9)
- A C++17-capable compiler (GCC >= 7 or Clang >= 5)

Install on Ubuntu/Debian:

```bash
sudo apt-get install git cmake g++
```

## Build

**1. Clone the repository**

```bash
git clone <repo-url>
cd dual-annealing-master
```

**2. Fetch the dependencies**

The submodules (pcg-cpp, gsl-lite, lbfgs-cpp) are downloaded manually since the
repo was distributed as a zip without `.git` history:

```bash
# pcg-cpp — random number generator
git clone https://github.com/imneme/pcg-cpp third_party/pcg-cpp

# gsl-lite — Guidelines Support Library
git clone https://github.com/martinmoene/gsl-lite third_party/gsl-lite

# lbfgs-cpp — L-BFGS local optimizer (also needs its own submodules)
git clone https://github.com/twesterhout/lbfgs-cpp third_party/lbfgs-cpp
cd third_party/lbfgs-cpp && git init && git submodule update --init && cd ../..
```

**3. Apply compatibility patches**

The code was written in 2019; three small fixes are needed for modern toolchains.

*`third_party/lbfgs-cpp/include/lbfgs/line_search.hpp`* — add missing headers after the existing includes:

```cpp
#include <algorithm>   // std::clamp
#include <functional>  // std::less
```

*`third_party/lbfgs-cpp/src/lbfgs.cpp`* — add before the first `#include`:

```cpp
#include <algorithm>
```

*`third_party/lbfgs-cpp/include/lbfgs/lbfgs.hpp`* — remove the entire `namespace gsl { ... }` block at the bottom of the file (lines ~722–751). It redefines `fail_fast_assert_handler` which is already declared in newer gsl-lite and causes a redeclaration error.

*`examples/tsallis.cpp`* — add to the includes:

```cpp
#include <memory>
```

**4. Configure and build**

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DDA_BUILD_TESTING=OFF
make -j$(nproc)
```

## Run the examples

All binaries are in `build/examples/`. Run from there:

```bash
cd build/examples
```

| Binary | Description | Command |
|--------|-------------|---------|
| `ex_1` | Hello World sanity check | `./ex_1` |
| `ex_2` | Tsallis distribution sampling — prints `x`, `log(empirical)`, `log(exact)` columns | `./ex_2 2.67 1.0 -` |
| `ex_3` | Empty stub (just links the library) | `./ex_3` |
| `ex_4` | Dual annealing on the 100-dim Rastrigin function | `./ex_4` |

Run all at once:

```bash
./ex_1 && ./ex_3 && ./ex_4 && ./ex_2 2.67 1.0 -
```

### Expected output

**ex_4 (Rastrigin)** — the optimizer should drive the function value from ~1337 down to ~0:

```
Before: f([...]) = 1337.94
After : f([...]) = 0
Number iterations: 49
Number function evaluations: 10029
Acceptance: 0.162041
```

**ex_2 (Tsallis)** — tab-separated columns: x, log(sampled density), log(exact density):

```
-9.97500e+01    -8.71564e+00    -8.00043e+00
...
```
