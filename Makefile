# ── Toolchain ─────────────────────────────────────────────────────────────────
#
# Build (on AiMOS):
#   module load xl_r spectrum-mpi cuda
#   make
#
# Run:
#   mpirun -np <N> ./app.o [config_file]
#   or use run.sbatch for SLURM submission

MPICC     = mpicc
MPI_FLAGS = -O2 -Wall -Wextra -std=c11

NVCC       = nvcc
NVCC_FLAGS = -O2 -arch=sm_70

# CUDA runtime libs — mpicc does not add these automatically
CUDA_LIB  = -L/usr/local/cuda/lib64 -lcudadevrt -lcudart -lstdc++ -lm

TARGET = app.o

# ── Build rules ────────────────────────────────────────────────────────────────
all: $(TARGET)

# Link: mpicc brings in MPI libs; we add CUDA libs explicitly
$(TARGET): main.o ga.o util.o render_cu.o
	$(MPICC) $(MPI_FLAGS) $^ -o $@ $(CUDA_LIB)

main.o: main.c app.h
	$(MPICC) $(MPI_FLAGS) -c main.c -o main.o

ga.o: ga.c app.h
	$(MPICC) $(MPI_FLAGS) -c ga.c -o ga.o

util.o: util.c app.h
	$(MPICC) $(MPI_FLAGS) -c util.c -o util.o

# nvcc compiles the CUDA translation unit; mpicc links it
render_cu.o: render.cu app.h
	$(NVCC) $(NVCC_FLAGS) -c render.cu -o render_cu.o

clean:
	rm -f main.o ga.o util.o render_cu.o $(TARGET)

.PHONY: all clean
