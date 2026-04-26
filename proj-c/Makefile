# ── Toolchain ─────────────────────────────────────────────────────────────────
NVCC  = /usr/local/cuda/bin/nvcc
CC    = gcc

ARCH      = sm_89
NVCCFLAGS = -O2 -arch=$(ARCH)
CFLAGS    = -O2 -Wall -Wextra -std=c11
LDFLAGS   = -lm

TARGET = triangle_ga

# ── Build rules ────────────────────────────────────────────────────────────────
all: $(TARGET)

$(TARGET): main.o ga.o util.o render_cu.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS)

main.o: main.c triangle_ga.h
	$(CC) $(CFLAGS) -c main.c -o main.o

ga.o: ga.c triangle_ga.h
	$(CC) $(CFLAGS) -c ga.c -o ga.o

util.o: util.c triangle_ga.h
	$(CC) $(CFLAGS) -c util.c -o util.o

# Explicit rule avoids the %.cu.o pattern ambiguity that caused the warning
render_cu.o: render.cu triangle_ga.h
	$(NVCC) $(NVCCFLAGS) -c render.cu -o render_cu.o

clean:
	rm -f main.o ga.o util.o render_cu.o $(TARGET) progress_GA_gen*.ppm final_result.ppm

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run
