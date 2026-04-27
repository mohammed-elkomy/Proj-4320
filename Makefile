# ── Toolchain ─────────────────────────────────────────────────────────────────
NVCC  = /usr/local/cuda/bin/nvcc
CC    = gcc

ARCH      = sm_70
NVCCFLAGS = -O2 -arch=$(ARCH)
CFLAGS    = -O2 -Wall -Wextra -std=c11
LDFLAGS   = -lm

TARGET = app.o

# ── Build rules ────────────────────────────────────────────────────────────────
all: $(TARGET)

$(TARGET): main.o ga.o util.o render_cu.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS)

main.o: main.c app.h
	$(CC) $(CFLAGS) -c main.c -o main.o

ga.o: ga.c app.h
	$(CC) $(CFLAGS) -c ga.c -o ga.o

util.o: util.c app.h
	$(CC) $(CFLAGS) -c util.c -o util.o

# Explicit rule avoids the %.cu.o pattern ambiguity that caused the warning
render_cu.o: render.cu app.h
	$(NVCC) $(NVCCFLAGS) -c render.cu -o render_cu.o

clean:
	rm -f main.o ga.o util.o render_cu.o $(TARGET)

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run

