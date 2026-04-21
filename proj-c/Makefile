CC      = gcc
CFLAGS  = -O2 -Wall -Wextra -std=c11
LDFLAGS = -lm

SRCS    = main.c ga.c render.c util.c
OBJS    = $(SRCS:.c=.o)
TARGET  = triangle_ga

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c triangle_ga.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET) progress_GA_gen*.ppm final_result.ppm

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run
