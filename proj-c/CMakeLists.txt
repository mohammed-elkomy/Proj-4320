cmake_minimum_required(VERSION 3.18)
project(triangle_ga LANGUAGES C CUDA)

set(CMAKE_C_STANDARD 11)

# Change this to match your GPU:
#   61  →  GTX 10xx
#   75  →  RTX 20xx / GTX 16xx
#   86  →  RTX 30xx
#   89  →  RTX 40xx
set(CMAKE_CUDA_ARCHITECTURES 89)

add_executable(triangle_ga
    main.c
    ga.c
    util.c
    render.cu
)

target_link_libraries(triangle_ga m)
