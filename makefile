CUBIOMES_SRC := $(addprefix cubiomes/,biomenoise.c biomes.c finders.c generator.c layers.c noise.c)

LARGE_BIOMES ?= 0
UNBOUND ?= 0
PRINT_INTERVAL ?= 256
# Auto-detect GPU architecture:
# - RTX 40xx/50xx series: sm_89 is faster than native sm_120
# - Everything else: use native
# Override manually with: make ARCH=sm_89
ifndef ARCH
  GPU_NAMES := $(shell nvidia-smi --query-gpu=name --format=csv,noheader)
  ifneq (,$(findstring RTX 40,$(GPU_NAMES)))
    ARCH := sm_89
  else ifneq (,$(findstring RTX 50,$(GPU_NAMES)))
    ARCH := sm_89
  else
    ARCH := native
  endif
endif

$(info Using ARCH = $(ARCH))
override CFLAGS += -O3
override CXXFLAGS += -O3 -std=c++20 -I asio/asio/include -DOMISSION_LARGE_BIOMES=$(LARGE_BIOMES) -DOMISSION_UNBOUND=$(UNBOUND) -DPRINT_INTERVAL=$(PRINT_INTERVAL)
override NVCC_FLAGS += $(CXXFLAGS) --expt-relaxed-constexpr --default-stream per-thread -arch=$(ARCH) -use_fast_math

ifeq ($(OS),Windows_NT)
all: main.exe

SRC_CPP := $(wildcard src/*.cpp)
SRC_C   := $(wildcard src/*.c)
SRC_CU  := $(wildcard src/*.cu)
SRC     := $(SRC_CPP) $(SRC_C) $(SRC_CU)

clean:
	del /Q main.exe

# nvcc src/*.cpp src/*.c src/*.cu -o main.exe cubiomes/biomenoise.c cubiomes/biomes.c cubiomes/finders.c cubiomes/generator.c cubiomes/layers.c cubiomes/noise.c -arch=native -O3 -std=c++20 -I asio-1.34.2/include -DOMISSION_LARGE_BIOMES=1 --expt-relaxed-constexpr --default-stream per-thread -D_WIN32_WINNT=0x0601
main.exe: $(SRC) $(CUBIOMES_SRC)
	nvcc $(SRC) $(CUBIOMES_SRC) -o $@ $(NVCC_FLAGS) -D_WIN32_WINNT=0x0601
else
override NVCC_FLAGS += -ccbin $(CXX)

MAIN_SRC := src/main.cpp
MAIN_DEP := $(MAIN_SRC) src/common.h

ifndef NO_GPU
	MAIN_SRC += gpu.o
	MAIN_DEP += gpu.o src/gpu.h
	MAIN_CXX := nvcc
	MAIN_CXXFLAGS += $(NVCC_FLAGS)
else
	MAIN_CXX := $(CXX)
	MAIN_CXXFLAGS += $(CXXFLAGS) -DNO_GPU
endif

ifndef NO_CPU
	MAIN_SRC += cpu.o cubiomes.o libcubiomes.a
	MAIN_DEP += cpu.o cubiomes.o libcubiomes.a src/cpu.h
else
	MAIN_CXXFLAGS += -DNO_CPU
endif

ifndef NO_NET
	MAIN_SRC += client.o server.o
	MAIN_DEP += client.o server.o src/client.h src/server.h
else
	MAIN_CXXFLAGS += -DNO_NET
endif

all: main

clean:
	rm -f main libcubiomes.a biomenoise.o biomes.o finders.o generator.o layers.o noise.o cubiomes.o gpu.o cpu.o client.o server.o

libcubiomes.a:
	$(CC) -c $(CUBIOMES_SRC) -fwrapv $(CFLAGS)
	$(AR) rcs libcubiomes.a biomenoise.o biomes.o finders.o generator.o layers.o noise.o

cubiomes.o: src/cubiomes.c src/cubiomes.h
	$(CC) -c $< -o $@ $(CFLAGS)

gpu.o: src/gpu.cu src/gpu.h src/common.h src/Random.h
	nvcc -c $< -o $@ $(NVCC_FLAGS)

cpu.o: src/cpu.cpp src/cpu.h src/common.h src/cubiomes.h
	$(CXX) -c $< -o $@ $(CXXFLAGS)

client.o: src/client.cpp src/client.h src/common.h
	$(CXX) -c $< -o $@ $(CXXFLAGS)

server.o: src/server.cpp src/server.h src/common.h
	$(CXX) -c $< -o $@ $(CXXFLAGS)

main: $(MAIN_DEP)
	$(MAIN_CXX) $(MAIN_SRC) -o $@ $(MAIN_CXXFLAGS)
endif
