CUBIOMES_SRC := $(addprefix cubiomes/,biomenoise.c biomes.c finders.c generator.c layers.c noise.c)

LARGE_BIOMES ?= 0
UNBOUND ?= 0
PRINT_INTERVAL ?= 256
BLACKWELL_GPU := $(shell { nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | grep -Eq '^(10|12)\.' || nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -Eiq 'Blackwell|GeForce RTX 50[0-9]+|GB10|GB20[0-9]|B[23]00'; } && echo 1)
ifeq ($(origin ARCH), undefined)
ifeq ($(BLACKWELL_GPU),1)
ARCH := sm_89
else
ARCH := native
endif
endif
override CFLAGS += -O3
override CXXFLAGS += -O3 -std=c++20 -I asio/asio/include -DOMISSION_LARGE_BIOMES=$(LARGE_BIOMES) -DOMISSION_UNBOUND=$(UNBOUND) -DPRINT_INTERVAL=$(PRINT_INTERVAL)
override NVCC_FLAGS += $(CXXFLAGS) --expt-relaxed-constexpr --default-stream per-thread -arch=$(ARCH)

ifeq ($(OS),Windows_NT)
all: main.exe

# nvcc src/*.cpp src/*.c src/*.cu -o main.exe cubiomes/biomenoise.c cubiomes/biomes.c cubiomes/finders.c cubiomes/generator.c cubiomes/layers.c cubiomes/noise.c -arch=native -O3 -std=c++20 -I asio-1.34.2/include -DOMISSION_LARGE_BIOMES=1 --expt-relaxed-constexpr --default-stream per-thread -D_WIN32_WINNT=0x0601
main.exe: src/*.*
	nvcc src/*.cpp src/*.c src/*.cu $(CUBIOMES_SRC) -o $@ $(NVCC_FLAGS) -D_WIN32_WINNT=0x0601
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
