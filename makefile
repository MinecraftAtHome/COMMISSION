CUBIOMES_SRC := $(addprefix cubiomes/,biomenoise.c biomes.c finders.c generator.c layers.c noise.c)

LARGE_BIOMES ?= 0
UNBOUND ?= 0

ifeq ($(LARGE_BIOMES),1)
    BIN_SUFFIX := lb
else
    BIN_SUFFIX := sb
endif

ifeq ($(UNBOUND),1)
    BIN_SUFFIX := $(BIN_SUFFIX)u
endif

# BOINC=1 builds the BOINC client app (checkpointing, progress reporting,
# result file via the BOINC API). Requires --start and --end at runtime.
BOINC ?= 0
ifeq ($(BOINC),1)
    BIN_SUFFIX := $(BIN_SUFFIX)-boinc
    override CXXFLAGS += -DBOINC -I boinc
endif

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

.PHONY: all clean

ifeq ($(OS),Windows_NT)
BIN := main-$(BIN_SUFFIX).exe

ifeq ($(BOINC),1)
# The prebuilt BOINC libs use the static CRT (/MT), so everything must.
# Written as -MT: MSYS2 (used by CI) rewrites /MT into a file path.
override NVCC_FLAGS += -Xcompiler -MT
BOINC_LIBS := -L boinc/lib/win -lboinc_api -lboinc -luser32 -ladvapi32 -lshell32
endif

all: $(BIN)

SRC_CPP := $(wildcard src/*.cpp)
SRC_C   := $(wildcard src/*.c)
SRC_CU  := $(wildcard src/*.cu)
SRC     := $(SRC_CPP) $(SRC_C) $(SRC_CU)

clean:
	-del /Q main-*.exe

$(BIN): $(SRC) $(CUBIOMES_SRC)
	nvcc $(SRC) $(CUBIOMES_SRC) -o $@ $(NVCC_FLAGS) -D_WIN32_WINNT=0x0601 $(BOINC_LIBS)
else
BIN := main-$(BIN_SUFFIX)

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

ifeq ($(BOINC),1)
	MAIN_SRC += boinc_glue.o
	MAIN_DEP += boinc_glue.o src/boinc_glue.h
	MAIN_CXXFLAGS += -L boinc/lib/lin -lboinc_api -lboinc -lpthread
    # Volunteers run all sorts of distros: don't depend on their libstdc++.
    ifndef NO_GPU
        MAIN_CXXFLAGS += -Xcompiler -static-libstdc++ -Xcompiler -static-libgcc
    else
        MAIN_CXXFLAGS += -static-libstdc++ -static-libgcc
    endif
endif

all: $(BIN)

clean:
	rm -f main main-* libcubiomes.a biomenoise.o biomes.o finders.o generator.o layers.o noise.o cubiomes.o gpu.o cpu.o client.o server.o boinc_glue.o

libcubiomes.a: $(CUBIOMES_SRC)
	$(CC) -c $(CUBIOMES_SRC) -fwrapv $(CFLAGS)
	$(AR) rcs libcubiomes.a biomenoise.o biomes.o finders.o generator.o layers.o noise.o

cubiomes.o: src/cubiomes.c src/cubiomes.h
	$(CC) -c $< -o $@ $(CFLAGS)

gpu.o: src/gpu.cu src/gpu.h src/common.h src/Random.h src/kernel_0A.h src/kernel_0B.h
	nvcc -c $< -o $@ $(NVCC_FLAGS)

cpu.o: src/cpu.cpp src/cpu.h src/common.h src/cubiomes.h
	$(CXX) -c $< -o $@ $(CXXFLAGS)

client.o: src/client.cpp src/client.h src/common.h
	$(CXX) -c $< -o $@ $(CXXFLAGS)

server.o: src/server.cpp src/server.h src/common.h
	$(CXX) -c $< -o $@ $(CXXFLAGS)

boinc_glue.o: src/boinc_glue.cpp src/boinc_glue.h src/common.h
	$(CXX) -c $< -o $@ $(CXXFLAGS)

$(BIN): $(MAIN_DEP)
	$(MAIN_CXX) $(MAIN_SRC) -o $@ $(MAIN_CXXFLAGS)
endif
