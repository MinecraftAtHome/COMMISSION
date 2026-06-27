CUBIOMES_SRC := $(addprefix cubiomes/,biomenoise.c biomes.c finders.c generator.c layers.c noise.c)

MAKEFLAGS += -j$(if $(filter Windows_NT,$(OS)),$(NUMBER_OF_PROCESSORS),$(shell getconf _NPROCESSORS_ONLN))

LARGE_BIOMES   ?= 0
UNBOUND        ?= 0
PRINT_INTERVAL ?= 256
AMD_GPU        ?= 0

# Auto-detect GPU architecture:
# - RTX 40xx/50xx series: sm_89 is faster than native sm_120
# - Everything else: use native
# Override manually with: make ARCH=sm_89
ifndef ARCH
  GPU_NAMES := $(shell nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)
  ifneq (,$(findstring RTX 40,$(GPU_NAMES)))
    ARCH := sm_89
  else ifneq (,$(findstring RTX 50,$(GPU_NAMES)))
    ARCH := sm_89
  else
    ARCH := native
  endif
endif

$(info Using ARCH = $(ARCH))

override CFLAGS   += -O3 -fwrapv
override CXXFLAGS += -O3 -std=c++20 -I asio/asio/include -DOMISSION_LARGE_BIOMES=$(LARGE_BIOMES) -DOMISSION_UNBOUND=$(UNBOUND) -DPRINT_INTERVAL=$(PRINT_INTERVAL)
override NVCC_FLAGS  += $(CXXFLAGS) --expt-relaxed-constexpr --default-stream per-thread -arch=$(ARCH) -use_fast_math
override HIPCC_FLAGS += $(CXXFLAGS) --offload-arch=native -ffast-math -fcuda-flush-denormals-to-zero -munsafe-fp-atomics

BUILD_DIR := build
OBJ_DIR   := $(BUILD_DIR)/obj

ifeq ($(OS),Windows_NT)
    SRC_CPP := $(wildcard src/*.cpp)
    SRC_C   := $(wildcard src/*.c)
    ifeq ($(AMD_GPU),1)
        SRC_GPU := src/gpu.hip
    else
        SRC_GPU := src/gpu.cu
    endif
    SRC := $(SRC_CPP) $(SRC_C) $(SRC_GPU)
    SRC_HEADERS := $(wildcard src/*.h)

    ifeq ($(MSYSTEM),)
        MKDIR = if not exist $(subst /,\\,$(1)) mkdir $(subst /,\\,$(1))
        RMDIR = if exist $(subst /,\\,$(1)) rmdir /s /q $(subst /,\\,$(1))
    else
        MKDIR = mkdir -p $(1)
        RMDIR = rm -rf $(1)
    endif
    TARGET := $(BUILD_DIR)/main.exe
else
    MKDIR = mkdir -p $(1)
    RMDIR = rm -rf $(1)
    TARGET := $(BUILD_DIR)/main
endif

.PHONY: all clean

all: $(TARGET)

ifeq ($(OS),Windows_NT)
$(TARGET): $(SRC) $(SRC_HEADERS) $(CUBIOMES_SRC)
	$(call MKDIR,$(BUILD_DIR))
	$(if $(filter 1,$(AMD_GPU)),\
        hipcc $(SRC) $(CUBIOMES_SRC) -o $@ $(HIPCC_FLAGS) -D_WIN32_WINNT=0x0601,\
        nvcc $(SRC) $(CUBIOMES_SRC) -o $@ $(NVCC_FLAGS) -D_WIN32_WINNT=0x0601)
else
    override NVCC_FLAGS += -ccbin $(CXX)
    MAIN_OBJ_NAMES := main.o
    CUBIOMES_LIB   := $(OBJ_DIR)/libcubiomes.a
    CUBIOMES_OBJS   := $(addprefix $(OBJ_DIR)/,$(notdir $(CUBIOMES_SRC:.c=.o)))

    ifndef NO_GPU
        MAIN_OBJ_NAMES += gpu.o
        ifeq ($(AMD_GPU),1)
            MAIN_CXX      := hipcc
            MAIN_CXXFLAGS += $(HIPCC_FLAGS)
            GPU_SRC       := src/gpu.hip
        else
            MAIN_CXX      := nvcc
            MAIN_CXXFLAGS += $(NVCC_FLAGS)
            GPU_SRC       := src/gpu.cu
        endif
    else
        MAIN_CXX      := $(CXX)
        MAIN_CXXFLAGS += $(CXXFLAGS) -DNO_GPU
    endif

    ifndef NO_CPU
        MAIN_OBJ_NAMES += cpu.o cubiomes.o
        EXTRA_DEPS     += $(CUBIOMES_LIB)
    else
        MAIN_CXXFLAGS  += -DNO_CPU
    endif

    ifndef NO_NET
        MAIN_OBJ_NAMES += client.o server.o
    else
        MAIN_CXXFLAGS  += -DNO_NET
    endif

    MAIN_OBJS := $(addprefix $(OBJ_DIR)/, $(MAIN_OBJ_NAMES))

    $(MAIN_OBJS) $(CUBIOMES_LIB) $(CUBIOMES_OBJS): | $(OBJ_DIR)
    $(TARGET): | $(BUILD_DIR)

    $(OBJ_DIR) $(BUILD_DIR):
		$(call MKDIR,$@)

    $(TARGET): $(MAIN_OBJS) $(EXTRA_DEPS)
		$(MAIN_CXX) $(MAIN_OBJS) $(if $(filter $(CUBIOMES_LIB),$(EXTRA_DEPS)),$(CUBIOMES_LIB),) -o $@ $(MAIN_CXXFLAGS)

    $(CUBIOMES_LIB): $(CUBIOMES_OBJS)
		$(AR) rcs $@ $(CUBIOMES_OBJS)

    $(CUBIOMES_OBJS): $(OBJ_DIR)/%.o: cubiomes/%.c
		$(CC) -c $< -o $@ $(CFLAGS)

    $(OBJ_DIR)/cubiomes.o: src/cubiomes.c src/cubiomes.h
		$(CC) -c $< -o $@ $(CFLAGS)

    $(OBJ_DIR)/gpu.o: $(GPU_SRC) src/gpu.h src/common.h src/Random.h src/kernel_0A.h src/kernel_0B.h
		$(MAIN_CXX) -c $< -o $@ $(MAIN_CXXFLAGS)

    $(OBJ_DIR)/cpu.o: src/cpu.cpp src/cpu.h src/common.h src/cubiomes.h
		$(CXX) -c $< -o $@ $(CXXFLAGS)

    $(OBJ_DIR)/client.o: src/client.cpp src/client.h src/common.h
		$(CXX) -c $< -o $@ $(CXXFLAGS)

    $(OBJ_DIR)/server.o: src/server.cpp src/server.h src/common.h
		$(CXX) -c $< -o $@ $(CXXFLAGS)

    $(OBJ_DIR)/main.o: src/main.cpp src/common.h
		$(MAIN_CXX) -c $< -o $@ $(MAIN_CXXFLAGS)
endif

clean:
	$(call RMDIR,$(BUILD_DIR))