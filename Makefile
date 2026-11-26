# Build targets
#   make            bin/audio_analyzer (needs the CUDA toolkit: nvcc, cuFFT, NPP)
#   make test       host-side unit tests (plain g++, no GPU needed)
#   make reference_check   CPU reference vs. dataset labels (no GPU needed)
#   make run        ./run.sh (data -> build -> run -> plots -> proof tarball)
#   make clean
#
# Options: CUDA_PATH=/path/to/cuda  ARCH=sm_75  NVCC=/path/to/nvcc

CUDA_PATH ?= /usr/local/cuda
NVCC ?= $(if $(wildcard $(CUDA_PATH)/bin/nvcc),$(CUDA_PATH)/bin/nvcc,nvcc)
CXX ?= g++

# sm_60 SASS + compute_60 PTX runs on every GPU from Pascal on (newer GPUs
# JIT-compile the PTX). CUDA 13 dropped sm_60, so use sm_75 there.
NVCC_MAJOR := $(shell $(NVCC) --version 2>/dev/null | sed -n 's/.*release \([0-9]*\)\..*/\1/p')
ARCH ?= $(if $(filter 13 14 15,$(NVCC_MAJOR)),sm_75,sm_60)

CXXFLAGS ?= -std=c++14 -O3 -Wall -Wextra
NVCCFLAGS ?= -std=c++14 -O3 -arch=$(ARCH) -Xcompiler -Wall,-Wextra
INCLUDES := -Isrc
LDLIBS := -lcufft -lnppc -lnpps

HOST_LIB_SRC := src/cli_options.cc src/cpu_reference.cc src/features.cc \
                src/file_util.cc src/labels.cc src/wav_io.cc
CUDA_SRC := src/gpu_analyzer.cu src/kernels.cu

HOST_LIB_OBJ := $(patsubst src/%.cc,build/%.o,$(HOST_LIB_SRC))
CUDA_OBJ := $(patsubst src/%.cu,build/%.cu.o,$(CUDA_SRC))
MAIN_OBJ := build/main.o

BIN := bin/audio_analyzer

.PHONY: all test reference_check run clean

all: $(BIN)

$(BIN): $(MAIN_OBJ) $(HOST_LIB_OBJ) $(CUDA_OBJ) | bin
	$(NVCC) -arch=$(ARCH) $^ -o $@ $(LDLIBS)

build/%.o: src/%.cc src/*.h | build
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

build/%.cu.o: src/%.cu src/*.h | build
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

bin/host_test: tests/host_test.cc $(HOST_LIB_SRC) src/*.h | bin
	$(CXX) $(CXXFLAGS) $(INCLUDES) tests/host_test.cc $(HOST_LIB_SRC) -o $@

bin/reference_check: tests/reference_check.cc $(HOST_LIB_SRC) src/*.h | bin
	$(CXX) $(CXXFLAGS) $(INCLUDES) tests/reference_check.cc $(HOST_LIB_SRC) -o $@

test: bin/host_test
	./bin/host_test

reference_check: bin/reference_check
	./bin/reference_check data/input

run:
	./run.sh

bin build:
	mkdir -p $@

clean:
	rm -rf bin build
