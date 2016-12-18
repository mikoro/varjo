rwildcard = $(foreach d,$(wildcard $1*),$(call rwildcard,$d/,$2)$(filter $(subst *,%,$2),$d))

UNAME := $(shell uname -s | tr "[:upper:]" "[:lower:]")
SOURCES := $(call rwildcard, src/, *.cpp)
OBJS := $(subst src/,build/,$(SOURCES:.cpp=.o))
CU_SOURCES := $(call rwildcard, src/, *.cu)
CU_OBJS := $(subst src/,build/,$(CU_SOURCES:.cu=.cu.o))
CFLAGS = -isystem include -Isrc -std=c++14 -Wpedantic -Wall -Wextra -Werror -Ofast -fopenmp -x c++
LDFLAGS = 
TARGET = varjo

# these might be needed
# -lXrandr -lXi -lXcursor -lXinerama
# -fopt-info-vec -fopt-info-vec-missed

# linux
ifneq "$(findstring linux,$(UNAME))" ""
	LDFLAGS += -lstdc++ -ldl -lm -lpthread -lGL -lglfw -lboost_system -lboost_filesystem -lboost_program_options
endif

# mac
ifneq "$(findstring darwin,$(UNAME))" ""
	CFLAGS += -isystem /opt/local/include -isystem /opt/local/include/libomp -mmacosx-version-min=10.9
	LDFLAGS += -L/opt/local/lib -L/opt/local/lib/libomp -framework Cocoa -framework OpenGL -framework IOKit -framework CoreVideo -lstdc++ -lglfw -lboost_system-mt -lboost_filesystem-mt -lboost_program_options-mt
endif

ifdef NATIVE
	CFLAGS += -march=native
endif

ifdef CUDA
	CXX = nvcc
	CFLAGS += -DUSE_CUDA
	CFLAGS := --std c++11 --machine 64 --gpu-architecture=sm_52 --use_fast_math --cudart static -DUSE_CUDA -Xcompiler "$(CFLAGS)"
endif

default: varjo

varjo: $(OBJS) $(CU_OBJS)
	@mkdir -p bin
	@echo "Linking $@"
	@$(CXX) $(OBJS) $(CU_OBJS) $(CFLAGS) $(LDFLAGS) -o bin/$(TARGET)
	@platform/linux/post-build.sh

build/%.o: src/%.cpp
	@mkdir -p $(@D)
	@echo "Compiling $<"
	@$(CXX) $(CFLAGS) -c -o $@ $<

build/%.cu.o: src/%.cu
	@mkdir -p $(@D)
	@echo "Compiling $<"
	@$(CXX) $(CFLAGS) -c -o $@ $<

clean:
	@rm -rf bin build
