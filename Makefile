rwildcard = $(foreach d,$(wildcard $1*),$(call rwildcard,$d/,$2)$(filter $(subst *,%,$2),$d))

UNAME := $(shell uname -s | tr "[:upper:]" "[:lower:]")
SOURCES := $(call rwildcard, src/, *.cpp)
OBJS := $(subst src/,build/,$(SOURCES:.cpp=.o))
CU_SOURCES := $(call rwildcard, src/, *.cu)
CU_OBJS := $(subst src/,build/,$(CU_SOURCES:.cu=.cu.o))
CXX = /opt/cuda/bin/nvcc
CFLAGS = -isystem include -Isrc -std=c++11 -Wall -Wextra -Werror -Ofast -x c++
CFLAGS := --std c++11 --machine 64 --gpu-architecture=sm_52 --use_fast_math --cudart static -Xcompiler "$(CFLAGS)"
LDFLAGS = -lstdc++ -ldl -lm -lpthread -lGL -lglfw -lboost_system -lboost_filesystem -lboost_program_options
TARGET = varjo

# these might be needed
# -lXrandr -lXi -lXcursor -lXinerama
# -fopt-info-vec -fopt-info-vec-missed

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
