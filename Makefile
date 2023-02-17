COMMON_FLAGS=-O3 #-g

CXXFLAGS=$(COMMON_FLAGS)
NVCCFLAGS=$(COMMON_FLAGS) -gencode arch=compute_80,code=sm_80 #-G

LDFLAGS =

SRCDIR=src
CU_SRCS=$(shell find $(SRCDIR) -name '*.cu')
OBJS=$(CU_SRCS)
CPP_SRCS=$(shell find $(SRCDIR) -name '*.cpp')
OBJS+=$(CPP_SRCS)
OBJS:=$(subst .cpp,.o,$(OBJS))
OBJS:=$(subst .cu,.o,$(OBJS))
TARGET=histogram

.SUFFIXES: .o

.PHONY: all
all:$(TARGET)

$(TARGET): $(OBJS)
	nvcc $(LDFLAGS) $(OBJS) -o $@

test: src/test.cu
	nvcc $(LDFLAGS) $(NVCCFLAGS) src/test.cu -o $@

%.o: %.cpp
	nvcc -c $(CXXFLAGS) $< -o $@

%.o: %.cu
	nvcc -c $(NVCCFLAGS) $< -o $@ 


.PHONY: clean
clean:
	rm -f $(OBJS) $(TARGET)