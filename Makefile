
COMPILER = g++

SRCS = $(wildcard *.c)
OBJS = $(SRCS:.c=.o)

OPT_MKL = -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -L/opt/intel/lib/intel64 -liomp5 -lpthread 
OPT = -O2 $(OPT_MKL) -lOpenCL -lm -mcmodel=medium -fopenmp -w -I/opt/intel/mkl/include/ -D_DEBUG

.phony: all

TARGET=test

%.o: %.c
	@$(COMPILER) -c $< $(OPT)

$(TARGET): $(OBJS)
	@$(COMPILER) -o $@ $(OBJS) $(OPT)

all: $(TARGET)

clean:
	@rm -f $(TARGET)
	@rm -f $(OBJS)
	@rm -f $(CPP_OBJS)

run: $(TARGET)
	@./$(TARGET) -f "data/DeplInfo_toy_BOC.txt"
	#@./$(TARGET) -f "data/test.txt"

