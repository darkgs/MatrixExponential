
CC = gcc
FC = gfortran

C_SRCS = $(wildcard *.c)
C_OBJS = $(C_SRCS:.c=.o)

OPT_MKL = -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -L/opt/intel/lib/intel64 -liomp5 -lpthread -I/opt/intel/mkl/include/
OPT_C_OPTI = -O2 -mcmodel=medium -fopenmp -w -D_DEBUG
OPT_CL = -lOpenCL

OPT_C = $(OPT_MKL) $(OPT_CL) $(OPT_C_OPTI)
OPT_F = $(OPT_MKL) $(OPT_CL)

.phony: all

TARGET=test

%.o: %.c
	@$(CC) -c $< $(OPT)

$(TARGET): $(OBJS)
	@$(CC) -o $@ $(OBJS) $(OPT)

all: $(TARGET)

clean:
	@rm -f $(TARGET)
	@rm -f $(OBJS)
	@rm -f $(CPP_OBJS)

run: $(TARGET)
	@./$(TARGET) -f "data/DeplInfo_toy_BOC.txt"
	#@./$(TARGET) -f "data/test.txt"

f_main.o: f_main.f
	$(FC) -ffree-form -c f_main.f

f_run: $(OBJS) f_main.o
	$(FC) -o f_main f_main.o $(OBJS) -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -L/opt/intel/lib/intel64 -liomp5 -lpthread  -lOpenCL -lm -I/opt/intel/mkl/include/
	./f_main

