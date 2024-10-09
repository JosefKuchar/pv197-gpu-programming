run: compile
	./bench

compile:
	nvcc -Xptxas=-v -arch=sm_86 -O3 -o bench framework.cu
