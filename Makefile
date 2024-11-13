run: compile
	./bench

compile:
	nvcc -Xptxas="-v" -o bench framework.cu
