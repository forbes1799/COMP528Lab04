COMPILER = icc
CFLAGS = -qopenmp -std=c99 -O3

hello: HelloOpenMP.c
	$(COMPILER) $(CFLAGS) HelloOpenMP.c -o hello.exe
