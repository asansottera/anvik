CC=g++ -std=c++11 -O2 -march=native -fopenmp -DNDEBUG -g -c
LINK=g++ -fopenmp
#CC=icpc -std=c++11 -O2 -march=native -fopenmp -DNDEBUG -g -c
#LINK=icpc -fopenmp -L/opt/intel/lib/intel64
NVCC=nvcc -c -O2 -DNDEBUG -g -arch=sm_30

CUDA=/usr/local/cuda

all: lib/anvik.a lib/anvik-gpu.a lib/anvik-sim.a lib/anvik-evaluator.a bin/anvik-run bin/anvik-run-paper-1

lib/anvik.a: anvik/obj/comb_with_rep.o anvik/obj/examples.o anvik/obj/full_model.o anvik/obj/cpu_optimizer.o anvik/obj/problem.o anvik/obj/problem_io.o
	ar rcs lib/anvik.a anvik/obj/comb_with_rep.o anvik/obj/examples.o anvik/obj/full_model.o anvik/obj/cpu_optimizer.o anvik/obj/problem.o anvik/obj/problem_io.o

anvik/obj/comb_with_rep.o: anvik/comb_with_rep.cpp anvik/*.h
	$(CC) anvik/comb_with_rep.cpp -o $@

anvik/obj/examples.o: anvik/examples.cpp anvik/*.h
	$(CC) anvik/examples.cpp -o $@

anvik/obj/full_model.o: anvik/full_model.cpp anvik/*.h
	$(CC) anvik/full_model.cpp -o $@

anvik/obj/cpu_optimizer.o: anvik/cpu_optimizer.cpp anvik/*.h
	$(CC) anvik/cpu_optimizer.cpp -o $@

anvik/obj/problem.o: anvik/problem.cpp anvik/*.h
	$(CC) anvik/problem.cpp -o $@

anvik/obj/problem_io.o: anvik/problem_io.cpp anvik/*.h
	$(CC) anvik/problem_io.cpp -o $@

lib/anvik-gpu.a: lib/anvik.a anvik-gpu/obj/gpu_analysis.o anvik-gpu/obj/gpu_problem.o anvik-gpu/obj/gpu_optdata.o anvik-gpu/obj/gpu_optimize.o anvik-gpu/obj/gpu_reduction.o anvik-gpu/obj/gpu_optimizer.o
	ar rcs lib/anvik-gpu.a\
		anvik-gpu/obj/gpu_analysis.o\
		anvik-gpu/obj/gpu_problem.o\
		anvik-gpu/obj/gpu_optdata.o\
		anvik-gpu/obj/gpu_optimize.o\
		anvik-gpu/obj/gpu_reduction.o\
		anvik-gpu/obj/gpu_optimizer.o

anvik-gpu/obj/gpu_analysis.o: anvik-gpu/gpu_analysis.cpp anvik-gpu/*.h anvik/*.h
	$(CC) -Ianvik/ -I$(CUDA)/include anvik-gpu/gpu_analysis.cpp -o $@

anvik-gpu/obj/gpu_problem.o: anvik-gpu/gpu_problem.cpp anvik-gpu/*.h anvik/*.h
	$(CC) -Ianvik/ -I$(CUDA)/include anvik-gpu/gpu_problem.cpp -o $@

anvik-gpu/obj/gpu_optdata.o: anvik-gpu/gpu_optdata.cpp anvik-gpu/*.h anvik/*.h
	$(CC) -Ianvik/ -I$(CUDA)/include anvik-gpu/gpu_optdata.cpp -o $@

anvik-gpu/obj/gpu_optimize.o: anvik-gpu/gpu_optimize.cu anvik-gpu/*.h anvik/*.h
	$(NVCC) --maxrregcount 48 -Xptxas="-v" -Ianvik/ anvik-gpu/gpu_optimize.cu -o $@

anvik-gpu/obj/gpu_reduction.o: anvik-gpu/gpu_reduction.cu anvik-gpu/*.h anvik/*.h
	$(NVCC) -Ianvik/ anvik-gpu/gpu_reduction.cu -o $@

anvik-gpu/obj/gpu_optimizer.o: anvik-gpu/gpu_optimizer.cpp anvik-gpu/*.h anvik/*.h
	$(CC) -Ianvik/ -I$(CUDA)/include anvik-gpu/gpu_optimizer.cpp -o $@

bin/anvik-run: lib/anvik.a lib/anvik-gpu.a lib/anvik-evaluator.a anvik-run/obj/main.o
	$(LINK) -L$(CUDA)/lib64 -lcudart anvik-run/obj/main.o lib/anvik.a lib/anvik-gpu.a lib/anvik-evaluator.a lib/anvik-sim.a -o $@

anvik-run/obj/main.o: anvik-run/main.cpp anvik/*.h anvik-gpu/*.h anvik-sim/*.h anvik-evaluator/*.h
	$(CC) -Ianvik/ -Ianvik-gpu/ -Ianvik-sim/ -Ianvik-evaluator/ anvik-run/main.cpp -o $@

bin/anvik-run-paper-1: lib/anvik.a lib/anvik-gpu.a lib/anvik-evaluator.a anvik-run-paper-1/obj/main.o
	$(LINK) -L$(CUDA)/lib64 -lcudart anvik-run-paper-1/obj/main.o lib/anvik.a lib/anvik-gpu.a lib/anvik-evaluator.a -o $@

anvik-run-paper-1/obj/main.o: anvik-run-paper-1/main.cpp anvik/*.h anvik-gpu/*.h
	$(CC) -Ianvik/ -Ianvik-gpu/ anvik-run-paper-1/main.cpp -o $@

lib/anvik-sim.a: lib/anvik.a anvik-sim/obj/simulator.o
	ar rcs lib/anvik-sim.a anvik-sim/obj/simulator.o

anvik-sim/obj/simulator.o: anvik-sim/simulator.cpp anvik-sim/*.h anvik/*.h
	$(CC) -Ianvik/ -Ianvik-sim/ anvik-sim/simulator.cpp -o $@

lib/anvik-evaluator.a: lib/anvik.a anvik-evaluator/obj/evaluator.o
	ar rcs lib/anvik-evaluator.a anvik-evaluator/obj/evaluator.o

anvik-evaluator/obj/evaluator.o: anvik-evaluator/evaluator.cpp anvik-evaluator/*.h anvik/*.h
	$(CC) -Ianvik/ -I/opt/eigen-3_2_0 -Ianvik-evaluator/ anvik-evaluator/evaluator.cpp -o $@

clean:
	rm -f anvik/obj/*.o
	rm -f anvik-gpu/obj/*.o
	rm -f anvik-run/obj/*.o
	rm -f anvik-sim/obj/*.o
	rm -f bin/*
	rm -f lib/*
