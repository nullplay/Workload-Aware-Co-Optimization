# Put the maximum number of core on your platform.
NUMCORE=$$(nproc --all)
#NUMCORE=12

gccspmm :	
	g++ -std=c++17 -O3 -march=native -fopenmp -DGCC -DNUMCORE=${NUMCORE} -I"./include" spmm.cc -o spmm -lm -ldl
gccspmv :	
	g++ -std=c++17 -O3 -march=native -fopenmp -DGCC -DNUMCORE=${NUMCORE} -I"./include" spmv.cc -o spmv -lm -ldl
gccsddmm :	
	g++ -std=c++17 -O3 -march=native -fopenmp -DGCC -DNUMCORE=${NUMCORE} -I"./include" sddmm.cc -o sddmm -lm -ldl
iccspmm :	
	icpc -std=c++17 -O3 -march=native -qopenmp -DICC -DNUMCORE=${NUMCORE} -I"./include" spmm.cc -o spmm -lm -ldl 
iccspmv :	
	icpc -std=c++17 -O3 -march=native -qopenmp -DICC -DNUMCORE=${NUMCORE} -I"./include" spmv.cc -o spmv -lm -ldl 
iccsddmm :	
	icpc -std=c++17 -O3 -march=native -qopenmp -DICC -DNUMCORE=${NUMCORE} -I"./include" sddmm.cc -o sddmm -lm -ldl

gcc : gccspmm gccspmv gccsddmm
icc : iccspmm iccspmv iccsddmm
