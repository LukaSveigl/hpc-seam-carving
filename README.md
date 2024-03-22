# hpc-seam-carving
A repository for the first assignment (Parallel seam carving) of the High performance computing course. 

## Project structure

The project consists of the following 3 files, each containing a different implementation of the algorithm:
- The `scarving.cpp` contains the sequential implementation of the Seam Carving algorithm.
- The `pscarving.cpp` contains the basic parallel implementation of the algorithm, where only the energy calculation and seam removal have been parallelized.
- The `upscarving.cpp` file contains the upgraded parallel implementation of the algorithm, where the greedy approach is used in finding and removal of seams (all seams are found and removed in parallel).

The input data should be placed in the `data/input/` directory, and the data will be output into either `data/output/seq/` if the algorithm is sequential (the scarving file) or into `data/output/par/` if the algorithm is parallel (pscarving and upscarving).

## Compiling and running

To use this project locally, compile either the `scarving.cpp`, `pscarving.cpp` or `upscarving.cpp` using the following commands:

| File           | Command                                                       |
|----------------|---------------------------------------------------------------|
| scarving.cpp   | `g++ -I ./lib -O2 scarving.cpp -o scarving.exe`               |
| pscarving.cpp  | `g++ -I ./lib -O2 --fopenmp pscarving.cpp -o pscarving.exe`   |
| upscarving.cpp | `g++ -I ./lib -O2 --fopenmp upscarving.cpp -o upscarving.exe` |

When the file is compiled, it expects a path to the image and the number of columns to remove, therefore to run, use: `*scarving.cpp <path-to-image> <num-to-remove>`.
