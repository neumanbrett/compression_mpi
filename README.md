# Compression Testing for MPI transfers

Testbed for sending sparse matrices using MPI on distributed systems.


# Compiling

Two versions of the code exist, the serial version and the MPI version.  To compile the serial version, any standard C++ compiler like GNU, Intel, or Cray should work.  The runs on this were using Intel’s compiler with the command:

	icpc compression_serial.cpp -o comp_serial -std=c++11

Replace icpc with g++ if using GNU compiler.

The MPI version was compiled using the MPI wrapper with the command:

	mpic++ compression_mpi.cpp -o comp_mpi -std=c++11

The debug verions will give you a lot of extra information about the matrix values and non-zero element indices.  It is recommended to output this to a file rather than to standard output.

    icpc compression_serial.cpp -o comp_serial -std=c++11 -DDEBUG
    mpic++ compression_mpi.cpp -o comp_mpi -std=c++11 -DDEBUG

There are three variables that the user can modify to perform different tests:

1.	nx: the grid size in the X direction
2.	ny: the grid size in the Y direction
3.	bernoulli_p: the probability that a non-zero element will be placed at the current location based on the main grid loop


# Running

To run the serial application:
	
    ./comp_serial

To run the MPI application, you will need two MPI ranks available.  For the runs in this paper it was data transferred between two separate nodes connected with Mellanox cables.  To run the MPI version:

	mpirun -np 2 comp_mpi


# Compression Algorithm

A modified version of the Compressed Sparse Row (CSR) sometimes referred to as COOrdinate is implemented in both versions.  The algorithm takes the original matrix, checks if the current element is a non-zero, if there is a non-zero element, the value and index are stored in two separate matrices.  One of the matrices is the “State” matrix that holds all of the values and the other is the “Index” matrix that contains the location of the non-zero element for decoding at the destination node.  The data matrices (compressed and uncompressed) are stored in a block of continguous memory so iterating over it requires some finesse of the row and column values.


# Methodology

There are a few prerequisites before we can measure the transfer times between uncompressed and compressed datatypes:

1.	Generate a sparse matrix with parameter for sparsity
  a.	Calculation for sparsity
  b.	Random number generator for non-zero element
      i.	Bernoulli distribution
  c.	Random number generator for value of non-zero element
      i.	Uniform distribution with defined range
  d.	Create unique seed value to ensure random result for each run
2.	Create MPI domain and determine domain decomposition type

The program steps are:	
1.	Create 2D Matrix
  a.	User defined variables for X and Y grid size
2.	Populate matrix based on number of non-zero elements and the random value for non-zero elements
  a.	User defined variable for Bernoulli distribution to determine sparsity of matrix
3.	Determine and verify sparsity
4.	Distribute matrix to multiple nodes or cores
5.	Transfer matrix without compression
6.	Transfer matrix with lossless compression	
  a.	Encode & Decode functions

For the MPI version, encoding of the sparse matrix is performed on the source node and then sent via the MPI_Send verb to the destination node.  The destination node receives the data using the MPI_Recv verb and then decodes the compressed matrices to the original matrix.  Timing accuracy for this section was a concern since the clocks vary between different nodes.  The current implementation of timing for MPI takes a time sample from the start of the encoding until the MPI_Send operation is complete on the source node.  Then the destination node starts time sample after the MPI_Recv command and ends after decoding is complete.
