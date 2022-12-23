///////////////////////////////////////////////////////////////////////////////////////
// Compress
// Author: Brett Neuman (bneuman@ucar.edu)
// This code creates sparse matrices and tests compression methods for sending
// data across ranks
///////////////////////////////////////////////////////////////////////////////////////
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <random>
#include <iostream>

using std::vector;

///////////////////////////////////////////////////////////////////////////////////////
// User configurable variables
///////////////////////////////////////////////////////////////////////////////////////
int         constexpr nx_global = 1000;  // Grid size X dimension
int         constexpr ny_global = 1000;  // Grid size Y dimension
double      constexpr bernoulli_p = 0.15;   // Probability of generating a 1 for sparse matrix generator
                                           // Ex. 0.2 should generate a matrix with 20 percent non-zero elements


///////////////////////////////////////////////////////////////////////////////////////
// Declaring functions after main
///////////////////////////////////////////////////////////////////////////////////////
//void    init                (int *argc, char ***argv); 
void    init                ();    
void    generate_sparse     (vector<double> &state, double p, int nx, int ny);
void    generate_random     (vector<double> &state);
void    output_matrix       (vector<double> &state, int nx, int ny);
double  calculate_sparsity  (vector<double> &state, int nx, int ny);
void    xfer                (vector<double> &state);
void    xfer_lossless       (vector<double> &state);
void    lossless_encode     (vector<double> &state, vector<double> &compressed_state, vector<double> &compressed_index, int nx, int ny);
void    lossless_decode     (vector<double> &state, vector<double> &compressed_state, vector<double> &compressed_index, int nx, int ny);
void    xfer_lossy          (vector<double> &state);
void    lossy_encode        (vector<double> &state, vector<double> &compressed_state, vector<double> &compressed_index, int nx, int ny);
void    lossy_decode        (vector<double> &state, vector<double> &compressed_state, vector<double> &compressed_index, int nx, int ny);
void    compare_c           (vector<double> &state, vector<double> &compressed_state);
bool    verify_decoding     (vector<double> &state, vector<double> &uncompressed_state, int nx, int ny);
void    cleanup             ();

///////////////////////////////////////////////////////////////////////////////////////
// Start of program
///////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {

    // Initialization Steps
    init();
    int rank, nprocs;
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);
    MPI_Status status;

    double start_regular, end_regular, start_compressed_rank0, end_compressed_rank0, start_compressed_rank1, end_compressed_rank1 = 0;
    double start_compressed, end_compressed = 0;
    double regular_elapsed, compressed_elapsed = 0;

    if (rank == 0) {
        vector<double>      v_state_send(nx_global*ny_global);
        // MPI Vectors to send encoded matrix
        vector<double>      v_compressed_state_send;
        vector<double>      v_compressed_index_send;
        // Reserve space since we don't know exact size of the compressed vector
        v_compressed_state_send.reserve(ceil(nx_global*ny_global*bernoulli_p));
        v_compressed_index_send.reserve(ceil(nx_global*ny_global*bernoulli_p));

        // Sparsity generation for original matrix
        generate_sparse(v_state_send, bernoulli_p, nx_global, ny_global);
        double sparsity = calculate_sparsity(v_state_send, nx_global, ny_global);
        std::cout << "Sparsity of Matrix: " << sparsity << "\n";   

        // Output values for debugging
        #ifdef DEBUG
            output_matrix(v_state_send, nx_global, ny_global);
            std::cout << "\n";
        #endif

        // Start Timer for Uncompressed MPI Transfer
        //MPI_Barrier(comm);
        start_regular = MPI_Wtime();
        MPI_Send(&v_state_send[0], nx_global*ny_global, MPI_DOUBLE, 1, 0, comm);
        //MPI_Barrier(comm);
        end_regular = MPI_Wtime();

        // Start Timer for Compressed MPI Transfer
        // Encode
        start_compressed_rank0 = MPI_Wtime();
        //start_compressed = MPI_Wtime();
        lossless_encode(v_state_send, v_compressed_state_send, v_compressed_index_send, nx_global, ny_global);

        std::cout << "\nState size after encode: " << v_compressed_state_send.size();
        std::cout << "\nIndex size after encode: " << v_compressed_index_send.size();

        #ifdef DEBUG
            for (vector<double>::iterator iter2 = v_compressed_state_send.begin(); iter2 != v_compressed_state_send.end(); ++iter2)
                std::cout << "\nCompressed value before MPI Send: " << *iter2;
            std::cout << "\n";
        #endif

        #ifdef DEBUG
            for (vector<double>::iterator iter3 = v_compressed_index_send.begin(); iter3 != v_compressed_index_send.end(); ++iter3)
                std::cout << "\nCompressed index before MPI Send: " << *iter3;
            std::cout << "\n";
        #endif
        int matrix_size = v_compressed_state_send.size();
        MPI_Send(&matrix_size, 1, MPI_INT, 1, 1, comm);
        MPI_Send(&v_compressed_state_send[0], v_compressed_state_send.size(), MPI_DOUBLE, 1, 2, comm);
        MPI_Send(&v_compressed_index_send[0], v_compressed_index_send.size(), MPI_DOUBLE, 1, 2, comm);
        end_compressed_rank0 = MPI_Wtime();
    }

    if (rank == 1) {
        // MPI Vectors to receive encoded matrix
        vector<double>      v_compressed_state_recv(1);
        vector<double>      v_compressed_index_recv(1);
        // Reserve space since we don't know exact size of the compressed vector
        //v_compressed_state_recv.reserve(ceil(nx_global*ny_global*bernoulli_p));
        //v_compressed_index_recv.reserve(ceil(nx_global*ny_global*bernoulli_p));
        vector<double> v_state_recv(nx_global*ny_global);
        vector<double> v_uncompressed_state_recv(nx_global*ny_global); 

        // All zero matrix for receive uncompressed matrix
        generate_sparse(v_state_recv, 0, nx_global, ny_global);
        generate_sparse(v_uncompressed_state_recv, 0, nx_global, ny_global);

        // Receive MPI data for regular transfer
        MPI_Recv(&v_state_recv[0], nx_global*ny_global, MPI_DOUBLE, 0, 0, comm, &status);
        // End Timer

        int matrix_size_recv = 0;
        // Receive MPI data for compressed transfer
        MPI_Recv(&matrix_size_recv, 1, MPI_INT, 0, 1, comm, &status);
        v_compressed_state_recv.resize(matrix_size_recv);
        v_compressed_index_recv.resize(matrix_size_recv);

        MPI_Recv(&v_compressed_state_recv[0], matrix_size_recv, MPI_DOUBLE, 0, 2, comm, &status);
        MPI_Recv(&v_compressed_index_recv[0], matrix_size_recv, MPI_DOUBLE, 0, 2, comm, &status);
        start_compressed_rank1 = MPI_Wtime();

        #ifdef DEBUG
            for (vector<double>::iterator iter = v_compressed_state_recv.begin(); iter != v_compressed_state_recv.end(); ++iter)
                std::cout << "\nCompressed value after MPI Send: " << *iter;
            std::cout << "\n\n";

            for (vector<double>::iterator iter2 = v_compressed_index_recv.begin(); iter2 != v_compressed_index_recv.end(); ++iter2)
                std::cout << "\nCompressed index after MPI Send: " << *iter2;
            std::cout << "\n";
        #endif

        lossless_decode(v_uncompressed_state_recv, v_compressed_state_recv, v_compressed_index_recv, nx_global, ny_global);
        // End Timer
        end_compressed_rank1 = MPI_Wtime();
        //end_compressed = MPI_Wtime();

        if (verify_decoding(v_state_recv, v_uncompressed_state_recv, nx_global, ny_global))
            std::cout << "\n" << "Original and Decoded Matrices are the same.\n\n";

        compressed_elapsed = (end_compressed_rank0 - start_compressed_rank0) + (end_compressed_rank1 - start_compressed_rank1);
        std::cout << "Size of uncompressed data: " << sizeof(double)*v_state_recv.size() << " bits.\n";
        std::cout << "Size of compressed data: " << 2*sizeof(double)*v_compressed_state_recv.size() << " bits.\n";
        std::cout << "\nElapsed time for compressed send with encode and decode: " << compressed_elapsed << " seconds\n";
    }

    if (rank == 0) {
        //end_compressed_rank0 = MPI_Wtime();
        //MPI_Barrier(comm);
        regular_elapsed = end_regular - start_regular;
        //compressed_elapsed = (end_compressed_rank0 - start_compressed_rank0) + (end_compressed_rank1 - start_compressed_rank1);
        //compressed_elapsed = end_compressed - start_compressed;
        std::cout << "\nElapsed time for uncompressed send: " << regular_elapsed << " seconds\n";
        
    }

    cleanup();
    MPI_Finalize();

    return 0;
}

///////////////////////////////////////////////////////////////////////////////////////
// Functions
///////////////////////////////////////////////////////////////////////////////////////

/* Generates a sparse matrix with sparsity value between 0 and 1.
/ The sparsity parameter should be 2 sigfigs (ex. 0.71).
/ 4x4 Matrix and Loop Access
    
*   | * * * * | i = rows
*   | * * * * |
*   | * * * * |
*   | * * * * |
    j = columns

    |  * * * *  |  * * * *  |  * * * *  |  * * * *  |
    
    | j=0 i=1:4 | j=1 i=1:4 | j=2 i=1:4 | j=3 i=1:4 |
    | j * 4 + i | j * 4 + i | j * 4 + i | j * 4 + i |
*/
void generate_sparse(vector<double> &state, double p, int nx, int ny) {
    // Valid probability value check to generate our matrix
    if (p > 0 && p < 1) {
        
        // Create Bernoulli Distribution to determine non-zero elements
        //srand(time(0));
        std::bernoulli_distribution d(p);
        std::random_device rd_b{};
        std::mt19937 rng_b{rd_b()};

        // Create Uniform Distribution to determine values for non-zero elements
        std::uniform_real_distribution<double> u(1,11);
        std::random_device rd_u{};
        std::mt19937 rng_u{rd_u()};

        // Loop through each index to determine if it should be non-zero
        // If non-zero, assign uniform random value to that index
        for(int i = 0; i < nx; i++) {
            for(int j = 0; j < ny; j++) {
                if (d(rng_b)==1) 
                    state[i*ny + j] = u(rng_u);
                else
                    state[i*ny + j] = 0;
            }
        }
    }
}

// Sparsity calculator
double calculate_sparsity(vector<double> &state, int nx, int ny) {
    int nonzero = 0;
    double elements = nx*ny;

    for(int i = 0; i < nx; i++) {
        for(int j = 0; j < ny; j++) {
            if (state[i*ny + j] != 0) 
                nonzero++;
        }
    }

    return nonzero / elements;
}

// Standard output of matrix values
void output_matrix(vector<double> &state, int nx, int ny) {
    for(int i = 0; i < nx; i++) {
        for(int j = 0; j < ny; j++) {
            if (!(j % ny))
                std::cout << "\n";
            std::cout << "[" << i << ',' << j << "]: " << state[i*ny + j] << " ";          
        }
    }
}

// Transfer without compression
void xfer(double* state) {}

// Transfer with lossless compression
void xfer_lossless(double* state) {}

// Implementation of Compressed Row Storage (CRS)
// Takes in a sparse matrix and uses CRS
void lossless_encode(vector<double> &state, vector<double> &compressed_state, vector<double> &compressed_index, int nx, int ny) {
    int inc, index = 0;
    for(int i = 0; i < nx; i++) {
        for(int j = 0; j < ny; j++) {
            if (state[i*ny + j] > 0) {
                // Add the value to the compressed matrix
                compressed_state.push_back(state[i*ny + j]);
                // Add the modulus ny index of the non-zero value
                compressed_index.push_back(i*ny + j);
                inc++;
            }   
        }
    }
}

// Takes in encoded CSR sparse matrix and returns it to original state
void lossless_decode(vector<double> &uncompressed_state, vector<double> &compressed_state, vector<double> &compressed_index, int nx, int ny) {  
    int i = 0;
    for (vector<double>::iterator iter = compressed_index.begin(); iter != compressed_index.end(); ++iter) {
        uncompressed_state[*iter] = compressed_state.at(i);
        i++;
    }   
}

// Transfer with lossy compression
void xfer_lossy(double* state) {}
void lossy_encode(double input) {}
void lossy_decode(double output) {}

// Verify the decoded 
bool verify_decoding(vector<double> &state, vector<double> &uncompressed_state, int nx, int ny) {
    for(int i = 0; i < nx; i++) {
        for(int j = 0; j < ny; j++) {
            if (state[i*ny + j] != uncompressed_state[i*ny + j])
                return false;
        }
    }
    return true;
}

// Initializes variables, grid space, and MPI domain
void init() {
    srand(time(0));
}

void cleanup() {}