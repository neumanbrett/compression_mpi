///////////////////////////////////////////////////////////////////////////////////////
// Compress
// Author: Brett Neuman (bneuman@ucar.edu)
// This code creates sparse matrices and tests compression methods for serial
///////////////////////////////////////////////////////////////////////////////////////
#include <stdlib.h>
#include <stdio.h>
#include <random>
#include <iostream>

using std::vector;

///////////////////////////////////////////////////////////////////////////////////////
// User configurable variables
///////////////////////////////////////////////////////////////////////////////////////
int         constexpr nx_global = 200;  // Grid size X dimension
int         constexpr ny_global = 200;  // Grid size Y dimension
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

    //init( &argc , &argv );
    init();

    vector<double>             v_state(nx_global*ny_global);
    vector<double>             v_compressed_state;
    vector<double> v_compressed_index;
    // Reserve space since we don't know exact size of the compressed vector
    v_compressed_state.reserve(ceil(nx_global*ny_global*bernoulli_p));
    v_compressed_index.reserve(ceil(nx_global*ny_global*bernoulli_p));

    generate_sparse(v_state, bernoulli_p, nx_global, ny_global);
    
    double sparsity = calculate_sparsity(v_state, nx_global, ny_global);
    std::cout << "Sparsity of Matrix: " << sparsity << "\n";   

    // Output values for debugging
    #ifdef DEBUG
        output_matrix(v_state, nx_global, ny_global);
        std::cout << "\n";
    #endif

    vector<double> v_uncompressed_state(nx_global*ny_global);
    generate_sparse(v_uncompressed_state, 0, nx_global, ny_global);

    lossless_encode(v_state, v_compressed_state, v_compressed_index, nx_global, ny_global);
    #ifdef DEBUG
        for (vector<double>::iterator iter = v_compressed_state.begin(); iter != v_compressed_state.end(); ++iter)
            std::cout << "\n" << *iter;
        std::cout << "\n\n";

        for (vector<double>::iterator iter2 = v_compressed_index.begin(); iter2 != v_compressed_index.end(); ++iter2)
            std::cout << "\n" << *iter2;
        std::cout << "\n";
    #endif
    lossless_decode(v_uncompressed_state, v_compressed_state, v_compressed_index, nx_global, ny_global);

    #ifdef DEBUG
        for (vector<double>::iterator iter2 = v_uncompressed_state.begin(); iter2 != v_uncompressed_state.end(); ++iter2)
            std::cout << "\n" << *iter2;
        std::cout << "\n";
    #endif

    #ifdef DEBUG
        for (vector<double>::iterator iter3 = v_compressed_index.begin(); iter3 != v_compressed_index.end(); ++iter3)
            std::cout << "\n" << *iter3;
        std::cout << "\n";
    #endif

    if (verify_decoding(v_state, v_uncompressed_state, nx_global, ny_global))
        std::cout << "\n" << "Original and Decoded Matrices are the same.\n\n";

    std::cout << "Size of uncompressed data: " << sizeof(double)*v_uncompressed_state.size() << " bits.\n";
    std::cout << "Size of compressed data: " << 2*sizeof(double)*v_compressed_state.size() << " bits.\n";

    cleanup();

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
                if (d(rng_b)==1) {
                    state[i*ny + j] = u(rng_u);
                    //#ifdef DEBUG
                    //    std::cout << "[" << i << ',' << j << "]: " << state[j*nx_global + i] << "\n";
                    //#endif
                }
                else
                    state[i*ny + j] = 0;
                    //#ifdef DEBUG
                    //    std::cout << " Bernoulli = 0\n";
                    //#endif
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