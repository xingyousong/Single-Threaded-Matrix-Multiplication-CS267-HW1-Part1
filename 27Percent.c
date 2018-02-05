#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <xmmintrin.h>
#include <immintrin.h>
#include <math.h>

const char* dgemm_desc = "Step 3 Yao Simple blocked dgemm.";

#if !defined(BLOCK_SIZE_L2)
#define BLOCK_SIZE_L2 128
#endif

#if !defined(BLOCK_SIZE_L1)
#define BLOCK_SIZE_L1 64
#endif

#define min(a,b) (((a)<(b))?(a):(b))
#define pad(a) ((a % 4) == (0)? (a):(a + 4 - a % 4))

static inline void prefetch(double* restrict dest, double* restrict src, int dest_offset, int src_offset, int width, int height, int padded_width, int padded_height, int src_stride){
  __assume(width > 0);
  __assume(height > 0);
  __assume(padded_width > 0);
  __assume(padded_height > 0);
  __assume_aligned(dest, 32);
  __assume_aligned(src, 32);

  int dataBytesNum = height * sizeof(double);
  int zeroBytesNum = (padded_height - height) * sizeof(double);

  for (int w = 0; w < width; ++w){
    memcpy(dest + dest_offset + w * padded_height, src + src_offset + w * src_stride, dataBytesNum);
    //pad 0 
    memset(dest + dest_offset + w * padded_height + height, 0, zeroBytesNum); 
  }
  for (int w = width; w < padded_width; ++w){
    memset(dest + dest_offset + w * padded_height, 0, padded_height * sizeof(double));
  }
}

static inline void store(double* restrict dest, double* restrict src, int dest_offset, int src_offset, int width, int height, int dest_stride, int src_stride){
  __assume(width > 0);
  __assume(height > 0);
  __assume(dest_stride > 0);
  __assume(src_stride > 0);
  __assume_aligned(dest, 32);
  __assume_aligned(src, 32);

  int dataBytesNum = height * sizeof(double);

  for (int w = 0; w < width; ++w){
    memcpy(dest + dest_offset + w * dest_stride, src + src_offset + w * src_stride, dataBytesNum);
  }
}

static inline void transpose(double* restrict dest, double* restrict src, int width, int height, int src_stride){
  __assume_aligned(dest, 32);
  __assume_aligned(src, 32);
  int pos = 0;
  for (int j = 0; j < height; j++){
    for (int i = 0; i < width; i++){
      dest[pos++] = src[j + i * src_stride];
    }
  }
}

/*
A = M * K
B = K * N
C = M * N
*/
static inline void four_by_four(double* restrict A, double* restrict B, double* restrict C, int M, int N, int K){
  __assume_aligned(A, 32);
  __assume_aligned(B, 32);
  __assume_aligned(C, 32);

  __m256d c_col_0, c_col_1, c_col_2, c_col_3;
  __m256d b00, b01, b02, b03;
  c_col_0 = _mm256_load_pd(C);
  c_col_1 = _mm256_load_pd(C+M);
  c_col_2 = _mm256_load_pd(C+2*M);
  c_col_3 = _mm256_load_pd(C+3*M);

  for (int i = 0; i < 4; ++i){
    __m256d a_col = _mm256_load_pd(A+M*i);

    b00 = _mm256_set1_pd(B[i]);
    b01 = _mm256_set1_pd(B[K+i]);
    b02 = _mm256_set1_pd(B[2*K+i]);
    b03 = _mm256_set1_pd(B[3*K+i]);

    c_col_0 = _mm256_add_pd(_mm256_mul_pd(a_col, b00), c_col_0);
    c_col_1 = _mm256_add_pd(_mm256_mul_pd(a_col, b01), c_col_1);
    c_col_2 = _mm256_add_pd(_mm256_mul_pd(a_col, b02), c_col_2);
    c_col_3 = _mm256_add_pd(_mm256_mul_pd(a_col, b03), c_col_3);
  }
  _mm256_store_pd(C, c_col_0);
  _mm256_store_pd(C+M, c_col_1);
  _mm256_store_pd(C+2*M, c_col_2);
  _mm256_store_pd(C+3*M, c_col_3);
}



static inline void compute(double* restrict A, double* restrict B, double* restrict C, int M, int N, int K, int padded_M, int padded_N, int padded_K){
  for (int i = 0; i < M; i += 4){
    for (int j = 0; j < N; j += 4){
      for (int k = 0; k < K; k += 4){
        four_by_four(A+i+k*padded_M, B+k+j*padded_K, C+i+j*padded_M, padded_M, padded_N, padded_K);
      }
    }
  }
}

/*
A = M * K
B = K * N
C = M * N
*/
void do_block_l1 (double* restrict A, double* restrict B, double * restrict C, int M, int N, int K, int lda){
    for (int j = 0; j < N; j += BLOCK_SIZE_L1) {
        int sub_N = min (BLOCK_SIZE_L1, N - j);

        for (int i = 0; i < M; i += BLOCK_SIZE_L1){
          int sub_M = min (BLOCK_SIZE_L1, M - i);

          /*---Allocate L1 C Start---*/
          double L1_C[BLOCK_SIZE_L1 * BLOCK_SIZE_L1] __attribute__((aligned(32)));
          prefetch(L1_C, C, 0, i + j * lda, sub_N, sub_M, BLOCK_SIZE_L1, BLOCK_SIZE_L1, lda);
          /*---Allocate L1 C End---*/

          for (int k = 0; k < K; k += BLOCK_SIZE_L1){
              int sub_K = min (BLOCK_SIZE_L1, K - k);
              
              /*---Allocate L1 B Start---*/
              double L1_B[BLOCK_SIZE_L1 * BLOCK_SIZE_L1] __attribute__((aligned(32)));
              prefetch(L1_B, B, 0, k + j * BLOCK_SIZE_L2, sub_N, sub_K, BLOCK_SIZE_L1, BLOCK_SIZE_L1, BLOCK_SIZE_L2);
              /*---Allocate L1 B End---*/
          
              /*---Allocate L1 A Start---*/
              double L1_A[BLOCK_SIZE_L1 * BLOCK_SIZE_L1] __attribute__((aligned(32)));
              prefetch(L1_A, A, 0, i + k * BLOCK_SIZE_L2, sub_K, sub_M, BLOCK_SIZE_L1, BLOCK_SIZE_L1, BLOCK_SIZE_L2);
              /*---Allocate L1 A End---*/

              
              /*---Computing the L1 blocks Start---*/
              compute(L1_A, L1_B, L1_C, sub_M, sub_N, sub_K, BLOCK_SIZE_L1, BLOCK_SIZE_L1, BLOCK_SIZE_L1);
              //compute_2(L1_A, L1_B, L1_C, sub_M, sub_N, sub_K);
              /*---Computing the L1 blocks End---*/
          }

          /*---Save the result back to C Start---*/
          store(C, L1_C, i + j * lda, 0, sub_N, sub_M, lda, BLOCK_SIZE_L1);
          /*---Save the result back to C End---*/ 
        }
    }
}

/*
A = M * K
B = K * N
C = M * N
*/
void do_block_l2 (int lda, double* A, double* B, double* C){
    // double A_T[lda * lda] __attribute__((aligned(32)));
    // transpose(A_T, A, lda, lda, lda);

    for (int j = 0; j < lda; j += BLOCK_SIZE_L2){
      int N = min (BLOCK_SIZE_L2, lda-j);

      for (int k = 0; k < lda; k += BLOCK_SIZE_L2){
        int K = min (BLOCK_SIZE_L2, lda-k);

        /*---Allocate L2 B Start---*/
        double L2_B[BLOCK_SIZE_L2 * BLOCK_SIZE_L2] __attribute__((aligned(32)));
        prefetch(L2_B, B, 0, k + j * lda, N, K, BLOCK_SIZE_L2, BLOCK_SIZE_L2, lda);
        /*---Allocate L2 B End---*/

        for (int i = 0; i < lda; i += BLOCK_SIZE_L2){
          int M = min (BLOCK_SIZE_L2, lda-i);

          /*---Allocate L2 A Start---*/
          double L2_A[BLOCK_SIZE_L2 * BLOCK_SIZE_L2] __attribute__((aligned(32)));
          prefetch(L2_A, A, 0, i + k * lda, K, M, BLOCK_SIZE_L2, BLOCK_SIZE_L2, lda);
          /*---Allocate L2 A End---*/
          
          /*---Do block L1 start----*/
          do_block_l1(L2_A, L2_B, C + i + j * lda, M, N, K, lda);
          /*---Do block L1 end----*/
          
          //four_by_four_SSE(lda, M, N, K, A + i + k*lda, B + k + j*lda, C + i + j*lda);
          //naive(lda, M, N, K, A + i + k*lda, B + k + j*lda, C + i + j*lda);
      }
    }
  }
}

void square_dgemm (int lda, double *A, double *B, double *C){
  do_block_l2(lda, A, B, C);
}


// static inline void gemm4x4_SSE(double *B, double *A, double *C) {
//     __assume_aligned(A, 32);
//     __assume_aligned(B, 32);
//     __assume_aligned(C, 32);
//     __m256d row[4], sum[4];

//     for(int i=0; i<4; ++i)  
//       row[i] = _mm256_load_pd(&B[i*4]);

//     for(int i=0; i<4; i++) {
//         sum[i] = _mm256_load_pd(C+4*i);      
//         for(int j=0; j<4; j++) {
//             sum[i] = _mm256_add_pd(_mm256_mul_pd(_mm256_set1_pd(A[i*4+j]), row[j]), sum[i]);
//         }        
//     }
//     for(int i=0; i<4; i++) 
//       _mm256_store_pd(&C[i*4], sum[i]); 
// }

// static inline void four_by_four_naive (int lda, int M, int N, int K, double* A, double* B, double* C){
//   /* For each column j of B */ 
//   for (int j = 0; j < N; ++j) {
//     /* For each row i of A */
//     for (int i = 0; i < M; ++i){
//       /* Compute C(i,j) */
//       double cij = C[i+j*lda];
//       for (int k = 0; k < K; ++k){
//         cij += A[i+k*lda] * B[k+j*lda];
//       }
//       C[i+j*lda] = cij;
//     }
//   }
// }