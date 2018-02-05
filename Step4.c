#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <xmmintrin.h>


const char* dgemm_desc = "Step 4 Yao Simple blocked dgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 64
#endif


#if !defined(BLOCK_SIZE_L1)
#define BLOCK_SIZE_L1 48
#endif

#if !defined(BLOCK_SIZE_L2)
#define BLOCK_SIZE_L2 144
#endif

#if !defined(REG_SIZE)
#define REG_SIZE 4
#endif

#define min(a,b) (((a)<(b))?(a):(b))

//utility functions
void col_major_pack(double* dest, double* src, int offset, int width, int height, int lda){
  __assume_aligned(dest, 32);
  __assume_aligned(src, 32);
  int numBytes = height * sizeof(double);
  double* restrict pointer_src = src + offset;
  double* restrict pointer_dest = dest;
  __assume_aligned(pointer_src, 32);
  __assume_aligned(pointer_dest, 32);
  for (int j = 0; j < width; ++j){
    memcpy(pointer_dest, pointer_src, numBytes);
    pointer_dest  +=  height;
    pointer_src   +=  lda;
  }
}
//utility functions
void row_major_pack(double* dest, double* src, int offset, int width, int height, int lda){
  __assume_aligned(dest, 32);
  __assume_aligned(src, 32);
  double* restrict pointer_dest = dest;
  double* restrict pointer_src = src + offset;
  __assume_aligned(pointer_dest, 32);
  __assume_aligned(pointer_src, 32);
  int pos = 0;
  for (int j = 0; j < height; ++j){
    for (int i = 0; i < width; ++i){
      pointer_dest[pos++] = pointer_src[j + i * lda];
    }
  }
}

//utility functions
void padded_pack(double* dest, double* src, int offset, int width, int height, int dest_stride, int src_stride){
  __assume_aligned(dest, 32);
  __assume_aligned(src, 32);
  int numBytes = height * sizeof(double);
  double* restrict pointer_src = src + offset;
  double* restrict pointer_dest = dest;
  __assume_aligned(pointer_src, 32);
  __assume_aligned(pointer_dest, 32);
  for (int j = 0; j < width; ++j){
    memcpy(pointer_dest, pointer_src, numBytes);
    memset(pointer_dest + height, 0, (dest_stride - height)*sizeof(double));
    pointer_dest  +=  dest_stride;
    pointer_src   +=  src_stride;
  }
}

void put_back(double* dest, double* src, int offset, int width, int height, int dest_stride, int src_stride){
  __assume_aligned(dest, 32);
  __assume_aligned(src, 32);
  int numBytes = height * sizeof(double);
  double* restrict pointer_src = src;
  double* restrict pointer_dest = dest + offset;
  __assume_aligned(pointer_src, 32);
  __assume_aligned(pointer_dest, 32);
  for (int j = 0; j < width; ++j){
    memcpy(pointer_dest, pointer_src, numBytes);
    pointer_dest  +=  dest_stride;
    pointer_src   +=  src_stride;
  }
}


//a_row . b_col + c
void row_mul_col_plus_val(double* a, double* b, double* c, int length){
  __assume_aligned(a, 32);
  __assume_aligned(b, 32);
  //__assume(length % 4 == 0);
  double output = c[0];
  for (int i = 0; i < length; i++){
    output += a[i] * b[i];
  }
  c[0] = output;
}

void SSE_FOUR(double* restrict A, double* restrict B, double* restrict C, int m_c, int n_r, int k_c){
  __assume_aligned(A, 32);
  __assume_aligned(B, 32);
  __assume_aligned(C, 32);

  //load B
  //first column
  register double b00 = B[0];
  register double b10 = B[1];
  register double b20 = B[2];
  register double b30 = B[3];
  //second column
  register double b01 = B[0 + k_c];
  register double b11 = B[1 + k_c];
  register double b21 = B[2 + k_c];
  register double b31 = B[3 + k_c];
  //third column
  register double b02 = B[0 + 2 * k_c];
  register double b12 = B[1 + 2 * k_c];
  register double b22 = B[2 + 2 * k_c];
  register double b32 = B[3 + 2 * k_c];
  //fourth column
  register double b03 = B[0 + 3 * k_c];
  register double b13 = B[1 + 3 * k_c];
  register double b23 = B[2 + 3 * k_c];
  register double b33 = B[3 + 3 * k_c];

  //minimize the amount of memory read as much as possible
  for (int i = 0; i < 4; ++i)
  {
    //load the row of A
    register double ai0 = A[i*k_c];
    register double ai1 = A[i*k_c + 1];
    register double ai2 = A[i*k_c + 2];
    register double ai3 = A[i*k_c + 3];

    //calculate the row of C
    C[i]            += ai0 * b00 + ai1 * b10 + ai2 * b20 + ai3 * b30;
    C[i + m_c]      += ai0 * b01 + ai1 * b11 + ai2 * b21 + ai3 * b31;
    C[i + 2 * m_c]  += ai0 * b02 + ai1 * b12 + ai2 * b22 + ai3 * b32;
    C[i + 3 * m_c]  += ai0 * b03 + ai1 * b13 + ai2 * b23 + ai3 * b33;
  }
}

/*
A = M * K
B = K * N
C = M * N
*/
//Row major A * Column major B
void matrix_mult(double* restrict A, double* restrict B, double* restrict C, int M, int N, int K, int padded_M, int padded_N, int padded_K){
  __assume_aligned(A, 32);
  __assume_aligned(B, 32);
  __assume_aligned(C, 32);
  __assume(padded_M % 4 == 0);
  __assume(padded_N % 4 == 0);
  __assume(padded_K % 4 == 0);
  //for each four rows of A
  for (int i = 0; i < padded_M; i += 4){
    //for each four cols of B
    for (int j = 0; j < padded_N; j += 4){
        //for each sub rows of B
        for (int k = 0; k < padded_K; k += 4){
          SSE_FOUR(A+i*padded_K+k, B+j*padded_K+k,  C+i+j*padded_M, padded_M, padded_N, padded_K);
          //row_mul_col_plus_val(A+i*padded_K, B+j*padded_K,  C+i+j*padded_M, padded_K);
      }
    }
  }
}

/*
A = M * K
B = K * N
C = M * N
*/
void L1_divide(int lda, double* restrict A, double* restrict B, double* restrict C, int M, int N, int K){
  __assume_aligned(A, 32);
  __assume_aligned(B, 32);
  __assume_aligned(C, 32);
  for (int i = 0; i < M; i += BLOCK_SIZE_L1){ //for each rows A
    
    int real_M = min(BLOCK_SIZE_L1, M-i);
    int padded_M = ceil(real_M / (double) REG_SIZE) * REG_SIZE;
    
    //sub cols of rows A, sub rows of cols B
     for (int k = 0; k < K; k += BLOCK_SIZE_L1){
      int real_K = min(BLOCK_SIZE_L1, K-k);
      int padded_K = ceil(real_K / (double) REG_SIZE) * REG_SIZE;

      //pack L1A  M_ * K_ 
      double A_padded[padded_M * padded_K] __attribute__((aligned(32))); 
      padded_pack(A_padded, A, i*K+k, real_M, real_K, padded_K, K);

      for (int j = 0; j < N; j+= BLOCK_SIZE_L1){ //for each cols of B
        
        int real_N = min(BLOCK_SIZE_L1, N-j);
        int padded_N = ceil(real_N / (double) REG_SIZE) * REG_SIZE;

        //pack L1B  K_ * N_
        double B_padded[padded_K * padded_N] __attribute__((aligned(32))); 
        padded_pack(B_padded, B, j*K+k, real_N, real_K, padded_K, K);

        //pack L1C  M_ * N_
        double C_padded[padded_M * padded_N] __attribute__((aligned(32))); 
        padded_pack(C_padded, C, j*lda+i, real_N, real_M, padded_M, lda);


        matrix_mult(A_padded, B_padded, C_padded, real_M, real_N, real_K, padded_M, padded_N, padded_K);

        put_back(C, C_padded, j*lda + i, real_N, real_M, lda, padded_M);
      }
    }
  }
}

/*
A = M * K
B = K * N
C = M * N
*/
void L2_divide(int lda, double* restrict A, double* restrict B, double* restrict C){
  __assume_aligned(A, 32);
  __assume_aligned(B, 32);
  __assume_aligned(C, 32);

  
  for (int i = 0; i < lda; i += BLOCK_SIZE_L2){ //for each rows A
    int M_ = min(BLOCK_SIZE_L2, lda-i);
    
    for (int k = 0; k < lda; k += BLOCK_SIZE_L2){ //sub cols of rows A, sub rows of cols B
      int K_ = min(BLOCK_SIZE_L2, lda-k);

      //pack L2 M_ * K_
      double A_packed_L2[M_ * K_] __attribute__((aligned(32)));
      row_major_pack(A_packed_L2, A, i+k*lda, K_, M_, lda);

      for (int j = 0; j < lda; j += BLOCK_SIZE_L2){ //for each cols of B
        int N_ = min(BLOCK_SIZE_L2, lda-j);
        //pack L2B  K_ * N_
        double B_packed_L2[K_ * N_] __attribute__((aligned(32))); 
        col_major_pack(B_packed_L2, B, k+j*lda, N_, K_, lda);
      
    
        L1_divide(lda, A_packed_L2, B_packed_L2, C+i+j*lda, M_, N_, K_);
      }
    }
  }
}


void square_dgemm(int lda, double* A, double* B, double* C){
  L2_divide(lda, A, B, C);
}
