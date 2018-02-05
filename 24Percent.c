#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <xmmintrin.h>


const char* dgemm_desc = "Step 3 Yao Simple blocked dgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 64
#endif


#if !defined(BLOCK_SIZE_L1)
#define BLOCK_SIZE_L1 64
#endif

#if !defined(BLOCK_SIZE_L2)
#define BLOCK_SIZE_L2 128
#endif

#if !defined(REG_SIZE)
#define REG_SIZE 4
#endif

#define min(a,b) (((a)<(b))?(a):(b))

//utility functions
void col_major_pack(double* dest, double* src, int offset, int width, int height, int lda){
  int numBytes = height * sizeof(double);
  double* restrict pointer_src = src + offset;
  double* restrict pointer_dest = dest;
  __assume_aligned(pointer_src, 32);
  __assume_aligned(pointer_dest, 32);
  for (int j = 0; j < width; ++j){
    memmove(pointer_dest, pointer_src, numBytes);
    pointer_dest  +=  height;
    pointer_src   +=  lda;
  }
}
//utility functions
void row_major_pack(double* dest, double* src, int offset, int width, int height, int lda){
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



//a_row . b_col + c
void row_mul_col_plus_val(double* a, double* b, double* c, int length){
  __assume_aligned(a, 32);
  __assume_aligned(b, 32);
  double output = c[0];
  for (int i = 0; i < length; i++){
    output += a[i] * b[i];
  }
  c[0] = output;
}

//a_row . b_col + c
void row_mul_col_plus_val_four(double* a0, double* a1, double* a2, double* a3, double* b, double* c0, double* c1, double* c2, double* c3, int length){
  __assume_aligned(a0, 32);
  __assume_aligned(b, 32);
  __assume_aligned(a1, 32);
  __assume_aligned(a2, 32);
  __assume_aligned(a3, 32);
  __assume_aligned(c0, 32);
  __assume_aligned(c1, 32);
  __assume_aligned(c2, 32);
  __assume_aligned(c3, 32);

  double output0 = c0[0];
  double output1 = c1[0];
  double output2 = c2[0];
  double output3 = c3[0];

  for (int i = 0; i < length; ++i){
    output0 += a0[i] * b[i];
    output1 += a1[i] * b[i];
    output2 += a2[i] * b[i];
    output3 += a3[i] * b[i];
  }
  c0[0] = output0;
  c1[0] = output1;
  c2[0] = output2;
  c3[0] = output3;
}

/*
A = M * K
B = K * N
C = M * N
*/
//Row major A * Column major B
void matrix_mult(int lda, double* restrict A, double* restrict B, double* restrict C, int M, int N, int K){
  __assume_aligned(A, 32);
  __assume_aligned(B, 32);

  for (int j = 0; j < N; j ++){
    double* B_pos __attribute__((aligned(32))) = B+j*K;
    double* C_pos __attribute__((aligned(32))) = C+j*lda;

    for (int i = 0; i <= M - 4; i += 4)
      row_mul_col_plus_val_four(A+i*K, A+(i+1)*K, A+(i+2)*K, A+(i+3)*K, B_pos, C_pos++, C_pos++, C_pos++, C_pos++, K);
    
    for (int i = (M/4)*4; i < M; ++i)
      row_mul_col_plus_val(A+i*K, B_pos, C_pos++, K);
    
  }
}

/*
A 
-->   
--> 
*/
void L1_divide(int lda, double* restrict A, double* restrict B, double* restrict C, int M, int N, int K){
  __assume_aligned(A, 32);
  __assume_aligned(B, 32);
  __assume_aligned(C, 32);
  for (int i = 0; i < M; i += BLOCK_SIZE_L1){ //for each rows A
    
    int M_ = min(BLOCK_SIZE_L1, M-i);
    
    for (int k = 0; k < K; k += BLOCK_SIZE_L1){ //sub cols of rows A, sub rows of cols B
      
      int K_ = min(BLOCK_SIZE_L1, K-k);

      //pack L1A  M_ * K_ 
      double A_packed_L1[M_ * K_] __attribute__((aligned(32))); 
      col_major_pack(A_packed_L1, A, i*K+k, M_, K_, K);

      for (int j = 0; j < N; j+= BLOCK_SIZE_L1){ //for each cols of B
        
        int N_ = min(BLOCK_SIZE_L1, N-j);

        //pack L1B  K_ * N_
        double B_packed_L1[K_ * N_] __attribute__((aligned(32))); 
        col_major_pack(B_packed_L1, B, j*K+k, N_, K_, K);

        matrix_mult(lda, A_packed_L1, B_packed_L1, C+i+j*lda, M_, N_, K_);
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

  for (int j = 0; j < lda; j += BLOCK_SIZE_L2){ //for each cols of B
    int N_ = min(BLOCK_SIZE_L2, lda-j);
    
    for (int k = 0; k < lda; k += BLOCK_SIZE_L2){ //sub cols of rows A, sub rows of cols B
      int K_ = min(BLOCK_SIZE_L2, lda-k);
      //pack L1B  K_ * N_
      double B_packed_L2[K_ * N_] __attribute__((aligned(32))); 
      col_major_pack(B_packed_L2, B, k+j*lda, N_, K_, lda);
      
      for (int i = 0; i < lda; i += BLOCK_SIZE_L2){ //for each rows A
        int M_ = min(BLOCK_SIZE_L2, lda-i);
        //pack L2 M_ * K_
        double A_packed_L2[M_ * K_] __attribute__((aligned(32)));
        row_major_pack(A_packed_L2, A, i+k*lda, K_, M_, lda);

       L1_divide(lda, A_packed_L2, B_packed_L2, C+i+j*lda, M_, N_, K_);
      }
    }
  }
}


void square_dgemm(int lda, double* A, double* B, double* C){
  L2_divide(lda, A, B, C);
}
