#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <xmmintrin.h>


const char* dgemm_desc = "Step 2 Yao Simple blocked dgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 64
#endif


#if !defined(BLOCK_SIZE_L1)
#define BLOCK_SIZE_L1 64
#endif

#define min(a,b) (((a)<(b))?(a):(b))

//utility functions
void col_major_pack(double* dest, double* src, int offset, int width, int height, int lda){
  int numBytes = height * sizeof(double);
  double* pointer_src = src + offset;
  double* pointer_dest = dest;
  for (int j = 0; j < width; j++){
    memmove(pointer_dest, pointer_src, numBytes);
    pointer_dest  +=  height;
    pointer_src   +=  lda;
  }
}
//utility functions
void row_major_pack(double* dest, double* src, int offset, int width, int height, int lda){
  double* restrict pointer_dest = dest;
  double* restrict pointer_src = src + offset;
  int pos = 0;
  for (int  j = 0; j < height; ++j){
    for (int i = 0; i < width; ++i){
      pointer_dest[pos++] = pointer_src[j + i * lda];
    }
  }
}


//a_row . b_col + c
void row_mul_col_plus_val(double* a, double* b, double* c, int length){
  double output = c[0];
  for (int i = 0; i < length; i++){
    output += a[i] * b[i];
  }
  c[0] = output;
}

/*
A = M * K
B = K * N
C = M * N
*/
//Row major A * Column major B
void matrix_mult(int lda, double* A, double* B, double* C, int M, int N, int K){
  //for each row of A
  for (int i = 0; i < M; i++){
    //for each row of B
    for (int j = 0; j < N; j++){
      //computer row vec * col vec + val
      //a_row . b_col + c
      row_mul_col_plus_val(A+i*K, B+j*K, C+i+j*lda, K);
    }
  }
}

/*
A 
-->   
--> 
*/
void L1_divide(int lda, double* A, double* B, double* C, int M, int N, int K){
  for (int i = 0; i < M; i += BLOCK_SIZE_L1){ //for each rows A
    
    int M_ = min(BLOCK_SIZE_L1, M-i);
    
    for (int k = 0; k < K; k += BLOCK_SIZE_L1){ //sub cols of rows A, sub rows of cols B
      
      int K_ = min(BLOCK_SIZE_L1, K-k);

      //pack L1A  M_ * K_ 
      double A_packed_L1[M_ * K_] __attribute__((aligned(16))); 
      col_major_pack(A_packed_L1, A, i*K+k, M_, K_, K);

      for (int j = 0; j < N; j+= BLOCK_SIZE_L1){ //for each cols of B
        
        int N_ = min(BLOCK_SIZE_L1, N-j);

        //pack L1B  K_ * N_
        double B_packed_L1[K_ * N_] __attribute__((aligned(16))); 
        col_major_pack(B_packed_L1, B, j*K+k, N_, K_, K);

        matrix_mult(lda, A_packed_L1, B_packed_L1, C+i+j*M, M_, N_, K_);
      }
    }
  }
}

void square_dgemm(int lda, double* A, double* B, double* C){
  double A_packed[lda * lda] __attribute__((aligned(16)));
  double B_packed[lda * lda] __attribute__((aligned(16)));

  //pack A transpose
  row_major_pack(A_packed, A, 0, lda, lda, lda);
  //pack B
  col_major_pack(B_packed, B, 0, lda, lda, lda);

  L1_divide(lda, A_packed, B_packed, C, lda, lda, lda);

}
