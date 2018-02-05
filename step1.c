/*
    Please include compiler name below (you may also include any other modules you would like to be loaded)

COMPILER= gnu

    Please include All compiler flags and libraries as you want them run. You can simply copy this over from the Makefile's first few lines

CC = cc
OPT = -O3
CFLAGS = -Wall -std=gnu99 $(OPT)
MKLROOT = /opt/intel/composer_xe_2013.1.117/mkl
LDLIBS = -lrt -Wl,--start-group $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_sequential.a $(MKLROOT)/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm

*/

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <xmmintrin.h>


const char* dgemm_desc = "SSE_0.1 Yao Simple blocked dgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 64
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
void matrix_mult(double* A, double* B, double* C, int M, int K, int N){
  //for each row of A
  for (int i = 0; i < M; i++){
    //for each row of B
    for (int j = 0; j < N; j++){
      //computer row vec * col vec + val
      //a_row . b_col + c
      row_mul_col_plus_val(A+i*K, B+j*K, C+i+j*M, K);
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

  matrix_mult(A_packed, B_packed, C, lda, lda, lda);

}
