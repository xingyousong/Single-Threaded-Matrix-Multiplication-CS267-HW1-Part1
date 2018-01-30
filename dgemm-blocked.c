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


const char* dgemm_desc = "My Simple blocked dgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 128
#endif

#define min(a,b) (((a)<(b))?(a):(b))
#define calculateOffset(row, col, numRow) (col * numRow + row)

//utility functions
void pack(double* dest, double* src, int offset, int width, int height, int lda){
  int numBytes = height * sizeof(double);
  double* pointer_src = src + offset;
  double* pointer_dest = dest;
  for (int j = 0; j < width / 4; j++){
    //unrolling here to leverage on concurrent computation
    memmove(pointer_dest, pointer_src, numBytes);
    memmove(pointer_dest + height, pointer_src + lda, numBytes);
    memmove(pointer_dest + (height << 1), pointer_src + (lda << 1), numBytes);
    memmove(pointer_dest + (height * 3), pointer_src + (lda * 3), numBytes);
    pointer_dest  += (height << 2);
    pointer_src   += (lda << 2);
  }
  for (int j = (width / 4) * 4; j < width; j++){
    memmove(pointer_dest, pointer_src, numBytes);
    pointer_dest  += height;
    pointer_src   += lda;
  }
}


/*
Each block
A = m_c * k_c
B = k_c * n_r
C = m_c * n_r
*/
void SSE_FOUR(double* A, double* B, double* C, int m_c, int n_r, int k_c, int lda){
  __m128d A_col_1, A_col_2;
  __m128d B_vec;
  __m128d C_col_1, C_col_2;

  for (int i = 0; i < 4; ++i){
    C_col_1 = _mm_loadu_pd(&C[lda * i]);
    C_col_2 = _mm_loadu_pd(&C[lda * i + 2]);
    for (int j = 0; j < 4; ++j){
      A_col_1 = _mm_loadu_pd(&A[m_c *j]);
      A_col_2 = _mm_loadu_pd(&A[m_c *j+2]);
      B_vec = _mm_set1_pd(B[j + i * k_c]);

      C_col_1 = _mm_add_pd(_mm_mul_pd(A_col_1, B_vec), C_col_1);
      C_col_2 = _mm_add_pd(_mm_mul_pd(A_col_2, B_vec), C_col_2);
    }
    _mm_storeu_pd(&C[lda * i], C_col_1);
    _mm_storeu_pd(&C[lda * i + 2], C_col_2);
  }
}

void do_block(int lda, int M, int N, int K, int m_c, int n_r, int k_c, double* A, double* B, double* C){
  for (int j = 0; j < N; ++j){
      
    for (int p = 0; p < K; ++p){
      int c_pos = calculateOffset(0, j, lda);
      int b_pos = calculateOffset(p, j, k_c);
      int a_pos = calculateOffset(0, p, m_c);
      for (int i = 0; i < M; ++i){
        C[c_pos++] += A[a_pos++] * B[b_pos];
      }
    }
  }
}

void do_block_SSE(int lda, int m_c, int n_r, int k_c, double* A, double* B, double* C){
  /* For each block-row of A */
  for (int j = 0; j < n_r; j += 4){
   int N = min (4, n_r-j);
   for (int k = 0; k < k_c; k += 4){
     int K = min (4, k_c-k);
     for (int i = 0; i < m_c; i += 4){
       int M = min (4, m_c-i);
       
       

       if (M % 4 == 0 && N % 4 == 0 && K % 4 == 0){
         SSE_FOUR(A+calculateOffset(i, k, m_c), B+calculateOffset(k, j, k_c), C+calculateOffset(i, j, lda), m_c, n_r, k_c, lda);
         //super_do_block(lda, 4, 4, 4, m_c, n_r, k_c, A+calculateOffset(i, k, m_c), B+calculateOffset(k, j, k_c), C+calculateOffset(i, j, lda));
       }else{
         do_block(lda, M, N, K, m_c, n_r, k_c, A+calculateOffset(i, k, m_c), B+calculateOffset(k, j, k_c), C+calculateOffset(i, j, lda));
        }
      }
    }
  }
}

void GEBP(int lda, int i, int p, double* A, double* B, double* C, double* packed_A, double* packed_B, int k_c, int m_c){
  //pack A into packed_A
  pack(packed_A, A, calculateOffset(i, p, lda), k_c, m_c, lda);

  for (int j = 0; j < lda; j += BLOCK_SIZE){
    int n_r = min(BLOCK_SIZE, lda - j);
    //call work horse here
    do_block_SSE(lda, m_c, n_r, k_c,
      packed_A,
      packed_B + calculateOffset(0, j, k_c),
      C + calculateOffset(i, j, lda));
  }
}

void GEPP(int lda, int p, double* A, double* B, double* C, double* packed_A, double* packed_B, int k_c){
  // //pack B into packed_B
  pack(packed_B, B, calculateOffset(p, 0, lda), lda, k_c, lda);

  //break each column of A into multiple blocks
  for (int i = 0; i < lda; i += BLOCK_SIZE){
    int m_c = min(BLOCK_SIZE, lda - i);
    //call GEBP here
    GEBP(lda, i, p, A, B, C, packed_A, packed_B, k_c, m_c);
  }
}


void square_dgemm(int lda, double* A, double* B, double* C){
  //prealocaate memory here
  double* packed_A = malloc((BLOCK_SIZE * BLOCK_SIZE + 1) * sizeof(double));
  double* packed_B = malloc((BLOCK_SIZE * lda + 1) * sizeof(double));

  //break B into multiple rows of size: k_c * lda
  //break A into multiple cols of size: lda * k_c
  for (int p = 0; p < lda; p += BLOCK_SIZE){
    int k_c = min(BLOCK_SIZE, lda - p);
    //call GEPP here
    GEPP(lda, p, A, B, C, packed_A, packed_B, k_c);
  }

  free(packed_A);
  free(packed_B);
}
