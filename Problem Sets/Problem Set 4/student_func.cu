//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>

/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */


__global__ void radsort(unsigned int* d_in,
			unsigned int* d_inpos,
			unsigned int* d_out,
			unsigned int* d_outpos,
			size_t N)
{
  extern __shared__ int edata[];
  unsigned int* idata  = (unsigned int*) edata + 0*N;
  unsigned int* din    = (unsigned int*) edata + 1*N;
  unsigned int* dout   = (unsigned int*) edata + 2*N;
  unsigned int* inpos  = (unsigned int*) edata + 3*N;
  unsigned int* outpos = (unsigned int*) edata + 4*N;
  unsigned int tid = threadIdx.x;

  unsigned int last;
  din[tid] = d_in[tid];
  inpos[tid] = d_inpos[tid];
  
  for(unsigned int sortmask = 1; sortmask != 0; sortmask <<= 1){
    /* First half: sum scan the elements with 0 at the sortmask */
    /* fill idata[] */
    idata[tid] = ( (din[tid] & sortmask) == 0) ? 1 : 0;
    __syncthreads();

    /* scan phase I */
    for (size_t skip = 1; skip < N; skip<<=1 ) {
      size_t mask = (skip<<1)-1;
      if( ( tid & mask ) == mask ) idata[tid] += idata[tid - skip];
      __syncthreads();
    }
    last = idata[N-1]; // save last
    /* scan phase II */
    idata[N-1] = 0;
    for (size_t skip = N/2; skip > 0; skip>>=1 ) {
      unsigned int mask = (skip<<1)-1;
      if( ( tid & mask ) == mask) {
	float tmp = idata[tid] + idata[tid - skip];
	idata[tid - skip] = idata[tid];
	idata[tid]=tmp;
      }
      __syncthreads();
    }
    /* save indices to dout[] */
    if( (din[tid] & sortmask) == 0 ) {
      dout[idata[tid]] = din[tid];
      outpos[idata[tid]] = inpos[tid];
    }
    __syncthreads();

    //////////////////////////////////// HALF /////////////////////////////
    /* First half: sum scan the elements with 1 at the sortmask */
    /* fill idata[] */
    idata[tid] = ( (din[tid] & sortmask) == sortmask ) ? 1 : 0;
    __syncthreads();

    /* scan phase I */
    for (size_t skip = 1; skip < N; skip<<=1 ) {
      size_t mask = (skip<<1)-1;
      if( ( tid & mask ) == mask ) idata[tid] += idata[tid - skip];
      __syncthreads();
    }
    /* scan phase II */
    idata[N-1] = last; // TODO try with 'last' instead of '0'
    for (size_t skip = N/2; skip > 0; skip>>=1 ) {
      unsigned int mask = (skip<<1)-1;
      if( ( tid & mask ) == mask) {
	float tmp = idata[tid] + idata[tid - skip];
	idata[tid - skip] = idata[tid];
	idata[tid]=tmp;
      }
      __syncthreads();
    }
    /* save indices to dout[] */
    if( (din[tid] & sortmask) == sortmask ) {
      dout[idata[tid]] = din[tid];
      outpos[idata[tid]] = inpos[tid];
    }
    __syncthreads();

    /* put dout[] to din[] as an input to the next sortmask */
    din[tid]=dout[tid];
  }
  d_out[tid] = dout[tid];
  d_outpos[tid] = outpos[tid];
}


void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{
  std::cout << "HELLOOOO:"  << numElems << std::endl;
  //TODO
  //PUT YOUR SORT HERE
  /*
    radsort<<<1,numElems,5*numElems*sizeof(int)>>>( d_inputVals,
				    d_inputPos,
				    d_outputVals,
				    d_outputPos,
				    numElems);
  //*/
}
