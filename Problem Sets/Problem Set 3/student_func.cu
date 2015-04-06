// -*- mode: c++ -*-
/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"



__global__ void calcMax1(float* d_max_out,
			   const float* const d_max_in,
			   size_t n){

  if(blockIdx.x*blockDim.x+threadIdx.x >= n) return;

  extern __shared__ float sdata[];
  size_t tid = threadIdx.x;

  // Copy to shared memory
  size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  sdata[tid] = d_max_in[i]; /**/
  __syncthreads();
  
  for(size_t s=1; s < blockDim.x; s*=2){
    if(tid %(2*s)==0){
      sdata[tid] = max(sdata[tid],sdata[tid+s]);
    }
    __syncthreads();
  }
  
  if(tid==0){
    d_max_out[blockIdx.x] = sdata[0];
  }
}



__global__ void calcMax2(float* d_max_out,
			   const float* const d_max_in,
			   size_t n){

  if(blockIdx.x*blockDim.x+threadIdx.x >= n) return;

  extern __shared__ float sdata[];
  size_t tid = threadIdx.x;

  // Copy to shared memory
  size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  sdata[tid] = d_max_in[i]; /**/
  __syncthreads();
  
  for(size_t s=1; s < blockDim.x; s*=2){
    size_t index = 2*s*tid;
    if(index<blockDim.x){
      sdata[index] = max(sdata[index],sdata[index+s]);
    }
    __syncthreads();
  }
  
  if(tid==0){
    d_max_out[blockIdx.x] = sdata[0];
  }
}



__global__ void calcMax3(float* d_max_out,
			   const float* const d_max_in,
			   size_t n){

  if(blockIdx.x*blockDim.x+threadIdx.x >= n) return;

  extern __shared__ float sdata[];
  size_t tid = threadIdx.x;

  size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  sdata[tid] = d_max_in[i]; /**/
  __syncthreads();
  
  for(size_t s=blockDim.x/2; s>0; s>>=1) {
    if(tid < s)
      sdata[tid] = max(sdata[tid],sdata[tid+s]);
    __syncthreads();
  }
  
  if(tid==0){
    d_max_out[blockIdx.x] = sdata[0];
  }
}



__global__ void calcMax4(float* d_max_out,
			   const float* const d_max_in,
			   size_t n){
  extern __shared__ float sdata[];
  size_t tid = threadIdx.x;
  size_t i = blockDim.x * blockIdx.x*2 + tid;
  if(i+blockDim.x >= n) return;
  sdata[tid] = max(d_max_in[i],d_max_in[i+blockDim.x]); /**/
  __syncthreads();
  
  for(size_t s=blockDim.x/2; s>0; s>>=1) {
    if(tid < s)
      sdata[tid] = max(sdata[tid],sdata[tid+s]);
    __syncthreads();
  }
  
  if(tid==0){
    d_max_out[blockIdx.x] = sdata[0];
  }
}


__device__ void warpMax(volatile float* sdata, int tid){
  sdata[tid] = max(sdata[tid],sdata[tid+32]);
  sdata[tid] = max(sdata[tid],sdata[tid+16]);
  sdata[tid] = max(sdata[tid],sdata[tid+ 8]);
  sdata[tid] = max(sdata[tid],sdata[tid+ 4]);
  sdata[tid] = max(sdata[tid],sdata[tid+ 2]);
  sdata[tid] = max(sdata[tid],sdata[tid+ 1]);
}

__global__ void calcMax5(float* d_max_out,
			   const float* const d_max_in,
			   size_t n){
  extern __shared__ float sdata[];
  size_t tid = threadIdx.x;
  size_t i = blockDim.x * blockIdx.x*2 + tid;
  if(i+blockDim.x >= n) return;
  sdata[tid] = max(d_max_in[i],d_max_in[i+blockDim.x]); /**/
  __syncthreads();
  
  for(size_t s=blockDim.x/2; s>32; s>>=1) {
    if(tid < s)
      sdata[tid] = max(sdata[tid],sdata[tid+s]);
    __syncthreads();
  }
  if(tid < 32) warpMax(sdata,tid);
  
  if(tid==0){
    d_max_out[blockIdx.x] = sdata[0];
  }
}



__device__ void warpMin(volatile float* sdata, int tid){
  sdata[tid] = min(sdata[tid],sdata[tid+32]);
  sdata[tid] = min(sdata[tid],sdata[tid+16]);
  sdata[tid] = min(sdata[tid],sdata[tid+ 8]);
  sdata[tid] = min(sdata[tid],sdata[tid+ 4]);
  sdata[tid] = min(sdata[tid],sdata[tid+ 2]);
  sdata[tid] = min(sdata[tid],sdata[tid+ 1]);
}

__global__ void calcMinMax(float* d_min_out,
			   float* d_max_out,
			   const float* const d_min_in,
			   const float* const d_max_in,
			   size_t n)
{
  extern __shared__ float sdata[];
  size_t tid = threadIdx.x;
  size_t i = blockDim.x * blockIdx.x*2 + tid;
  if(i+blockDim.x >= n) return;
  sdata[tid] = max(d_max_in[i],d_max_in[i+blockDim.x]); /**/
  __syncthreads();
  
  for(size_t s=blockDim.x/2; s>32; s>>=1) {
    if(tid < s)
      sdata[tid] = max(sdata[tid],sdata[tid+s]);
    __syncthreads();
  }
  if(tid < 32) warpMax(sdata,tid);
  
  if(tid==0){
    d_max_out[blockIdx.x] = sdata[0];
  }
  sdata[tid] = min(d_min_in[i],d_min_in[i+blockDim.x]); /**/
  __syncthreads();
  
  for(size_t s=blockDim.x/2; s>32; s>>=1) {
    if(tid < s)
      sdata[tid] = min(sdata[tid],sdata[tid+s]);
    __syncthreads();
  }
  if(tid < 32) warpMin(sdata,tid);
  
  if(tid==0){
    d_min_out[blockIdx.x] = sdata[0];
  }
}




void check_d_mem(const float* const d_mem, size_t size, size_t offset){
  float *tmp; // delthis
  tmp = (float*)malloc(size*sizeof(float));
  checkCudaErrors(cudaMemcpy(tmp,d_mem,
			     size*sizeof(float),
			     cudaMemcpyDeviceToHost));
  //for(size_t i=0 ; i<size; i++) if(tmp[i]!=-4) std::cout << "good i: " << i << std::endl;
  for(size_t i=offset; i<offset+16; i++) std::cout << "tmp[" << i << "]=" << tmp[i] << std::endl;
  std::cout <<std::endl; //**/
  free(tmp);
}



void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  /*
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
  */
  size_t N = 1024;
  int blockSize = N;
  int gridSize = (numCols*numRows)/N;

  std::cout << "numCols: " << numCols;
  std::cout << ", numRows: " << numRows;
  std::cout << ", blockSize: " << blockSize;
  std::cout << ", gridSize: " << gridSize;
  std::cout << ", size: " << numCols*numRows;
  std::cout << std::endl;
  
  




  float *d_max_out,*d_min_out;
  checkCudaErrors(cudaMalloc(&d_max_out,numCols*numRows*sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_min_out,numCols*numRows*sizeof(float)));
  /*
#define calcMax calcMax5
  calcMax<<<gridSize/2,blockSize,blockSize*sizeof(float)>>>
    (d_max_out,d_logLuminance,numRows*numCols);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  // check_d_mem(d_max_out,numCols*numRows,0);

  calcMax<<<1,gridSize/2,blockSize*sizeof(float)>>>
    (d_max_out,d_max_out,numRows*numCols);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  // check_d_mem(d_max_out,numCols*numRows,0);
  /**/

  calcMinMax<<<gridSize/2,blockSize,blockSize*sizeof(float)>>>
    (d_min_out,d_max_out,d_logLuminance,d_logLuminance,numRows*numCols);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  calcMinMax<<<1,gridSize/2,blockSize*sizeof(float)>>>
    (d_min_out,d_max_out,d_min_out,d_max_out,numRows*numCols);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  
  float maxLogLum,minLogLum;
  checkCudaErrors(cudaMemcpy(&maxLogLum, d_max_out, sizeof(float), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(&minLogLum, d_min_out, sizeof(float), cudaMemcpyDeviceToHost));
  
  std::cout << "---> HI <---" << std::endl;
  std::cout << "MAX: " << maxLogLum << std::endl;
  std::cout << "MIN: " << minLogLum << std::endl;
  std::cout << "DIFF: " << maxLogLum - minLogLum << std::endl;

  // std::cout << std::min_element(h_logLuminance, h_logLuminance+numCols*numRows);
  std::cout << std::endl;
  //// logLumMax = std::max(h_logLuminance[i], logLumMax);

  /*
    //Step 2 
    2) subtract them to find the range
  */
  float logLumRange = max_logLum - min_logLum;

  logLumRange+=1.;
  logLumRange-=1.;
  N = (size_t)d_cdf;
  N= numBins;

  /*
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */

  //Step 3
  //next we use the now known range to compute
  //a histogram of numBins bins
  /*
  unsigned int *histo = new unsigned int[numBins];

  for (size_t i = 0; i < numBins; ++i) histo[i] = 0;

  for (size_t i = 0; i < numCols * numRows; ++i) {
    unsigned int bin = std::min(static_cast<unsigned int>(numBins - 1),
                           static_cast<unsigned int>((d_logLuminance[i] - min_logLum) / logLumRange * numBins));
    histo[bin]++;
  }

  //Step 4
  //finally we perform and exclusive scan (prefix sum)
  //on the histogram to get the cumulative distribution
  d_cdf[0] = 0;
  for (size_t i = 1; i < numBins; ++i) {
    d_cdf[i] = d_cdf[i - 1] + histo[i - 1];
  }

  delete[] histo;
  */
}
