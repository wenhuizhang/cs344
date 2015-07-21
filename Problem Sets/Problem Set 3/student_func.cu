/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Definition Range (HDR) image contains a wider variation of intensity
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

__global__ void reduce_minimum(float * d_out, const float * const d_in, const size_t numItem) {
    // sdata is allocated in the kernel call: 3rd arg to <<<b, t, shmem>>>
  extern __shared__ float sdata[];

  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  int tid  = threadIdx.x;

  // load shared mem from global mem
  sdata[tid] = 99999999999.0f;
  if (myId < numItem)
    sdata[tid] = d_in[myId];

  __syncthreads();            // make sure entire block is loaded!

  // do reduction in shared mem
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
        sdata[tid] = min(sdata[tid], sdata[tid + s]);
    }
    __syncthreads();        // make sure all adds at one stage are done!
  }

  // only thread 0 writes result for this block back to global mem
  if (tid == 0) {
    d_out[blockIdx.x] = sdata[0];
  }
}

__global__ void reduce_maximum(float * d_out, const float * const d_in, const size_t numItem) {
  // sdata is allocated in the kernel call: 3rd arg to <<<b, t, shmem>>>
  extern __shared__ float sdata[];

  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  int tid  = threadIdx.x;

  // load shared mem from global mem
  sdata[tid] = -99999999999.0f;
  if (myId < numItem)
    sdata[tid] = d_in[myId];

  __syncthreads();            // make sure entire block is loaded!

  // do reduction in shared mem
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] = max(sdata[tid], sdata[tid + s]);
    }
    __syncthreads();        // make sure all adds at one stage are done!
  }

  // only thread 0 writes result for this block back to global mem
  if (tid == 0) {
    d_out[blockIdx.x] = sdata[0];
  }
}

__global__ void histogram(unsigned int *d_bins, const float * const d_in, const size_t numBins, const float min_logLum, const float range, const size_t numRows, const size_t numCols) {
  
  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  if (myId >= (numRows * numCols))
    return;

  float myItem = d_in[myId];
  int myBin = (myItem - min_logLum) / range * numBins;
  atomicAdd(&(d_bins[myBin]), 1);
}

__global__ void scan(unsigned int *d_out, unsigned int *d_sums, const unsigned int * const d_in, const unsigned int numBins, const unsigned int numElems)  {

  extern __shared__ float sdata[];
  int myId = blockIdx.x * blockDim.x + threadIdx.x;
  int tid = threadIdx.x;
  int offset = 1;

  // load two items per thread into shared memory
  if ((2 * myId) < numBins) {
    sdata[2 * tid] = d_in[2 * myId];
  }
  else {
    sdata[2 * tid] = 0;
  }
  
  if ((2 * myId + 1) < numBins) {
    sdata[2 * tid + 1] = d_in[2 * myId + 1];
  }
  else {
    sdata[2 * tid + 1] = 0;
  }

 	// Reduce
  for (unsigned int d = numElems >> 1; d > 0; d >>= 1) {
    if (tid < d)  {
      int ai = offset * (2 * tid + 1) - 1;
      int bi = offset * (2 * tid + 2) - 1;
      sdata[bi] += sdata[ai];
    }
    offset *= 2;
    __syncthreads();
  }
    
  // clear the last element
  if (tid == 0) {
    d_sums[blockIdx.x] = sdata[numElems - 1];
    sdata[numElems - 1] = 0;
  }
  
  // Down Sweep
  for (unsigned int d = 1; d < numElems; d *= 2) {
    offset >>= 1;
    if (tid < d) {
      int ai = offset * (2 * tid + 1) - 1;
      int bi = offset * (2 * tid + 2) - 1;
 	    float t = sdata[ai];
      sdata[ai] = sdata[bi];
      sdata[bi] += t;
    }
    __syncthreads();
  }
 
  // write the output to global memory
  if ((2 * myId) < numBins) {
    d_out[2 * myId] = sdata[2 * tid];
  }
  if ((2 * myId + 1) < numBins) {
    d_out[2 * myId + 1] = sdata[2 * tid + 1];
  }
}

// This version only works for one single block! The size of the array of items
__global__ void scan2(unsigned int *d_out, const unsigned int * const d_in, const unsigned int numBins, const unsigned int numElems)  {

  extern __shared__ float sdata[];
  int tid = threadIdx.x;
  int offset = 1;

  // load two items per thread into shared memory
  if ((2 * tid) < numBins) {
    sdata[2 * tid] = d_in[2 * tid];  
  }
  else {
    sdata[2 * tid] = 0;
  }

  if ((2 * tid + 1) < numBins) {
    sdata[2 * tid + 1] = d_in[2 * tid + 1];  
  }
  else {
    sdata[2 * tid + 1] = 0;
  }

 	// Reduce
  for (unsigned int d = numElems >> 1; d > 0; d >>= 1) {
    if (tid < d)  {
      int ai = offset * (2 * tid + 1) - 1;
      int bi = offset * (2 * tid + 2) - 1;
      sdata[bi] += sdata[ai];
    }
    offset *= 2;
    __syncthreads();
  }
    
  // clear the last element
  if (tid == 0) {
    sdata[numElems - 1] = 0;
  }
  
  // Down Sweep
  for (unsigned int d = 1; d < numElems; d *= 2) {
    offset >>= 1;
    if (tid < d) {
      int ai = offset * (2 * tid + 1) - 1;
      int bi = offset * (2 * tid + 2) - 1;
 	    float t = sdata[ai];
      sdata[ai] = sdata[bi];
      sdata[bi] += t;
    }
    __syncthreads();
  }
 
  // write the output to global memory
  if ((2 * tid) < numBins) {
    d_out[2 * tid] = sdata[2 * tid];
  }

  if ((2 * tid + 1) < numBins) {
    d_out[2 * tid + 1] = sdata[2 * tid + 1];
  }
}

__global__ void add_scan(unsigned int *d_out, const unsigned int * const d_in, const unsigned int numBins) {

  if (blockIdx.x == 0)
    return;

  int myId = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int myOffset = d_in[blockIdx.x];

  if ((2 * myId) < numBins) {
    d_out[2 * myId] += myOffset;
  }
  if ((2 * myId + 1) < numBins) {
    d_out[2 * myId + 1] += myOffset;
  }

}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */

  // Initialization
  unsigned int numItem = numRows * numCols;
  dim3 blockSize(256, 1, 1);
  dim3 gridSize(numItem / blockSize.x + 1, 1, 1);
    
  float * d_inter_min;
  float * d_inter_max;
  unsigned int * d_histogram;
  unsigned int * d_sums;
  unsigned int * d_incr;

  checkCudaErrors(cudaMalloc(&d_inter_min, sizeof(float) * gridSize.x));
  checkCudaErrors(cudaMalloc(&d_inter_max, sizeof(float) * gridSize.x));
  checkCudaErrors(cudaMalloc(&d_histogram, sizeof(unsigned int) * numBins));
  checkCudaErrors(cudaMemset(d_histogram, 0, sizeof(unsigned int) * numBins));
     
  // Step 1: Reduce (min and max). It could be done in one step only!
  reduce_minimum<<<gridSize, blockSize, sizeof(float) * blockSize.x>>>(d_inter_min, d_logLuminance, numItem);
  reduce_maximum<<<gridSize, blockSize, sizeof(float) * blockSize.x>>>(d_inter_max, d_logLuminance, numItem);
  numItem = gridSize.x;
  gridSize.x = numItem / blockSize.x + 1;

  while (numItem > 1) {
    reduce_minimum<<<gridSize, blockSize, sizeof(float) * blockSize.x>>>(d_inter_min, d_inter_min, numItem);
    reduce_maximum<<<gridSize, blockSize, sizeof(float) * blockSize.x>>>(d_inter_max, d_inter_max, numItem);
    numItem = gridSize.x;
    gridSize.x = numItem / blockSize.x + 1;
  }

  // Step 2: Range
  checkCudaErrors(cudaMemcpy(&min_logLum, d_inter_min, sizeof(float), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(&max_logLum, d_inter_max, sizeof(float), cudaMemcpyDeviceToHost));

  float range = max_logLum - min_logLum;

  // Step 3: Histogram
  gridSize.x = numRows * numCols / blockSize.x + 1;
  histogram<<<gridSize, blockSize>>>(d_histogram, d_logLuminance, numBins, min_logLum, range, numRows, numCols);

  // Step 4: Exclusive scan - Blelloch
  unsigned int numElems = 256;
  blockSize.x = numElems / 2;
  gridSize.x = numBins / numElems;
  if (numBins % numElems != 0)
    gridSize.x++;
  checkCudaErrors(cudaMalloc(&d_sums, sizeof(unsigned int) * gridSize.x));
  checkCudaErrors(cudaMemset(d_sums, 0, sizeof(unsigned int) * gridSize.x));

  // First-level scan to obtain the scanned blocks
  scan<<<gridSize, blockSize, sizeof(float) * numElems>>>(d_cdf, d_sums, d_histogram, numBins, numElems);

  // Second-level scan to obtain the scanned blocks sums
  numElems = gridSize.x;

  // Look for the next power of 2 (32 bits)
  unsigned int nextPow = numElems;
  nextPow--;
  nextPow = (nextPow >> 1) | nextPow;
  nextPow = (nextPow >> 2) | nextPow;
  nextPow = (nextPow >> 4) | nextPow;
  nextPow = (nextPow >> 8) | nextPow;
  nextPow = (nextPow >> 16) | nextPow;
  nextPow++;

  blockSize.x = nextPow / 2;
  gridSize.x = 1;
  checkCudaErrors(cudaMalloc(&d_incr, sizeof(unsigned int) * numElems));
  checkCudaErrors(cudaMemset(d_incr, 0, sizeof(unsigned int) * numElems));
  scan2<<<gridSize, blockSize, sizeof(float) * nextPow>>>(d_incr, d_sums, numElems, nextPow);

  // Add scanned block sum i to all values of scanned block i
  numElems = 256;
  blockSize.x = numElems / 2;
  gridSize.x = numBins / numElems;
  if (numBins % numElems != 0)
    gridSize.x++;
  add_scan<<<gridSize, blockSize>>>(d_cdf, d_incr, numBins);

  // Clean memory
  checkCudaErrors(cudaFree(d_inter_min));
  checkCudaErrors(cudaFree(d_inter_max));
  checkCudaErrors(cudaFree(d_histogram));
  checkCudaErrors(cudaFree(d_sums));
  checkCudaErrors(cudaFree(d_incr));
}
