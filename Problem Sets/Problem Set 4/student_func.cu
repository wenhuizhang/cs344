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


void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{ 
  //TODO
  //PUT YOUR SORT HERE
  
  int threads_per_block = 512;
    int num_blocks = ceil( float(num_elems) /threads_per_block );
    
    unsigned int h_temp[num_elems];  
    unsigned int *d_map_out_group_0;
    unsigned int *d_map_out_group_1;
    
    unsigned int *d_scan_values;
    unsigned int *d_temp0;
    unsigned int *d_temp1;
    unsigned int *d_sum_results;
    unsigned int *d_sort_addresses;
    
    for( int i = 1; i <= 4; i <<= 1){
        map_kernel<<<num_blocks, threads_per_block>>>(d_outputVals, d_map_out_group_0, numElems, i, 0);
        map_kernel<<<num_blocks, threads_per_block>>>(d_outputVals, d_map_out_group_1, numElems, i, 1);
        
        scan_kernel<<<num_blocks, threads_per_block>>>(d_map_out_group_0, d_scan_values, d_temp, d_sum_results, numElems);
        sum_scan_kernel<<<num_blocks, threads_per_block>>>(d_scan_values, d_sum_results, numElems);

        unsigned int addr;
        addr++;
        printf("Addr: %u \n", addr);
        printf("i: %u \n", i);
        
        scan_kernel<<<num_blocks, threads_per_block>>>(d_map_out_group_1, d_scan_values, d_sum_results, numElems);
        sum_scan_kernel<<<num_blocks, threads_per_block>>>(d_scan_values, d_sum_results, numElems);
        
        cudaDeviceSynchronize();
        
        map_add_kernel<<<num_blocks, threads_per_block>>>(d_scan_values, d_sort_addresses, d_map_out_group_1, numElems, addr);
        
        resort_addresses<<<num_blocks, threads_per_block>>>(d_outputVals, d_outputPos, d_sort_addresses, d_temp0, d_temp1, numElems, 0);
        resort_addresses<<<num_blocks, threads_per_block>>>(d_outputVals, d_outputPos, d_sort_addresses, d_temp0, d_temp1, numElems, 1);
        
        
        
        
        
        
}
