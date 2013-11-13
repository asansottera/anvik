#ifndef ANVIK_GPU_REDUCTION_H
#define ANVIK_GPU_REDUCTION_H

float gpu_reduce_max_absdiff(unsigned count, const float * w1, const float * w2);

float gpu_reduce_max(unsigned count, const float * w);

#endif