#include "gpu_reduction.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cfloat>
#include <cstdlib>
#include <stdexcept>
#include <iostream>

#define USE_THRUST

#ifdef USE_THRUST

#ifdef _MSC_VER
#pragma warning(disable : 4996)
#pragma warning(disable : 4267)
#endif

#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>

template<class T>
struct absdiff : public thrust::unary_function<thrust::tuple<T,T>,T> {
	__host__ __device__ T operator()(const thrust::tuple<T,T> & pair) const {
		return abs(thrust::get<0>(pair) - thrust::get<1>(pair));
	}
};

float gpu_reduce_max_absdiff(unsigned count, const float * w1, const float * w2) {
	typedef thrust::device_ptr<const float> tptr;
	typedef thrust::tuple<tptr,tptr> tptr2;
	tptr p1(w1);
	tptr p2(w2);
	thrust::zip_iterator<tptr2> first(thrust::make_tuple(p1, p2));
	thrust::zip_iterator<tptr2> last(thrust::make_tuple(p1 + count, p2 + count));
	return thrust::transform_reduce(first, last, absdiff<float>(), 0.0f, thrust::maximum<float>());
}

float gpu_reduce_max(unsigned count, const float * w) {
	thrust::device_ptr<const float> p(w);
	return thrust::reduce(p, p + count, FLT_MIN, thrust::maximum<float>());
}

#else

template<unsigned elements>
__global__ void gpu_reduce_max_absdiff_kernel(unsigned count, const float * w1, const float * w2, float * out);

template<unsigned elements>
__global__ void gpu_reduce_max_kernel(unsigned count, const float * w, float * out);

#define CUDA_CHECK(err) if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

float gpu_reduce_max_absdiff(unsigned count, const float * dev_w1, const float * dev_w2)
{
	// elements per thread to use in version 4
	static const unsigned elements = 1024;

	int dev;
	cudaDeviceProp devProp;

	CUDA_CHECK( cudaGetDevice(&dev) );
	CUDA_CHECK( cudaGetDeviceProperties(&devProp, dev) );

	float * dev_tmp1 = 0;
	float * dev_tmp2 = 0;

	unsigned maxThreads = std::min<unsigned>(devProp.maxThreadsPerBlock, static_cast<unsigned>(devProp.sharedMemPerBlock / 4));

	unsigned threads;
	unsigned blocks;

	threads = maxThreads;
	blocks = ((count + elements - 1)/elements + threads - 1) / threads;

	CUDA_CHECK( cudaMalloc(&dev_tmp1, blocks * sizeof(float)) );
	CUDA_CHECK( cudaMalloc(&dev_tmp2, blocks * sizeof(float)) );

	if (blocks > (unsigned)devProp.maxGridSize[0]) {
		throw std::runtime_error("Too many elements");
	}

	float result;
	
	gpu_reduce_max_absdiff_kernel<elements><<<blocks,threads,threads*sizeof(float)>>>(count, dev_w1, dev_w2, dev_tmp1);
	CUDA_CHECK( cudaDeviceSynchronize() );

	float * dev_tmp_in = dev_tmp1;
	float * dev_tmp_out = dev_tmp2;
	while (blocks > 1) {
		unsigned remaining = blocks;
		threads = std::min(maxThreads, remaining);
		blocks = ((remaining + 1) / 2 + threads - 1) / threads;
		gpu_reduce_max_kernel<1u><<<blocks,threads,threads*sizeof(float)>>>(remaining, dev_tmp_in, dev_tmp_out);
		CUDA_CHECK( cudaDeviceSynchronize() );
		// swap temporary buffers
		std::swap(dev_tmp_in, dev_tmp_out);
	}

	CUDA_CHECK( cudaMemcpy(&result, dev_tmp_in, 1 * sizeof(float), cudaMemcpyDeviceToHost) );

	cudaFree(dev_tmp1);
	cudaFree(dev_tmp2);

	return result;
}

float gpu_reduce_max(unsigned count, const float * dev_w)
{
	// elements per thread to use in version 4
	static const unsigned elements = 1024;

	int dev;
	cudaDeviceProp devProp;

	CUDA_CHECK( cudaGetDevice(&dev) );
	CUDA_CHECK( cudaGetDeviceProperties(&devProp, dev) );

	float * dev_tmp1 = 0;
	float * dev_tmp2 = 0;

	unsigned maxThreads = std::min<unsigned>(devProp.maxThreadsPerBlock, static_cast<unsigned>(devProp.sharedMemPerBlock / 4));

	unsigned threads;
	unsigned blocks;

	threads = maxThreads;
	blocks = ((count + elements - 1)/elements + threads - 1) / threads;

	CUDA_CHECK( cudaMalloc(&dev_tmp1, blocks * sizeof(float)) );
	CUDA_CHECK( cudaMalloc(&dev_tmp2, blocks * sizeof(float)) );

	if (blocks > (unsigned)devProp.maxGridSize[0]) {
		throw std::runtime_error("Too many elements");
	}

	float result;
	
	gpu_reduce_max_kernel<elements><<<blocks,threads,threads*sizeof(float)>>>(count, dev_w, dev_tmp1);
	CUDA_CHECK( cudaDeviceSynchronize() );

	float * dev_tmp_in = dev_tmp1;
	float * dev_tmp_out = dev_tmp2;
	while (blocks > 1) {
		unsigned remaining = blocks;
		threads = std::min(maxThreads, remaining);
		blocks = ((remaining + 1) / 2 + threads - 1) / threads;
		gpu_reduce_max_kernel<1u><<<blocks,threads,threads*sizeof(float)>>>(remaining, dev_tmp_in, dev_tmp_out);
		CUDA_CHECK( cudaDeviceSynchronize() );
		// swap temporary buffers
		std::swap(dev_tmp_in, dev_tmp_out);
	}

	CUDA_CHECK( cudaMemcpy(&result, dev_tmp_in, 1 * sizeof(float), cudaMemcpyDeviceToHost) );

	cudaFree(dev_tmp1);
	cudaFree(dev_tmp2);

	return result;
}

template<unsigned elements>
__global__ void gpu_reduce_max_absdiff_kernel(unsigned count, const float * w1, const float * w2, float * out)
{
	extern __shared__ float absdiff[];
	unsigned idx = elements * blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < count) {
		unsigned stride = blockDim.x;
		// first iteration in global memory
		float tmp = 0.0f;
		for (unsigned i = 0; i < elements; ++i) {
			unsigned e = idx + stride * i;
			if (e < count) {
				tmp = max(tmp, abs(w1[e]-w2[e]));
			}
		}
		absdiff[threadIdx.x] = tmp;
		stride >>= 1;
		__syncthreads();
		// other iterations in shared memory
		while (stride >= 64) {
			if (threadIdx.x < stride) {
				absdiff[threadIdx.x] = max(absdiff[threadIdx.x], absdiff[threadIdx.x + stride]);
			}
			stride >>= 1;
			__syncthreads();
		}
		if (threadIdx.x < 32) {
			absdiff[threadIdx.x] = max(absdiff[threadIdx.x], absdiff[threadIdx.x + 32]);
			__syncthreads();
			absdiff[threadIdx.x] = max(absdiff[threadIdx.x], absdiff[threadIdx.x + 16]);
			__syncthreads();
			absdiff[threadIdx.x] = max(absdiff[threadIdx.x], absdiff[threadIdx.x + 8]);
			__syncthreads();
			absdiff[threadIdx.x] = max(absdiff[threadIdx.x], absdiff[threadIdx.x + 4]);
			__syncthreads();
			absdiff[threadIdx.x] = max(absdiff[threadIdx.x], absdiff[threadIdx.x + 2]);
			__syncthreads();
			absdiff[threadIdx.x] = max(absdiff[threadIdx.x], absdiff[threadIdx.x + 1]);
			__syncthreads();
		}
		if (threadIdx.x == 0) {
			out[blockIdx.x] = absdiff[0];
		}
	}
}

template<unsigned elements>
__global__ void gpu_reduce_max_kernel(unsigned count, const float * w, float * out)
{
	extern __shared__ float values[];
	unsigned idx = elements * blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < count) {
		unsigned stride = blockDim.x;
		// first iteration in global memory
		float tmp = FLT_MIN;
		for (unsigned i = 0; i < elements; ++i) {
			unsigned e = idx + stride * i;
			if (e < count) {
				tmp = max(tmp, w[e]);
			}
		}
		values[threadIdx.x] = tmp;
		stride >>= 1;
		__syncthreads();
		while (stride >= 64) {
			if (threadIdx.x < stride) {
				values[threadIdx.x] = max(values[threadIdx.x], values[threadIdx.x + stride]);
			}
			stride >>= 1;
			__syncthreads();
		}
		if (threadIdx.x < 32) {
			values[threadIdx.x] = max(values[threadIdx.x], values[threadIdx.x + 32]);
			__syncthreads();
			values[threadIdx.x] = max(values[threadIdx.x], values[threadIdx.x + 16]);
			__syncthreads();
			values[threadIdx.x] = max(values[threadIdx.x], values[threadIdx.x + 8]);
			__syncthreads();
			values[threadIdx.x] = max(values[threadIdx.x], values[threadIdx.x + 4]);
			__syncthreads();
			values[threadIdx.x] = max(values[threadIdx.x], values[threadIdx.x + 2]);
			__syncthreads();
			values[threadIdx.x] = max(values[threadIdx.x], values[threadIdx.x + 1]);
			__syncthreads();
		}
		if (threadIdx.x == 0) {
			out[blockIdx.x] = values[0];
		}
	}
}

#endif