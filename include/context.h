#ifndef __CONTEXT_H__
#define __CONTEXT_H__
#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "particle.hpp"
#include <math.h>
#define FIELDS				GravityField, WhirlWindField
#define M_2_PI          	6.283185307179586
#define M_PI            	3.141592653589793
#define PARTICLE_PER_BLOCK  256
#define SHAPE_PER_BLOCK     256
#define VERTICE_PER_BLOCK 	512
#define TRIANGLE_PER_BLOCK 	256
#define NUM_BLOCKS			256
#define BLOCK_SIZE			256
#define BLOCK_X				16
#define BLOCK_Y				16
#define CLIP_BOUND			1.3f
#define EPSILON				0.00001f
#define max(a, b) 			(a > b ? a : b)
#define min(a, b) 			(a < b ? a : b)

__forceinline__ void checkCudaError(cudaError_t result, char const* const func, const char* const file, int const line)
{
    if (result) {
        std::cerr << "CUDA error at " << file << ":" << line << " code=" << static_cast<unsigned int>(result) << " \"" << cudaGetErrorString(result) << "\" \n";
        cudaDeviceReset();
        exit(99);
    }
}

__forceinline__ void checkKernelError(const char* const file, int const line)
{
	auto err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		std::cerr << "CUDA error at " << file << ":" << line << " code=" << static_cast<unsigned int>(err) << " \"" << cudaGetErrorString(err) << "\" \n";
    	cudaDeviceReset();
    	exit(99);
	}
}

#define CHECK_CUDA_ERROR(val) checkCudaError((val), #val, __FILE__, __LINE__)
#define CHECK_KERNEL_ERROR() checkKernelError(__FILE__, __LINE__)

template<typename T>
__forceinline__ __device__ T rand_intep(T min_bound, T max_bound, curandState* state)
{
    float eta = curand_uniform(state);
    return static_cast<T>(min_bound + (max_bound - min_bound) * eta);
}

__forceinline__ __device__ float3 rand_intep_float3(float3 min_bound, float3 max_bound, curandState* state)
{
    float eta_x = curand_uniform(state);
    float eta_y = curand_uniform(state);
    float eta_z = curand_uniform(state);

	float3 delta = make_float3(max_bound.x - min_bound.x, max_bound.y - min_bound.y, max_bound.z - min_bound.z);
    return make_float3(min_bound.x + delta.x * eta_x, min_bound.y + delta.y * eta_y, min_bound.z + delta.z * eta_z);
}

__forceinline__ __device__ bool in_clip_space(const float3 v)
{
	return fabsf(v.x) < CLIP_BOUND && fabsf(v.y) < CLIP_BOUND && fabsf(v.z) < CLIP_BOUND;
}

__forceinline__ __device__ bool in_frustum(const float3 cv1, const float3 cv2, const float3 cv3)
{
	return in_clip_space(cv1) && in_clip_space(cv2) && in_clip_space(cv3);
}

__forceinline__ __device__ void getRect(const float2 v1, const float2 v2, const float2 v3, uint2& rect_min, uint2& rect_max, dim3 grid)
{
	float max_x = max(v1.x, max(v2.x, v3.x));
	float max_y = max(v1.y, max(v2.y, v3.y));
	float min_x = min(v1.x, min(v2.x, v3.x));
	float min_y = min(v1.y, min(v2.y, v3.y));

	rect_min = make_uint2(
		min(grid.x, max((uint32_t)0, (uint32_t)(min_x / BLOCK_X))),
		min(grid.y, max((uint32_t)0, (uint32_t)(min_y / BLOCK_Y)))
	);
	rect_max = make_uint2(
		min(grid.x, max((uint32_t)0, (uint32_t)((max_x + BLOCK_X - 1) / BLOCK_X))),
		min(grid.y, max((uint32_t)0, (uint32_t)((max_y + BLOCK_Y - 1) / BLOCK_Y)))
	);

}

__forceinline__ __device__ float3 transformPoint3x4(const float3 p, const float* matrix)
{
	return make_float3(
		matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z + matrix[3], 
		matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z + matrix[7], 
		matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z + matrix[11]
	);
}

__forceinline__ __device__ float4 transformPoint4x4(const float3 p, const float* matrix)
{
	return make_float4(
		matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z + matrix[3], 
		matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z + matrix[7], 
		matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z + matrix[11], 
		matrix[12] * p.x + matrix[13] * p.y + matrix[14] * p.z + matrix[15]
	);
}

__forceinline__ __device__ float3 fromHomo4(const float4 p)
{
	float inv_w = 1 / (p.w + 0.00001f);
	return make_float3(p.x * inv_w, p.y * inv_w, p.z * inv_w);
}

__forceinline__ __host__ __device__ float3 normalization(const float3 v)
{
	float norm = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z) + 0.00001f;
	return make_float3(v.x / norm, v.y / norm, v.z / norm);
}

__forceinline__ __host__ __device__ float cos_similarity_with_normalization(const float3 x, const float3 y)
{
    float dot_product = x.x * y.x + x.y * y.y + x.z * y.z;
	float norms = sqrtf(x.x * x.x + x.y * x.y + x.z * x.z) * sqrtf(y.x * y.x + y.y * y.y + y.z * y.z);
    return dot_product / (norms + 0.00001f);
}

__forceinline__ __host__ __device__ float cos_similarity(const float3 x, const float3 y)
{
    return x.x * y.x + x.y * y.y + x.z * y.z;
}

__forceinline__ __host__ __device__ float3 halfway_vector(const float3 l, const float3 v)
{
    float3 h;
    h.x = l.x + v.x;
    h.y = l.y + v.y;
    h.z = l.z + v.z;
    float inv_norm = 1 / (sqrtf(h.x * h.x + h.y * h.y + h.z * h.z) + 0.00001f);
    return make_float3(h.x * inv_norm, h.y * inv_norm, h.z * inv_norm);
}

__forceinline__ __host__ __device__ float3 cross_product(const float3 v1, const float3 v2)
{
	return make_float3(
        v1.y * v2.z - v1.z * v2.y,
        v1.z * v2.x - v1.x * v2.z,
        v1.x * v2.y - v1.y * v2.x
    );
}

__forceinline__ __host__ __device__ float dot_product(const float3 v1, const float3 v2)
{
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

__forceinline__ __host__ __device__ float3 sub_f3(const float3 v1, const float3 v2)
{
	return make_float3(
		v1.x - v2.x, 
		v1.y - v2.y, 
		v1.z - v2.z
	);
}

__forceinline__ __host__ __device__ float3 rotate_around_y(const float3 v, float angle)
{
    float cos_theta = cosf(angle);
    float sin_theta = sinf(angle);
    return make_float3(
        cos_theta * v.x + sin_theta * v.z,
        v.y,
        -sin_theta * v.x + cos_theta * v.z
    );
}

__forceinline__ __host__ __device__ float3 rotate_around_x(const float3 v, float angle)
{
    float cos_phi = cosf(angle);
    float sin_phi = sinf(angle);
    return make_float3(
        v.x,
        cos_phi * v.y - sin_phi * v.z,
        sin_phi * v.y + cos_phi * v.z
    );
}

__forceinline__ float* get_persp_matrix(const float near, const float far, const float ar, const float fov_y_rad)
{
	float* persp = new float[16];
	memset((void*)persp, 0, sizeof(float) * 16);
	float inv_tan_half_fov_y = 1 / tanf(fov_y_rad / 2);
	persp[0] = inv_tan_half_fov_y / ar;
	persp[5] = inv_tan_half_fov_y;
	persp[10] = (near + far) / (near - far);
	persp[11] = -2 * near * far / (near - far);
	persp[14] = 1;
	return persp;
}

__forceinline__ float* get_view_matrix(const float3 eye_pos, const float3 lookat, const float3 up)
{
    // Step 1: Create the forward vector from eye_pos to lookat
	float3 f = normalization(sub_f3(lookat, eye_pos));

    // Step 2: Create the right vector
	float3 r = normalization(cross_product(up, f));

    // Step 3: Define the up vector (re-calculate to ensure orthogonality)
	float3 u = normalization(cross_product(f, r));

    // Step 4: Construct the inverse view matrix
	float* view = new float[16];
	memset((void*)view, 0, sizeof(float) * 16);
	view[0] = r.x;
	view[1] = r.y;
	view[2] = r.z;
	
	view[4] = u.x;
	view[5] = u.y;
	view[6] = u.z;
	
	view[8] = f.x;
	view[9] = f.y;
	view[10] = f.z;
	
	view[3] = -dot_product(r, eye_pos);
    view[7] = -dot_product(u, eye_pos);
    view[11] = -dot_product(f, eye_pos);
    view[15] = 1;
    return view;
}

__forceinline__ float* get_view_persp_matrix(const float* view, const float* persp)
{
	float* view_persp = new float[16];
	memset((void*)view_persp, 0, sizeof(float) * 16);
	view_persp[0] = persp[0] * view[0];
	view_persp[1] = persp[0] * view[1];
	view_persp[2] = persp[0] * view[2];
	view_persp[3] = persp[0] * view[3];

	view_persp[4] = persp[5] * view[4];
	view_persp[5] = persp[5] * view[5];
	view_persp[6] = persp[5] * view[6];
	view_persp[7] = persp[5] * view[7];

	view_persp[8] = persp[10] * view[8] + persp[11] * view[12];
	view_persp[9] = persp[10] * view[9] + persp[11] * view[13];
	view_persp[10] = persp[10] * view[10] + persp[11] * view[14];
	view_persp[11] = persp[10] * view[11] + persp[11] * view[15];

	view_persp[12] = view[8];
	view_persp[13] = view[9];
	view_persp[14] = view[10];
	view_persp[15] = view[11];

	return view_persp;
}

__forceinline__ float* get_vp_matrix(uint32_t height, uint32_t width)
{
	float* vp = new float[16];
	memset((void*)vp, 0, sizeof(float) * 16);
	vp[0] = width * 0.5f;
	vp[3] = width * 0.5f;
	vp[5] = height * 0.5f;
	vp[7] = height * 0.5f;
	vp[10] = 1;
	vp[15] = 1;
	return vp;
}

__forceinline__ uint32_t getHigherMsb(uint32_t n)
{
    uint32_t msb = 0;
    for (int shift = 16; shift > 0; shift /= 2)
    {
        if (n >> shift)
        {
            n >>= shift;
            msb += shift;
        }
    }
    return msb + (n > 0);
}

#endif