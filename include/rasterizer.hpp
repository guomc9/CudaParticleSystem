#ifndef __RASTERIZAER_HPP__
#define __RASTERIZAER_HPP__
#include "context.h"

class BlinnPhong
{
	public:
		uint32_t size;
		float3 ambient;
		float shininess;
		float3* kas;
		float3* kds;
		float3* kss;

		BlinnPhong(const uint32_t size, const float3 ambient, const float shininess):
		size(size), ambient(ambient), shininess(shininess)
		{
			CHECK_CUDA_ERROR(cudaMalloc((void**)&kas, sizeof(float3) * size));
			CHECK_CUDA_ERROR(cudaMalloc((void**)&kds, sizeof(float3) * size));
			CHECK_CUDA_ERROR(cudaMalloc((void**)&kss, sizeof(float3) * size));
		};
		void generate();
		void free();
};



class Rasterizer
{
    private:
		bool binded = false;
		BlinnPhong* _materials;
		uint32_t _light_num;
        float3* _lights_pos;
        float3* _lights_emit;
        float3* _vertices;
		uint32_t _vertex_num;
        float3* _normals;
		uint32_t _shape_num;
        uint3* _shape_indices;
		bool* _mask;

    public:
        Rasterizer(){};
        void bind(uint32_t light_num, float3* lights_pos, float3* lights_emit, uint32_t vertex_num, float3* vertices, float3* normals, uint32_t shape_num, uint3* shape_indices, bool* mask);
        void render(uint32_t height, uint32_t width, float* view_matrix, float* view_persp_matrix, float* vp_matrix, cudaGraphicsResource* cuda_pbo_resource);
        void save_frame_buffer();
		void free();
};



#endif