#ifndef __SCENE_HPP__
#define __SCENE_HPP__
#include "context.h"

class Scene
{
    private:
        uint64_t _num_vertex;
        float3* _vertices;
        float3* _normals;
        uint64_t _num_shape;
        uint3* _indices;

    public:
        Scene(const uint64_t num_vertex, const float3* vertices, const float3* normals, const uint64_t num_shape, const uint3* indices):
        _num_vertex(num_vertex), _num_shape(num_shape)
        {
            CHECK_CUDA_ERROR(cudaMalloc((void**)&_vertices, num_vertex * sizeof(float3)));
            CHECK_CUDA_ERROR(cudaMalloc((void**)&_normals, num_vertex * sizeof(float3)));
            CHECK_CUDA_ERROR(cudaMalloc((void**)&_indices, num_shape * sizeof(uint3)));
            CHECK_CUDA_ERROR(cudaMemcpy((void*)_vertices, (void*)vertices, num_vertex * sizeof(float3), cudaMemcpyHostToDevice));
            CHECK_CUDA_ERROR(cudaMemcpy((void*)_normals, (void*)normals, num_vertex * sizeof(float3), cudaMemcpyHostToDevice));
            CHECK_CUDA_ERROR(cudaMemcpy((void*)_indices, (void*)indices, num_shape * sizeof(uint3), cudaMemcpyHostToDevice));
        }

        inline float3* get_vertices() const 
        {
            return _vertices;
        }

        inline float3* get_normals() const 
        {
            return _normals;
        }

        inline uint3* get_indices() const
        {
            return _indices;
        }

        inline uint64_t get_vertices_size() const
        {
            return _num_vertex;
        }

        inline uint64_t get_shape_size() const
        {
            return _num_shape;
        }

        void free()
        {
            CHECK_CUDA_ERROR(cudaFree(_vertices));
            CHECK_CUDA_ERROR(cudaFree(_normals));
            CHECK_CUDA_ERROR(cudaFree(_indices));
        }

};

#endif