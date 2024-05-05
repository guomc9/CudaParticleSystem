#ifndef __EXPORTER_HPP__
#define __EXPORTER_HPP__
#include <iostream>
#include <fstream>
#include <string>
#include "context.h"

class MeshExporter
{
    public:
        void bind(uint32_t vertex_num, uint32_t shape_num, float3* vertices, uint3* indices)
        {
            if (binded)
            {
                delete[] _vertices;
                delete[] _indices;
            }
            _vertex_num = vertex_num;
            _shape_num = shape_num;
            _vertices = new float3[vertex_num];
            _indices = new uint3[shape_num];
            binded = true;
            CHECK_CUDA_ERROR(cudaMemcpy((void*)_vertices, (void*)vertices, vertex_num * sizeof(float3), cudaMemcpyDeviceToHost));
            CHECK_CUDA_ERROR(cudaMemcpy((void*)_indices, (void*)indices, shape_num * sizeof(uint3), cudaMemcpyDeviceToHost));
        }

        bool saveToOBJ(const std::string& filename)
        {
            std::ofstream file(filename);
            if (!file.is_open())
            {
                std::cerr << "Failed to open file: " << filename << std::endl;
                return false;
            }

            for (uint32_t i = 0; i < _vertex_num; i++)
            {
                file << "v " << _vertices[i].x << ' ' << _vertices[i].z << ' ' << _vertices[i].y << '\n';
            }

            for (uint32_t i = 0; i < _shape_num; i++)
            {
                file << "f " << _indices[i].x + 1 << ' ' << _indices[i].y + 1 << ' ' << _indices[i].z + 1 << '\n';
            }

            file.close();
            return true;
        }

    private:
        bool binded = false;
        uint32_t _vertex_num;
        uint32_t _shape_num;
        float3* _vertices;
        uint3* _indices;
};
#endif