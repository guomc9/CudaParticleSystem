#ifndef __IMPORTER_HPP__
#define __IMPORTER_HPP__
#include "context.h"
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>

class MeshImporter
{
    private:
        std::vector<float3> _vertices;
        std::vector<float3> _normals;
        std::vector<uint3> _indices;

    public:
        bool load(const std::string &filename)
        {
            std::ifstream file(filename);
            if (!file.is_open())
            {
                std::cerr << "Failed to open file: " + filename << std::endl;
                return false;
            }

            std::string line;
            while (getline(file, line))
            {
                std::stringstream ss(line);
                std::string prefix;
                ss >> prefix;
                if (prefix == "v")
                {
                    float x, y, z;
                    ss >> x >> y >> z;
                    _vertices.push_back(make_float3(x, y, z));
                }
                else if (prefix == "f")
                {
                    int v1, v2, v3;
                    ss >> v1 >> v2 >> v3;
                    _indices.push_back(make_uint3(v1 - 1, v2 - 1, v3 - 1));
                }
                else if (prefix == "vn")
                {
                    float x, y, z;
                    ss >> x >> y >> z;
                    _normals.push_back(make_float3(x, y, z));
                }
            }
            file.close();
            return true;
        }

        inline const float3* get_vertices() const 
        {
            return _vertices.data();
        }
        
        inline const float3* get_normals() const 
        {
            return _normals.data();
        }

        inline const uint3* get_indices() const
        {
            return _indices.data();
        }

        inline uint64_t get_vertices_size() const
        {
            return _vertices.size();
        }

        inline uint64_t get_shape_size() const
        {
            return _indices.size();
        }
};
#endif