#ifndef __EXPORTER_HPP__
#define __EXPORTER_HPP__
#include <iostream>
#include <fstream>
#include <string>
#include "context.h"
#include <vector>

// class Exporter
// {
//     private:
//         std::string filename;
//     public:
//         Exporter(const std::string& filename) : filename(filename) {}

//         void bind() {};

//         void save(const std::vector<Point>& points)
//         {
//             std::ofstream outFile(filename);
//             if (!outFile.is_open())
//             {
//                 std::cerr << "Failed to open file for writing: " << filename << std::endl;
//                 return;
//             }

//             outFile << "ply\n";
//             outFile << "format ascii 1.0\n";
//             outFile << "element vertex " << points.size() << "\n";
//             outFile << "property float x\n";
//             outFile << "property float y\n";
//             outFile << "property float z\n";
//             outFile << "property uchar red\n";
//             outFile << "property uchar green\n";
//             outFile << "property uchar blue\n";
//             outFile << "end_header\n";

//             for (const auto& point : points)
//             {
//                 outFile << point.x << " " << point.y << " " << point.z << " ";
//                 outFile << static_cast<int>(point.r) << " " << static_cast<int>(point.g) << " " << static_cast<int>(point.b) << "\n";
//             }
//             outFile.close();
//         }
// };
#endif