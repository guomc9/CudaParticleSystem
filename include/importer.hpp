#ifndef __IMPORTER_HPP__
#define __IMPORTER_HPP__
#include "context.h"
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>

// class Importer {
//     private:
//         std::string filename;
//     public:
//         Importer(const std::string& filename) : filename(filename) {}

//         std::vector<Point> importPoints()
//         {
//             std::vector<Point> points;
//             std::ifstream inFile(filename);
//             std::string line;
            
//             if (!inFile.is_open()) {
//                 std::cerr << "Failed to open file: " << filename << std::endl;
//                 return points;
//             }

//             while (std::getline(inFile, line) && line != "end_header") {
//                 // Skip header lines
//             }
            
//             while (std::getline(inFile, line)) {
//                 std::istringstream iss(line);
//                 Point pt;
//                 int r, g, b;
//                 iss >> pt.x >> pt.y >> pt.z >> r >> g >> b;
//                 pt.r = static_cast<unsigned char>(r);
//                 pt.g = static_cast<unsigned char>(g);
//                 pt.b = static_cast<unsigned char>(b);
//                 points.push_back(pt);
//             }
//             inFile.close();
//             return points;
//         }
// }
#endif