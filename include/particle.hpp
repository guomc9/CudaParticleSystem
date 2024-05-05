#ifndef __PARTICLE_HPP__
#define __PARTICLE_HPP__
#include <cuda_runtime.h>
#include "context.h"

class Particle
{
    private:
        uint32_t _begin_step;
        uint32_t _cur_step;
        uint32_t _end_step;
        float _mass;
        float _radius;
        float3 _velocity;
        float3 _position;

    public:
        __host__ __device__ Particle(){};
        __host__ __device__ Particle(const uint32_t begin_step, const uint32_t end_step, const float mass, const float radius, const float3 velocity, const float3 position):
        _begin_step(begin_step), _end_step(end_step), _mass(mass), _radius(radius), _velocity(velocity), _position(position){};

        inline __host__ __device__ uint32_t get_begin_step() const  {   return _begin_step; };
        inline __host__ __device__ uint32_t get_cur_step() const  {   return _cur_step; };
        inline __host__ __device__ uint32_t get_end_step() const  {   return _end_step; };
        inline __host__ __device__ float get_mass() const {    return _mass;   };
        inline __host__ __device__ float get_radius() const {   return _radius; };
        inline __host__ __device__ float3 get_velocity() const {   return _velocity;   };
        inline __host__ __device__ float3 get_position() const {   return _position;   };
        
        inline __host__ __device__ void set_begin_step(uint32_t begin_step)    {   _begin_step = begin_step;  };
        inline __host__ __device__ void set_cur_step(uint32_t cur_step)    {   _cur_step = cur_step;  };
        inline __host__ __device__ void set_end_step(uint32_t end_step) {   _end_step = end_step;   };
        inline __host__ __device__ void set_mass(float mass) { _mass = mass;   };
        inline __host__ __device__ void set_radius(float radius)    {   _radius = radius;  };
        inline __host__ __device__ void set_velocity(float3 velocity)  { _velocity = velocity; };
        inline __host__ __device__ void set_position(float3 position) {    _position = position;   };
};

#endif