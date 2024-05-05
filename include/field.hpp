#ifndef _FIELD_HPP_
#define _FIELD_HPP_
#include <cuda_runtime.h>
#include "particle.hpp"


class Field
{
    __device__ virtual float3 get_force(const Particle* p) const = 0;
};

#endif