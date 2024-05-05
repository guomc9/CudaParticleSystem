#ifndef __GRAVITY_FIELD_HPP__
#define __GRAVITY_FIELD_HPP__
#include "field.hpp"
#include "context.h"

class GravityField : Field
{
    public:
        __device__ float3 get_force(const Particle* p) const override
        {
            const float g = 10.0f;
            float3 force{0.0f, -p->get_mass() * g, 0.f};
            return force;
        }
};


#endif