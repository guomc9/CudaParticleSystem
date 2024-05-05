#ifndef __WHIRLWIND_FIELD_HPP__
#define __WHIRLWIND_FIELD_HPP__
#include "field.hpp"
#include "context.h"


class WhirlWindField : Field
{
    public:
        __device__ float3 get_force(const Particle* p) const override
        {
            const float3 center {0, 0, 0};
            const float P0 = 1000.0f;
            const float alpha = 0.0001f;
            const float beta = 20.0f;
            const float lambda = 0.2f;
            const float omega = 15.0f;
            float3 force {0, 0, 0};
            const float3 pos = p->get_position();
            float3 r = {center.x - pos.x, center.y - pos.y, 0};
            float r_norm = sqrt(r.x * r.x + r.y * r.y);
            r.x /= r_norm;
            r.y /= r_norm;
            float radius = p->get_radius();
            // float f_r = 2 * alpha * r_norm * P0 * expf(-2 * alpha * r_norm * r_norm) * 3.14f * radius * radius;
            float f_r = r_norm * P0 * expf(-2 * alpha * r_norm * r_norm) * 3.14f * radius * radius;
            float3 F_r {f_r * r.x / r_norm, f_r * r.y / r_norm, 0};
            
            float3 a {0, 0, 1};
            float3 F_z {0, 0, p->get_mass() * beta * expf(-lambda * fabsf(pos.z))};

            float3 t = {r.y, -r.x, 0};
            float f_t = p->get_mass() * omega * omega * r_norm;
            float3 F_t {t.x * f_t, t.y * f_t, 0};

            return float3{F_r.x + F_t.x, F_r.y + F_t.y, F_r.z + F_t.z + F_z.z};
        }
};


#endif