#ifndef __PARTICLE_SYSTEM_HPP__
#define __PARTICLE_SYSTEM_HPP__
#include <vector>
#include <stdint.h>
#include "particle.hpp"
#include "field.hpp"
#include "gravity_field.hpp"
#include "whirlwind_fileld.hpp"

class ParticleSystem
{
    private:
        bool inited = false;
        uint32_t _particle_num;
        uint32_t _min_steps;
        uint32_t _max_steps;
        float _min_radius;
        float _max_radius;
        float _min_mass;
        float _max_mass;
        float3 _min_bound;
        float3 _max_bound;
        float3 _min_velocity;
        float3 _max_velocity;
        uint32_t _num_lati_lines;
        uint32_t _num_longi_lines;
        Particle* _particles;
        uint32_t _num_vertices;
        uint32_t _num_shapes;
        uint32_t _num_vertices_per_particle;
        uint32_t _num_quads_per_particle;
        float3* _vertices;
        float3* _normals;
        uint3* _shape_indices;
        bool* _live;
        bool* _mask;

    public:
        ParticleSystem(const uint32_t particle_num, const uint32_t min_steps, const uint32_t max_steps, const float min_radius, const float max_radius, const float min_mass, const float max_mass, const float3 min_bound, const float3 max_bound, const float3 min_velocity, const float3 max_velocity, const uint32_t num_lati_lines=8, const uint32_t num_longi_lines=8):
        _particle_num(particle_num), _min_steps(min_steps), _max_steps(max_steps), _min_radius(min_radius), _max_radius(max_radius), _min_mass(min_mass), _max_mass(max_mass), _min_bound(min_bound), _max_bound(max_bound), _min_velocity(min_velocity), _max_velocity(max_velocity), _num_lati_lines(num_lati_lines), _num_longi_lines(num_longi_lines)
        {
            _num_vertices_per_particle = (_num_lati_lines + 1) * (_num_longi_lines + 1);
            _num_vertices = _particle_num * _num_vertices_per_particle;
            _num_quads_per_particle = _num_lati_lines * _num_longi_lines;
            _num_shapes = _particle_num * _num_quads_per_particle * 2;
        };

        ~ParticleSystem(){}
        
        void init();
        void step(const float delta_time);
        void refresh_vertices();
        void meshify();
        void free();
        uint32_t get_vertex_num() const {   return _num_vertices;   };
        uint32_t get_shape_num() const {    return _num_shapes;     };
        float3* get_vertices() const;
        float3* get_normals() const;
        uint3* get_shape_indices() const;
        bool* get_mask() const;
};

#endif