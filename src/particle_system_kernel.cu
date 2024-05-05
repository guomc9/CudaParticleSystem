#include <curand_kernel.h>
#include "particle_system.hpp"
#include "particle.hpp"
#include "field.hpp"

template<typename... F>
__device__ float3 integrate_force(Particle* p)
{
    float3 s = make_float3(0, 0, 0);
    using expander = int[];
    (void) expander{0, (s.x += F().get_force(p).x, s.y += F().get_force(p).y, s.z += F().get_force(p).z, 0)...};
    return s;
}

__global__ void init_kernel(const uint64_t particle_num, Particle* particles, const uint32_t min_steps, const uint32_t max_steps, const float min_radius, const float max_radius, const float min_mass, const float max_mass, const float3 min_bound, const float3 max_bound, const float3 min_velocity, const float3 max_velocity)
{
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= particle_num)
        return ;
    curandState state;
    curand_init(clock() + idx, 0, 0, &state);
    Particle p = particles[idx];
    p.set_radius(rand_intep<float>(min_radius, max_radius, &state));
    p.set_mass(rand_intep<float>(min_mass, max_mass, &state));
    p.set_begin_step(0);
    p.set_cur_step(0);
    p.set_end_step(rand_intep<uint32_t>(min_steps, max_steps, &state));
    p.set_position(rand_intep_float3(min_bound, max_bound, &state));
    p.set_velocity(rand_intep_float3(min_velocity, max_velocity, &state));
    particles[idx] = p;
}


__global__ void step_kernel(uint64_t particle_num, Particle* particles, bool* live, const float delta_time)
{
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= particle_num)
        return ;
    Particle p = particles[idx];
    if (p.get_cur_step() >= p.get_end_step())
    {
        live[idx] = false;
        return ;
    }
    else
    {
        p.set_cur_step(p.get_cur_step() + 1);
    }
    float half_delta_time_o2 = 0.5 * delta_time * delta_time;
    float3 force = integrate_force<FIELDS>(&p);
    float inv_mass = 1 / p.get_mass();
    float3 acc {force.x * inv_mass, force.y * inv_mass, force.z * inv_mass};
    float3 vel = p.get_velocity();
    float3 new_vel {vel.x + acc.x * delta_time, vel.y + acc.y * delta_time, vel.z + acc.z * delta_time};
    float3 pos = p.get_position();
    float3 new_pos {pos.x + vel.x * delta_time + acc.x * half_delta_time_o2, 
                    pos.y + vel.y * delta_time + acc.y * half_delta_time_o2, 
                    pos.z + vel.z * delta_time + acc.z * half_delta_time_o2};
    
    p.set_velocity(new_vel);
    p.set_position(new_pos);
    particles[idx] = p;
}

__global__ void generate_vertices_kernel(Particle* particles, float3* vertices, float3* normals, uint32_t num_vertices, uint32_t num_vertices_per_particle, uint32_t num_lati_lines, uint32_t num_longi_lines)
{
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_vertices)
    {
        int particle_idx = idx / num_vertices_per_particle;
        int localIdx = idx % num_vertices_per_particle;
        int i = localIdx / (num_lati_lines + 1);
        int j = localIdx % (num_lati_lines + 1);

        Particle p = particles[particle_idx];
        float radius = p.get_radius();
        float3 center = p.get_position();
        float phi = M_PI * i / num_longi_lines;
        float theta = M_2_PI * j / num_lati_lines;

        float dx = radius * sinf(phi) * cosf(theta);
        float dy = radius * sinf(phi) * sinf(theta);
        float dz = radius * cosf(phi);

        float x = dx + center.x;
        float y = dy + center.y;
        float z = dz + center.z;

        vertices[idx] = make_float3(x, y, z);
        normals[idx] = normalization(make_float3(dx, dy, dz));
    }
}

__global__ void generate_shapes_kernel(uint3 *indices, bool* live, bool* mask, uint32_t num_quad_shapes, uint32_t num_vertices_per_particle, uint32_t num_quads_per_particle, uint32_t num_lati_lines, uint32_t num_longi_lines)
{
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_quad_shapes)
    {
        int particle_idx = idx / num_quads_per_particle;
        
        int quad_idx_within_particle = idx % num_quads_per_particle;

        int i = quad_idx_within_particle / num_lati_lines;
        int j = quad_idx_within_particle % num_lati_lines;

        uint32_t top_left = particle_idx * num_vertices_per_particle + i * (num_lati_lines + 1) + j;
        uint32_t top_right = top_left + 1;
        uint32_t bottom_left = top_left + (num_lati_lines + 1);
        uint32_t bottom_right = bottom_left + 1;

        indices[idx * 2] = make_uint3(top_left, bottom_left, bottom_right);
        indices[idx * 2 + 1] = make_uint3(top_left, bottom_right, top_right);
        if (!live[particle_idx])
        {
            mask[idx * 2] = false;
            mask[idx * 2 + 1] = false;
        }
    }
}

__global__ void update_shapes_mask_kernel(uint3 *indices, bool* live, bool* mask, uint32_t num_quad_shapes, uint32_t num_quads_per_particle)
{
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_quad_shapes)
    {
        int particle_idx = idx / num_quads_per_particle;
        if (!live[particle_idx])
        {
            mask[idx * 2] = false;
            mask[idx * 2 + 1] = false;
        }
    }
}



void ParticleSystem::init()
{
    if (inited)
    {
        free();
    }
    CHECK_CUDA_ERROR(cudaMallocManaged((void**)&_particles, sizeof(Particle) * _particle_num));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&_live, sizeof(bool) * _particle_num));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&_vertices, sizeof(float3) * _num_vertices));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&_shape_indices, sizeof(uint3) * _num_shapes));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&_normals, sizeof(float3) * _num_vertices));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&_mask, sizeof(bool) * _num_shapes));
    CHECK_CUDA_ERROR(cudaMemset((void*)_live, true, sizeof(bool) * _particle_num));
    CHECK_CUDA_ERROR(cudaMemset((void*)_mask, true, sizeof(bool) * _num_shapes));
    init_kernel<<<(_particle_num + PARTICLE_PER_BLOCK - 1) / PARTICLE_PER_BLOCK,  PARTICLE_PER_BLOCK>>>(_particle_num, _particles, _min_steps, _max_steps, _min_radius, _max_radius, _min_mass, _max_mass, _min_bound, _max_bound, _min_velocity, _max_velocity);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    inited = true;
}

void ParticleSystem::step(const float delta_time)
{
    step_kernel<<<(_particle_num + PARTICLE_PER_BLOCK - 1) / PARTICLE_PER_BLOCK,  PARTICLE_PER_BLOCK>>>(_particle_num, _particles, _live, delta_time);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

void ParticleSystem::meshify()
{
    generate_shapes_kernel<<<(_num_shapes / 2 + SHAPE_PER_BLOCK - 1) / SHAPE_PER_BLOCK, SHAPE_PER_BLOCK>>>(_shape_indices, _live, _mask, _num_shapes / 2, _num_vertices_per_particle, _num_quads_per_particle, _num_lati_lines, _num_longi_lines);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

void ParticleSystem::refresh_vertices()
{
    generate_vertices_kernel<<<(_num_vertices + VERTICE_PER_BLOCK - 1) / VERTICE_PER_BLOCK, VERTICE_PER_BLOCK>>>(_particles, _vertices, _normals, _num_vertices, _num_vertices_per_particle, _num_lati_lines, _num_longi_lines);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    update_shapes_mask_kernel<<<(_num_shapes / 2 + SHAPE_PER_BLOCK - 1) / SHAPE_PER_BLOCK, SHAPE_PER_BLOCK>>>(_shape_indices, _live, _mask, _num_shapes / 2, _num_quads_per_particle);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

void ParticleSystem::free()
{
    CHECK_CUDA_ERROR(cudaFree(_particles));
    CHECK_CUDA_ERROR(cudaFree(_vertices));
    CHECK_CUDA_ERROR(cudaFree(_normals));
    CHECK_CUDA_ERROR(cudaFree(_shape_indices));
    CHECK_CUDA_ERROR(cudaFree(_live));
    CHECK_CUDA_ERROR(cudaFree(_mask));
    inited = false;
}

float3* ParticleSystem::get_vertices() const
{
    return _vertices;
}

bool* ParticleSystem::get_mask() const
{
    return _mask;
}

float3* ParticleSystem::get_normals() const
{
    return _normals;
}

uint3* ParticleSystem::get_shape_indices() const
{
    return _shape_indices;
}