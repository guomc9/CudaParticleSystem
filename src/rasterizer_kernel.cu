#include "rasterizer.hpp"
#include <cub/cub.cuh>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

__global__ void view_transform_kernel(
    const uint32_t V, 
    const float* vertices, 
    const float* view_matrix, 
    float* cam_vertices
)
{
    const uint32_t v_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (v_idx < V)
    {
        const float3 v = make_float3(vertices[v_idx * 3], vertices[v_idx * 3 + 1], vertices[v_idx * 3 + 2]);
        const float3 cv = fromHomo4(transformPoint4x4(v, view_matrix));
        cam_vertices[v_idx * 3] = cv.x;
        cam_vertices[v_idx * 3 + 1] = cv.y;
        cam_vertices[v_idx * 3 + 2] = cv.z;
    }
}

__global__ void view_persp_transform_kernel(
    const uint32_t V, 
    const float* vertices, 
    const float* view_persp_matrix, 
    float* clip_vertices
)
{
    const uint32_t v_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (V <= v_idx)
        return ;
    const float3 v = make_float3(vertices[v_idx * 3], vertices[v_idx * 3 + 1], vertices[v_idx * 3 + 2]);
    const float3 cv = fromHomo4(transformPoint4x4(v, view_persp_matrix));
    clip_vertices[v_idx * 3] = cv.x;
    clip_vertices[v_idx * 3 + 1] = cv.y;
    clip_vertices[v_idx * 3 + 2] = cv.z;
    // printf("clv = (%f, %f, %f)\n", cv.x, cv.y, cv.z);
}

__global__ void vp_transform_kernel(
    const uint32_t V, 
    const float* clip_vertices, 
    const float* vp_matrix, 
    float* screen_vertices
)
{
    const uint32_t v_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (V <= v_idx)
        return ;
    const float3 v = make_float3(clip_vertices[v_idx * 3], clip_vertices[v_idx * 3 + 1], clip_vertices[v_idx * 3 + 2]);
    const float3 sv = fromHomo4(transformPoint4x4(v, vp_matrix));
    screen_vertices[v_idx * 2] = sv.x;
    screen_vertices[v_idx * 2 + 1] = sv.y;
    // printf("v = (%f, %f), sv = (%f, %f)\n", v.x, v.y, sv.x, sv.y);
}

__global__ void replication_kernel(   
    const uint32_t S, 
    float* depths, 
	dim3 grid, 
    uint32_t* rect_mins, 
    uint32_t* rect_maxs, 
    uint32_t* touched_tiles, 
    uint32_t* offsets, 
    uint64_t* unsorted_key_list,
    uint32_t* unsorted_value_list
)
{
    const uint32_t shape_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (S <= shape_idx)
        return ;
    uint32_t k = (shape_idx == 0) ? 0 : offsets[shape_idx - 1];
    const uint2 rect_min = make_uint2(rect_mins[shape_idx * 2], rect_mins[shape_idx * 2 + 1]);
    const uint2 rect_max = make_uint2(rect_maxs[shape_idx * 2], rect_maxs[shape_idx * 2 + 1]);
    for (auto i = rect_min.y; i < rect_max.y; i++)
    {
        for (auto j = rect_min.x; j < rect_max.x; j++)
        {
            uint64_t key = i * grid.x + j;
            key <<= 32;
            key |= *((uint32_t*)&depths[shape_idx]);
            // printf("k=%d\n", k);
            unsorted_key_list[k] = key;
            unsorted_value_list[k] = shape_idx;
            k++;
        }
    }
}

__global__ void preprocess_kernel(
    const uint32_t S, 
    const float* cam_vertices, 
    const float* clip_vertices, 
    const float* screen_vertices, 
    const uint32_t* shape_indices, 
    const bool* mask, 
    const uint32_t image_width, 
    const uint32_t image_height, 
    const dim3 grid, 
    float* depths, 
    uint32_t* touched_tiles, 
    uint32_t* rect_mins, 
    uint32_t* rect_maxs
)
{
    const uint32_t shape_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (S <= shape_idx)
        return ;
    if (!mask[shape_idx])
    {
        touched_tiles[shape_idx] = 0;
        return ;
    }
    const uint32_t v1_idx = shape_indices[3 * shape_idx];
    const uint32_t v2_idx = shape_indices[3 * shape_idx + 1];
    const uint32_t v3_idx = shape_indices[3 * shape_idx + 2];
    
    const float3 cl_v1 = make_float3(clip_vertices[v1_idx * 3], clip_vertices[v1_idx * 3 + 1], clip_vertices[v1_idx * 3 + 2]);
    const float3 cl_v2 = make_float3(clip_vertices[v2_idx * 3], clip_vertices[v2_idx * 3 + 1], clip_vertices[v2_idx * 3 + 2]);
    const float3 cl_v3 = make_float3(clip_vertices[v3_idx * 3], clip_vertices[v3_idx * 3 + 1], clip_vertices[v3_idx * 3 + 2]);

    const float3 ca_v1 = make_float3(cam_vertices[v1_idx * 3], cam_vertices[v1_idx * 3 + 1], cam_vertices[v1_idx * 3 + 2]);
    const float3 ca_v2 = make_float3(cam_vertices[v2_idx * 3], cam_vertices[v2_idx * 3 + 1], cam_vertices[v2_idx * 3 + 2]);
    const float3 ca_v3 = make_float3(cam_vertices[v3_idx * 3], cam_vertices[v3_idx * 3 + 1], cam_vertices[v3_idx * 3 + 2]);

    const float2 s_v1 = make_float2(screen_vertices[v1_idx * 2], screen_vertices[v1_idx * 2 + 1]);
    const float2 s_v2 = make_float2(screen_vertices[v2_idx * 2], screen_vertices[v2_idx * 2 + 1]);
    const float2 s_v3 = make_float2(screen_vertices[v3_idx * 2], screen_vertices[v3_idx * 2 + 1]);

    // Clip Shape and Match Tiles
    if (!in_frustum(cl_v1, cl_v2, cl_v3))
    {
        touched_tiles[shape_idx] = 0;
        return ;
    }

    depths[shape_idx] = (ca_v1.z + ca_v2.z + ca_v3.z) / 3;
    uint2 rect_min, rect_max;
    getRect(s_v1, s_v2, s_v3, rect_min, rect_max, grid);
    touched_tiles[shape_idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
    rect_maxs[shape_idx * 2] = rect_max.x;
    rect_maxs[shape_idx * 2 + 1] = rect_max.y;
    rect_mins[shape_idx * 2] = rect_min.x;
    rect_mins[shape_idx * 2 + 1] = rect_min.y;
}

__global__ void identify_tile_ranges_kernel(
    const uint32_t L, 
    uint64_t* sorted_list_keys, 
    uint2* ranges
)
{
    auto idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= L)
		return ;

	uint64_t key = sorted_list_keys[idx];
	uint32_t currtile = key >> 32;
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = sorted_list_keys[idx - 1] >> 32;
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currtile].y = L;
}

__global__ void vertex_shader_kernel(
    const uint32_t V, 
    const uint32_t L, 
    const float* __restrict__ lights_pos, 
    const float* __restrict__ lights_emit, 
    const float* __restrict__ cam_vertices, 
    const float* __restrict__ normals, 
    const float* __restrict__ kds, 
    const float* __restrict__ kss, 
    const float* __restrict__ kas, 
    const float3 ambient, 
    const float shininess, 
    float* __restrict__ vertex_colors
)
{
    auto idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= V)
    {
        return ;
    }
    float3 v = make_float3(cam_vertices[3 * idx], cam_vertices[3 * idx + 1], cam_vertices[3 * idx + 2]);
    float3 ka = make_float3(kas[3 * idx], kas[3 * idx + 1], kas[3 * idx + 2]);
    float3 kd = make_float3(kds[3 * idx], kds[3 * idx + 1], kds[3 * idx + 2]);
    float3 ks = make_float3(kss[3 * idx], kss[3 * idx + 1], kss[3 * idx + 2]);
    float3 n = normalization(make_float3(normals[3 * idx], normals[3 * idx + 1], normals[3 * idx + 2]));
    float3 c = make_float3(0, 0, 0);
    for (auto i = 0; i < L; i++)
    {
        float3 lp = make_float3(lights_pos[3 * i], lights_pos[3 * i + 1], lights_pos[3 * i + 2]);
        float3 le = make_float3(lights_emit[3 * i], lights_emit[3 * i + 1], lights_emit[3 * i + 2]);
        float3 vl = make_float3(lp.x - v.x, lp.y - v.y, lp.z - v.z);
        float3 ve = make_float3(-v.x, -v.y, -v.z);
        float inv_r_2 = 1 / (vl.x * vl.x + vl.y * vl.y + vl.z * vl.z);
        
        float3 vl_normalized = normalization(vl);
        float3 ve_normalized = normalization(ve);

        float diff_cos = max(0, cos_similarity(vl_normalized, n));

        float3 h = halfway_vector(vl_normalized, ve_normalized);
        float spec_cos = max(0, cos_similarity(n, h));
        if (spec_cos > 0.00001f)
        {
            spec_cos = powf(spec_cos, shininess);
        }
        // Ambient
        float3 amb = {ka.x * ambient.x, ka.y * ambient.y, ka.z * ambient.z};
        // Diffuse
        float3 diff = {kd.x * diff_cos * inv_r_2 * le.x, kd.y * diff_cos * inv_r_2 * le.y, kd.z * diff_cos * inv_r_2 * le.z};
        // Specular
        float3 spec = {ks.x * spec_cos * inv_r_2 * le.x, ks.y * spec_cos * inv_r_2 * le.y, ks.z * spec_cos * inv_r_2 * le.z};

        c.x += (amb.x + diff.x + spec.x);
        c.y += (amb.y + diff.y + spec.y);
        c.z += (amb.z + diff.z + spec.z);
    }
    vertex_colors[3 * idx] = c.x;
    vertex_colors[3 * idx + 1] = c.y;
    vertex_colors[3 * idx + 2] = c.z;
    
    // vertex_colors[3 * idx] = kd.x;
    // vertex_colors[3 * idx + 1] = kd.y;
    // vertex_colors[3 * idx + 2] = kd.z;
    // if (c.x < EPSILON || c.y < EPSILON || c.z < EPSILON)
    //     printf("c = (%f, %f, %f)\n", c.x, c.y, c.z);
}


// template<uint32_t N_VERTICES>
// __global__ void __launch_bounds__(BLOCK_X * BLOCK_Y) fragment_shader_kernel(
//     const uint32_t height, 
//     const uint32_t width, 
//     const float* __restrict__ screen_vertices, 
//     const float* __restrict__ vertex_colors, 
//     const uint2* __restrict__ ranges, 
//     const uint32_t* __restrict__ shape_list, 
//     const uint32_t* __restrict__ shape_indices, 
//     uint8_t* __restrict__ frame_buffer
// )
// {
//     auto block = cg::this_thread_block();
// 	uint32_t horizontal_blocks = (width + BLOCK_X - 1) / BLOCK_X;
// 	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
// 	uint2 pix_max = { min(pix_min.x + BLOCK_X, width), min(pix_min.y + BLOCK_Y , height) };
// 	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
// 	uint32_t pix_id = width * pix.y + pix.x;
// 	float2 pixf = { (float)pix.x + 0.5f, (float)pix.y + 0.5f};
//     float3 color = make_float3(0, 0, 0);
//     bool inside = pixf.x < width && pixf.y < height;
// 	bool done = !inside;
// 	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
// 	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
// 	int toDo = range.y - range.x;
// 	__shared__ float2 collected_xy[N_VERTICES * BLOCK_SIZE];
// 	__shared__ float3 collected_color[N_VERTICES * BLOCK_SIZE];
//     // printf("pixf=(%f, %f), range=(%d, %d)\n", pixf.x, pixf.y, range.x, range.y);
// 	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
//     {
// 		int num_done = __syncthreads_count(done);
// 		if (num_done == BLOCK_SIZE)
// 			break ;
        
// 		int progress = range.x + i * BLOCK_SIZE + block.thread_rank();
//         if (progress < range.y)
// 		{
// 			int shape_idx = N_VERTICES * shape_list[progress];
//             // printf("pixf=(%f, %f), shape_idx=%d\n", pixf.x, pixf.y, shape_idx);
//             for (int k = 0; k < N_VERTICES; k++)
//             {
//                 int vertex_idx = shape_indices[shape_idx];
//                 collected_xy[N_VERTICES * block.thread_rank() + k] = make_float2(screen_vertices[2 * vertex_idx], screen_vertices[2 * vertex_idx + 1]);
//                 collected_color[N_VERTICES * block.thread_rank() + k] = make_float3(vertex_colors[3 * vertex_idx], vertex_colors[3 * vertex_idx + 1], vertex_colors[3 * vertex_idx + 2]);
//                 shape_idx++;
//             }
// 		}
// 		block.sync();

//         for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
//         {
//             float A = 0;
//             float Ap = 0;
//             float d[N_VERTICES];
//             float sd = 0;
//             for (int k = 0; k < N_VERTICES - 1; k++)
//             {
//                 const float2 vi = collected_xy[N_VERTICES * j + k];
//                 const float2 vj = collected_xy[N_VERTICES * j + k + 1];
//                 const float2 pvi = make_float2(pixf.x - vi.x, pixf.y - vi.y);
//                 const float2 pvj = make_float2(pixf.x - vj.x, pixf.y - vj.y);
//                 A += vi.x * vj.y - vi.y * vj.x;
//                 Ap += fabsf(pvi.x * pvj.y - pvi.y * pvj.x);
//                 d[k] = sqrt(pvi.x * pvi.x + pvi.y * pvi.y);
//                 sd += d[k];
//             }
//             const float2 vlast = collected_xy[N_VERTICES * j + N_VERTICES - 1];
//             const float2 vfirst = collected_xy[N_VERTICES * j];
//             float2 pvlast = make_float2(pixf.x - vlast.x, pixf.y - vlast.y);
//             float2 pvfirst = make_float2(pixf.x - vfirst.x, pixf.y - vfirst.y);
//             A = fabsf(A + vlast.x * vfirst.y - vlast.y * vfirst.x);
//             Ap = Ap + fabsf(pvlast.x * pvfirst.y - pvlast.y * pvfirst.x);
//             d[N_VERTICES - 1] = sqrt(pvlast.x * pvlast.x + pvlast.y * pvlast.y);
//             sd += d[N_VERTICES - 1];
//             // if (fabsf(A - Ap) < EPSILON)
//             if (fabsf(A - Ap) < 0.01)
//             {
//                 // for (int k = 0; k < N_VERTICES; k++)
//                 // {
//                 //     float w = d[k] / sd;
//                 //     float3 c = collected_color[N_VERTICES * j + k];
//                 //     color.x += w * c.x;
//                 //     color.y += w * c.y;
//                 //     color.z += w * c.z;
//                 // }
//                 float3 c = collected_color[N_VERTICES * j];
//                 color.x = c.x;
//                 color.y = c.y;
//                 color.z = c.z;
//                 done = true;
                
//                 // if (color.x > EPSILON || color.y > EPSILON || color.z > EPSILON)
//                 //     printf("pix = (%f, %f), color = (%f, %f, %f)\n", pixf.x, pixf.y, color.x, color.y, color.z);

//                 frame_buffer[3 * pix_id] = static_cast<uint8_t>(255 * color.x);
//                 frame_buffer[3 * pix_id + 1] = static_cast<uint8_t>(255 * color.y);
//                 frame_buffer[3 * pix_id + 2] = static_cast<uint8_t>(255 * color.z);
//             }
//         }
//     }
// }

__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y) fragment_shader_kernel(
    const uint32_t height, 
    const uint32_t width, 
    const float* __restrict__ screen_vertices, 
    const float* __restrict__ vertex_colors, 
    const uint2* __restrict__ ranges, 
    const uint32_t* __restrict__ shape_list, 
    const uint32_t* __restrict__ shape_indices, 
    uint8_t* __restrict__ frame_buffer
)
{
    auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (width + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, width), min(pix_min.y + BLOCK_Y , height) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = width * pix.y + pix.x;
	float2 pixf = { (float)pix.x + 0.5f, (float)pix.y + 0.5f};
    float3 color = make_float3(0, 0, 0);
    bool inside = pixf.x < width && pixf.y < height;
	bool done = !inside;
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;
	__shared__ float2 collected_xy[3 * BLOCK_SIZE];
	__shared__ float3 collected_color[3 * BLOCK_SIZE];
    // printf("pixf=(%f, %f), range=(%d, %d)\n", pixf.x, pixf.y, range.x, range.y);
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
    {
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break ;
        
		int progress = range.x + i * BLOCK_SIZE + block.thread_rank();
        if (progress < range.y)
		{
			int shape_idx = 3 * shape_list[progress];
            // printf("pixf=(%f, %f), shape_idx=%d\n", pixf.x, pixf.y, shape_idx);
            for (int k = 0; k < 3; k++)
            {
                int vertex_idx = shape_indices[shape_idx];
                collected_xy[3 * block.thread_rank() + k] = make_float2(screen_vertices[2 * vertex_idx], screen_vertices[2 * vertex_idx + 1]);
                collected_color[3 * block.thread_rank() + k] = make_float3(vertex_colors[3 * vertex_idx], vertex_colors[3 * vertex_idx + 1], vertex_colors[3 * vertex_idx + 2]);
                shape_idx++;
            }
		}
		block.sync();

        for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
        {
            float2 A = collected_xy[3 * j];
            float2 B = collected_xy[3 * j + 1];
            float2 C = collected_xy[3 * j + 2];
            float2 AP = make_float2(pixf.x - A.x, pixf.y - A.y);
            float2 AB = make_float2(B.x - A.x, B.y - A.y);
            float2 AC = make_float2(C.x - A.x, C.y - A.y);
            float u = (AP.x * AB.y - AB.x * AP.y) / (AC.x * AB.y - AB.x * AC.y);
            float v = (AP.x * AC.y - AC.x * AP.y) / (AB.x * AC.y - AC.x * AB.y);
            if (u >= 0 && v >= 0 && u + v <= 1)
            {
                float2 BP = make_float2(pixf.x - B.x, pixf.y - B.y);
                float2 CP = make_float2(pixf.x - C.x, pixf.y - C.y);
                float _2S = fabsf(AB.x * AC.y - AC.x * AB.y);
                float alpha = fabsf(BP.x * CP.y - CP.x * BP.y) / _2S;
                float beta = fabsf(AP.x * CP.y - CP.x * AP.y) / _2S;
                float gamma = 1 - alpha - beta;
                
                float3 Ac = collected_color[3 * j];
                float3 Bc = collected_color[3 * j + 1];
                float3 Cc = collected_color[3 * j + 2];

                color.x = alpha * Ac.x + beta * Bc.x + gamma * Cc.x;
                color.y = alpha * Ac.x + beta * Bc.y + gamma * Cc.y;
                color.z = alpha * Ac.z + beta * Bc.z + gamma * Cc.z;
                done = true;
                
                frame_buffer[3 * pix_id] = static_cast<uint8_t>(255 * color.x);
                frame_buffer[3 * pix_id + 1] = static_cast<uint8_t>(255 * color.y);
                frame_buffer[3 * pix_id + 2] = static_cast<uint8_t>(255 * color.z);
            }
        }
    }
}



__global__ void check_touched_tiles(uint32_t S, uint32_t* touched_tiles)
{
    const uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= S)
    {
        return ;
    }
    printf("touched_tiles[%u]=%u\n", idx, touched_tiles[idx]);
}
__global__ void check_offsets(uint32_t S, uint32_t* offsets)
{
    const uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= S)
    {
        return ;
    }
    printf("offsets[%u]=%u\n", idx, offsets[idx]);
}

__global__ void check_(uint32_t S, uint32_t* touched_tiles)
{
    const uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= S)
    {
        return ;
    }
    printf("touched_tiles[%u]=%u\n", idx, touched_tiles[idx]);
}

__global__ void check_sorted_value_list(uint32_t R, uint32_t* sorted_value_list)
{
    const uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= R)
    {
        return ;
    }
    printf("sorted_value_list[%u]=%u\n", idx, sorted_value_list[idx]);
}

__global__ void check_unsorted_value_list(uint32_t R, uint32_t* unsorted_value_list)
{
    const uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= R)
    {
        return ;
    }
    printf("unsorted_value_list[%u]=%u\n", idx, unsorted_value_list[idx]);
}

void Rasterizer::bind(uint32_t light_num, float3* lights_pos, float3* lights_emit, uint32_t vertex_num, float3* vertices, float3* normals, uint32_t shape_num, uint3* shape_indices, bool* mask)
{
    if (binded)
        _materials->free();
    else
    {
        _materials = new BlinnPhong(vertex_num, make_float3(0.5f, 0.5f, 0.5f), 0.1f);
        _materials->generate();
        binded = true;
    }
    _light_num = light_num;
    _lights_pos = lights_pos;
    _lights_emit = lights_emit;
    _vertex_num = vertex_num;
    _vertices = vertices;
    _normals = normals;
    _shape_num = shape_num;
    _shape_indices = shape_indices;
    _mask = mask;
}

void Rasterizer::render(uint32_t height, uint32_t width, float* view_matrix, float* view_persp_matrix, float* vp_matrix, cudaGraphicsResource* cuda_pbo_resource)
{
    uint32_t S = _shape_num;
    uint32_t V = _vertex_num;
    uint32_t L = _light_num;
    dim3 grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
    dim3 block(BLOCK_X, BLOCK_Y, 1);
    float* cam_vertices;
    float* clip_vertices;
    float* screen_vertices;
    float* depths;
    float* d_view_matrix;
    float* d_view_persp_matrix;
    float* d_vp_matrix;
    uint32_t* touched_tiles;
    uint32_t* offsets;
    uint32_t* rect_mins;
    uint32_t* rect_maxs;
    float* vertex_colors;
    uint32_t pixel_num = height * width;
    uint8_t* frame_buffer;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&cam_vertices, sizeof(float3) * V));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&clip_vertices, sizeof(float3) * V));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&screen_vertices,sizeof(float2) * V));

    CHECK_CUDA_ERROR(cudaMalloc((void**)&depths, sizeof(float) * S));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&touched_tiles, sizeof(uint32_t) * S));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&offsets, sizeof(uint32_t) * S));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&rect_mins, sizeof(uint32_t) * S * 2));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&rect_maxs, sizeof(uint32_t) * S * 2));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&vertex_colors, sizeof(float) * V * 3));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&frame_buffer, sizeof(uint8_t) * pixel_num * 3));

    CHECK_CUDA_ERROR(cudaMemset((void*)depths, 0, sizeof(float) * S));
    CHECK_CUDA_ERROR(cudaMemset((void*)touched_tiles, 0, sizeof(uint32_t) * S));
    CHECK_CUDA_ERROR(cudaMemset((void*)offsets, 0, sizeof(uint32_t) * S));
    CHECK_CUDA_ERROR(cudaMemset((void*)rect_mins, 0, sizeof(uint32_t) * S * 2));
    CHECK_CUDA_ERROR(cudaMemset((void*)rect_maxs, 0, sizeof(uint32_t) * S * 2));
    CHECK_CUDA_ERROR(cudaMemset((void*)vertex_colors, 0, sizeof(float) * V * 3));
    CHECK_CUDA_ERROR(cudaMemset((void*)frame_buffer, 0, sizeof(uint8_t) * pixel_num * 3));

    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_view_matrix, sizeof(float) * 16));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_view_persp_matrix, sizeof(float) * 16));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_vp_matrix, sizeof(float) * 16));
    CHECK_CUDA_ERROR(cudaMemcpy((void*)d_view_matrix, (void*)view_matrix, sizeof(float) * 16, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy((void*)d_view_persp_matrix, (void*)view_persp_matrix, sizeof(float) * 16, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy((void*)d_vp_matrix, (void*)vp_matrix, sizeof(float) * 16, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    view_transform_kernel<<<(V + VERTICE_PER_BLOCK - 1) / VERTICE_PER_BLOCK, VERTICE_PER_BLOCK>>>
    (
        V, 
        (const float*)_vertices, 
        d_view_matrix, 
        cam_vertices
    );
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    view_persp_transform_kernel<<<(V + VERTICE_PER_BLOCK - 1) / VERTICE_PER_BLOCK, VERTICE_PER_BLOCK>>>
    (
        V, 
        (const float*)_vertices, 
        d_view_persp_matrix, 
        clip_vertices
    );
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    vp_transform_kernel<<<(V + VERTICE_PER_BLOCK - 1) / VERTICE_PER_BLOCK, VERTICE_PER_BLOCK>>>
    (
        V, 
        (const float*)clip_vertices, 
        d_vp_matrix, 
        screen_vertices
    );
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // Preprocess (compute depths, match tiles, and get rectangles)
    preprocess_kernel<<<(S + TRIANGLE_PER_BLOCK - 1) / TRIANGLE_PER_BLOCK, TRIANGLE_PER_BLOCK>>>
    (
        S, 
        cam_vertices, 
        clip_vertices, 
        screen_vertices, 
        (const uint32_t*)_shape_indices, 
        (const bool*)_mask, 
        width, 
        height, 
        grid, 
        depths, 
        touched_tiles, 
        rect_mins, 
        rect_maxs
    );
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Inclusive Sum
    void* scan_buf = nullptr;
    uint64_t scan_buf_bytes = 0;
    CHECK_CUDA_ERROR(cub::DeviceScan::InclusiveSum(scan_buf, scan_buf_bytes, touched_tiles, offsets, S));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR(cudaMalloc((void**)&scan_buf, scan_buf_bytes));
    CHECK_CUDA_ERROR(cudaMemset((void*)scan_buf, 0, scan_buf_bytes));
    CHECK_CUDA_ERROR(cub::DeviceScan::InclusiveSum(scan_buf, scan_buf_bytes, touched_tiles, offsets, S));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // check_touched_tiles<<<(S + TRIANGLE_PER_BLOCK - 1) / TRIANGLE_PER_BLOCK, TRIANGLE_PER_BLOCK>>>(S, touched_tiles);
    // CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    // check_offsets<<<(S + TRIANGLE_PER_BLOCK - 1) / TRIANGLE_PER_BLOCK, TRIANGLE_PER_BLOCK>>>(S, offsets);
    // CHECK_CUDA_ERROR(cudaDeviceSynchronize());


    uint32_t R;
    CHECK_CUDA_ERROR(cudaMemcpy(&R, offsets + S - 1, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    uint64_t* unsorted_key_list;
    uint32_t* unsorted_value_list;
    uint64_t* sorted_key_list;
    uint32_t* sorted_value_list;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&unsorted_key_list, R * sizeof(uint64_t)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&unsorted_value_list, R * sizeof(uint32_t)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&sorted_key_list, R * sizeof(uint64_t)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&sorted_value_list, R * sizeof(uint32_t)));

    CHECK_CUDA_ERROR(cudaMemset((void*)unsorted_key_list, 0, R * sizeof(uint64_t)));
    CHECK_CUDA_ERROR(cudaMemset((void*)unsorted_value_list, 0, R * sizeof(uint32_t)));
    CHECK_CUDA_ERROR(cudaMemset((void*)sorted_key_list, 0, R * sizeof(uint64_t)));
    CHECK_CUDA_ERROR(cudaMemset((void*)sorted_value_list, 0, R * sizeof(uint32_t)));

    // Replication
    replication_kernel<<<(S + TRIANGLE_PER_BLOCK) / TRIANGLE_PER_BLOCK, TRIANGLE_PER_BLOCK>>>
    (
        S, 
        depths, 
        grid, 
        rect_mins, 
        rect_maxs, 
        touched_tiles, 
        offsets, 
        unsorted_key_list, 
        unsorted_value_list
    );
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // check_unsorted_value_list<<<(R + TRIANGLE_PER_BLOCK - 1) / TRIANGLE_PER_BLOCK, TRIANGLE_PER_BLOCK>>>(R, unsorted_value_list);
    // CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Sort
    void* sort_buf = nullptr;
    uint32_t tiles_num = grid.x * grid.y;
    uint64_t sort_buf_bytes = 0;
    cub::DeviceRadixSort::SortPairs(sort_buf, sort_buf_bytes, unsorted_key_list, sorted_key_list, unsorted_value_list, sorted_value_list, R, 0, 32 + getHigherMsb(tiles_num));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR(cudaMalloc((void**)&sort_buf, sort_buf_bytes));
    CHECK_CUDA_ERROR(cudaMemset((void*)sort_buf, 0, sort_buf_bytes));
    cub::DeviceRadixSort::SortPairs(sort_buf, sort_buf_bytes, unsorted_key_list, sorted_key_list, unsorted_value_list, sorted_value_list, R, 0, 32 + getHigherMsb(tiles_num));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // check_unsorted_value_list<<<(R + TRIANGLE_PER_BLOCK - 1) / TRIANGLE_PER_BLOCK, TRIANGLE_PER_BLOCK>>>(R, unsorted_value_list);
    // CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    // check_sorted_value_list<<<(R + TRIANGLE_PER_BLOCK - 1) / TRIANGLE_PER_BLOCK, TRIANGLE_PER_BLOCK>>>(R, sorted_value_list);
    // CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // printf("V=%d, R=%d, S=%d, blocks=%d\n", V, R, S, grid.x * grid.y);
    // Identify tile range
    uint2* ranges;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&ranges, sizeof(uint2) * grid.x * grid.y));
    CHECK_CUDA_ERROR(cudaMemset((void*)ranges, 0, sizeof(uint2) * grid.x * grid.y));
    identify_tile_ranges_kernel<<<(R + TRIANGLE_PER_BLOCK - 1) / TRIANGLE_PER_BLOCK, TRIANGLE_PER_BLOCK>>>(R, sorted_key_list, ranges);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Vertex shader
    vertex_shader_kernel<<<(V + VERTICE_PER_BLOCK - 1) / VERTICE_PER_BLOCK, VERTICE_PER_BLOCK>>>(
        V, 
        L, 
        (const float*)_lights_pos, 
        (const float*)_lights_emit, 
        (const float*)_vertices, 
        (const float*)_normals, 
        (const float*)_materials->kds, 
        (const float*)_materials->kss, 
        (const float*)_materials->kas,
        _materials->ambient, 
        _materials->shininess, 
        vertex_colors
        );
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    printf("vertex shader complete.\n");

    // Fragment shader
    fragment_shader_kernel<<<grid, block>>>(
        height, 
        width, 
        (const float*)screen_vertices, 
        (const float*)vertex_colors, 
        (const uint2*)ranges, 
        (const uint32_t*)sorted_value_list, 
        (const uint32_t*)_shape_indices, 
        frame_buffer
    );
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    printf("fragment shader complete.\n");

    // Write frame buffer to pbo
    CHECK_CUDA_ERROR(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
    size_t num_bytes = sizeof(uchar3) * pixel_num;
    uchar3* cuda_pbo_pointer;
    CHECK_CUDA_ERROR(cudaGraphicsResourceGetMappedPointer((void**)&cuda_pbo_pointer, &num_bytes, cuda_pbo_resource));
    CHECK_CUDA_ERROR(cudaMemcpy((void*)cuda_pbo_pointer, (void*)frame_buffer, num_bytes, cudaMemcpyDeviceToDevice));
    CHECK_CUDA_ERROR(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));
    CHECK_CUDA_ERROR(cudaFree(cam_vertices));
    CHECK_CUDA_ERROR(cudaFree(clip_vertices));
    CHECK_CUDA_ERROR(cudaFree(screen_vertices));
    CHECK_CUDA_ERROR(cudaFree(depths));
    CHECK_CUDA_ERROR(cudaFree(touched_tiles));
    CHECK_CUDA_ERROR(cudaFree(offsets));
    CHECK_CUDA_ERROR(cudaFree(rect_mins));
    CHECK_CUDA_ERROR(cudaFree(rect_maxs));
    CHECK_CUDA_ERROR(cudaFree(vertex_colors));
    CHECK_CUDA_ERROR(cudaFree(frame_buffer));
    CHECK_CUDA_ERROR(cudaFree(ranges));
    CHECK_CUDA_ERROR(cudaFree(sort_buf));
    CHECK_CUDA_ERROR(cudaFree(unsorted_key_list));
    CHECK_CUDA_ERROR(cudaFree(unsorted_value_list));
    CHECK_CUDA_ERROR(cudaFree(sorted_key_list));
    CHECK_CUDA_ERROR(cudaFree(sorted_value_list));
    CHECK_CUDA_ERROR(cudaFree(scan_buf));
}

void Rasterizer::save_frame_buffer()
{
    
}


void Rasterizer::free()
{
    if (binded)
        _materials->free();
    binded = false;
}

__global__ void generate_blinn_phong_kernel(
	uint32_t size, 
	float3* kas, 
	float3* kds,
	float3* kss, 
	float3 ks_min_bound={0.4f, 0.4f, 0.4f}, 
	float3 ks_max_bound={0.8f, 0.8f, 0.8f}, 
	float3 kd_min_bound={0.8f, 0.6f, 0.4f},
	float3 kd_max_bound={0.85f, 0.7f, 0.45f}, 
	float3 ka_min_bound={0.4f, 0.4f, 0.4f}, 
	float3 ka_max_bound={0.6f, 0.6f, 0.6f}
)
{
	auto idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < size)
	{
		curandState state;
		curand_init(clock() + idx, 0, 0, &state);
		kss[idx] = rand_intep_float3(ks_min_bound, ks_max_bound, &state);
		kas[idx] = rand_intep_float3(ka_min_bound, ka_max_bound, &state);
		kds[idx] = rand_intep_float3(kd_min_bound, kd_max_bound, &state);
	}
}
void BlinnPhong::generate()
{
	generate_blinn_phong_kernel<<<(size + VERTICE_PER_BLOCK - 1) / VERTICE_PER_BLOCK, VERTICE_PER_BLOCK>>>(size, kas, kds, kss);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

void BlinnPhong::free()
{
	CHECK_CUDA_ERROR(cudaFree(kas));
	CHECK_CUDA_ERROR(cudaFree(kds));
	CHECK_CUDA_ERROR(cudaFree(kss));
}