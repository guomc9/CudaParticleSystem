#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <cuda_gl_interop.h>
#include <fstream>
#include <sstream>
#include <chrono>

#include "json.hpp"
#include "particle_system.hpp"
#include "rasterizer.hpp"
#include "gui.hpp"
#include "importer.hpp"
#include "scene.hpp"


using json = nlohmann::json;
struct Conf {
    float3 lookat;
    float3 up;
    float3 eye_pos;
    float ar;
    float fov_y;
    float far;
    float near;
    unsigned int width;
    unsigned int height;

};
Conf cfg;

#ifdef _MSC_VER
std::string config_path("../../conf/config.json");
#else
std::string config_path("../conf/config.json");
#endif

bool config_sys()
{
    std::ifstream json_file(config_path);
    std::stringstream json_data;
    json_data << json_file.rdbuf();
    json conf_json = json::parse(json_data);

    cfg.eye_pos = make_float3(conf_json["eye_pos"]["x"], conf_json["eye_pos"]["y"], conf_json["eye_pos"]["z"]);
    cfg.lookat = make_float3(conf_json["lookat"]["x"], conf_json["lookat"]["y"], conf_json["lookat"]["z"]);
    cfg.up = make_float3(conf_json["up"]["x"], conf_json["up"]["y"], conf_json["up"]["z"]);
    cfg.fov_y = conf_json["fov_y"];
    cfg.width = conf_json["width"];
    cfg.height = conf_json["height"];
    cfg.ar = conf_json["ar"];
    cfg.near = conf_json["near"];
    cfg.far = conf_json["far"];
    return true;
}

bool config_CUDA()
{
    cudaDeviceProp deviceProp;
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if(deviceCount > 0)
    {
        cudaGetDeviceProperties(&deviceProp, 0);
        cudaSetDevice(0);
        return true;
    }
    printf("Failed to find CUDA device\n");
    return false;
}

void printMatrix4x4(const float* matrix)
{
    for (int i = 0; i < 16; i++)
    {
        std::cout << std::setw(10) << std::fixed << std::setprecision(2) << matrix[i];
        if ((i + 1) % 4 == 0)
        {
            std::cout << std::endl;
        }
        else
        {
            std::cout << " ";
        }
    }
}

int main(void)
{
    // config_CUDA();
    config_sys();
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(800, 600, "CudaParticleSystem", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return ;
    }
    glfwMakeContextCurrent(window);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return ;
    }
    printf("width:%d, height=%d\n", cfg.width, cfg.height);
    glViewport(0, 300, 800, 600);
    GLuint pbo;
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, cfg.width * cfg.height * 3, 0, GL_STREAM_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    GLuint textureID;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);
    
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, cfg.width, cfg.height, 0, GL_RGB, GL_UNSIGNED_BYTE, 0);
    glGenerateMipmap(GL_TEXTURE_2D);

    glBindTexture(GL_TEXTURE_2D, 0);


    float vertices[] = {
        // positions        // texture coords
        -1.0f, -1.0f, 0.0f, 0.0f, 0.0f, // bottom left
         1.0f, -1.0f, 0.0f, 1.0f, 0.0f, // bottom right
         1.0f,  1.0f, 0.0f, 1.0f, 1.0f, // top right

        -1.0f, -1.0f, 0.0f, 0.0f, 0.0f, // bottom left
         1.0f,  1.0f, 0.0f, 1.0f, 1.0f, // top right
        -1.0f,  1.0f, 0.0f, 0.0f, 1.0f  // top left
    };

    GLuint VAO, VBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    
    // texture coord attribute
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    const char *vertexShaderSource = R"glsl(
        #version 330 core
        layout (location = 0) in vec3 aPos;
        layout (location = 1) in vec2 aTexCoord;

        out vec2 TexCoord;

        void main()
        {
            gl_Position = vec4(aPos, 1.0);
            TexCoord = vec2(aTexCoord.x, 1.0 - aTexCoord.y);
        })glsl";

    const char *fragmentShaderSource = R"glsl(
        #version 330 core
        out vec4 FragColor;

        in vec2 TexCoord;

        uniform sampler2D texture1;

        void main()
        {
            FragColor = texture(texture1, TexCoord);
        })glsl";

    GLint vs_id = glCreateShader(GL_VERTEX_SHADER);
    GLint fs_id = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(vs_id, 1, &vertexShaderSource, nullptr);
    glShaderSource(fs_id, 1, &fragmentShaderSource, nullptr);
    
    GLint prog_id = glCreateProgram();
    
    glCompileShader(vs_id);
    glCompileShader(fs_id);
    
    glAttachShader(prog_id, vs_id);
    glAttachShader(prog_id, fs_id);
    
    glLinkProgram(prog_id);
    glValidateProgram(prog_id);

    glDeleteShader(vs_id);
    glDeleteShader(fs_id);

    glUseProgram(prog_id);

    cudaGraphicsResource* cuda_pbo_resource;
    cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard);
    Rasterizer renderer;
    
    // MeshImporter inp;
    // inp.load("../../test/view_test.obj");
    // Scene s(inp.get_vertices_size(), inp.get_vertices(), inp.get_normals(), inp.get_shape_size(), inp.get_indices());

    ParticleSystem ps(5000, 100, 200, 0.1f, 0.2f, 1.0f, 2.0f, make_float3(-10, -10, 0), make_float3(10, 10, 10), make_float3(-2.0f, -2.0f, -2.0f), make_float3(2.0f, 2.0f, 2.0f), 10, 10);
    ps.init();
    ps.meshify();
    uint32_t lights_num = 3;
    float3 lights_pos[] = { make_float3(0, 0, 20),   make_float3(5, 5, 20), make_float3(-5, -5, 20)};
    float3 lights_emit[] = {    make_float3(1.0, 1.0, 1.0), make_float3(1.0, 1.0, 1.0), make_float3(1.0, 1.0, 1.0)  };
    float3* d_lights_pos;
    float3* d_lights_emit;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_lights_pos, sizeof(float3) * lights_num));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_lights_emit, sizeof(float3) * lights_num));
    CHECK_CUDA_ERROR(cudaMemcpy((void*)d_lights_pos, lights_pos, sizeof(float3) * lights_num, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy((void*)d_lights_emit, lights_emit, sizeof(float3) * lights_num, cudaMemcpyHostToDevice));
    // printf("vertice num=%llu, shape num=%llu", s.get_vertices_size(), s.get_shape_size());
    renderer.bind(lights_num, d_lights_pos, d_lights_emit, ps.get_vertex_num(), ps.get_vertices(), ps.get_normals(), ps.get_shape_num(), ps.get_shape_indices(), ps.get_mask());
    // renderer.bind(lights_num, d_lights_pos, d_lights_emit, (uint32_t)s.get_vertices_size(), s.get_vertices(), s.get_normals(), (uint32_t)s.get_shape_size(), s.get_indices());
    Gui gui(cfg.height, cfg.width, cfg.eye_pos, cfg.lookat, cfg.up);
    bool rendered = false;
    bool stopped = true;
    auto last_frame = std::chrono::high_resolution_clock::now();
    glUseProgram(prog_id);
    while(!glfwWindowShouldClose(window))
    {
        glfwPollEvents();
        gui.update(window);
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::Begin("CudaParticleSystem");
        if (ImGui::SliderFloat("delta time", gui.delta_time(), 0, 1)){}
        if (ImGui::SliderInt("fps", gui.fps(), 1, 100)){}
        if (ImGui::SliderFloat("move speed", gui.move_speed(), 0, 1)){}
        if (ImGui::SliderFloat("rot speed", gui.rot_speed(), 0, 1)){}
        if (ImGui::SliderInt("simulation steps", gui.simulation_steps(), 0, 1000)){}
        
        // if (ImGui::Button("init"))
        // {
        //     stopped = true;
        //     ps.init();
        //     ps.meshify();
        //     renderer.bind(lights_num, d_lights_pos, d_lights_emit, ps.get_vertex_num(), ps.get_vertices(), ps.get_normals(), ps.get_shape_num(), ps.get_shape_indices(), ps.get_mask());
        // }
        ImGui::Text("simulation progress: %d / %d", gui.get_current_step(), gui.get_simulation_steps());
        ImGui::ProgressBar(1.0f * gui.get_current_step() / gui.get_simulation_steps());
        
        if (ImGui::Button("Run"))
        {
            stopped = false;
            rendered = true;
        }
        if (ImGui::Button("stop"))
        {
            stopped = true;
        }
        ImGui::End();
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        if (!stopped)
        {
            auto begin = std::chrono::high_resolution_clock::now();
            std::chrono::duration<float> delta = begin - last_frame;
            if (delta.count() > 1.0f / gui.get_fps())
            {
                if (gui.get_simulation_steps() > gui.get_current_step())
                {
                    (*gui.current_step()) += 1;
                    auto view_matrix = get_view_matrix(gui.get_eye_pos_vec(), gui.get_lookat_vec(), gui.get_up_vec());
                    auto persp_matrix = get_persp_matrix(cfg.near, cfg.far, cfg.ar, cfg.fov_y * (float)M_PI / 180);
                    auto view_persp_matrix = get_view_persp_matrix(view_matrix, persp_matrix);
                    auto vp_matrix = get_vp_matrix(cfg.height, cfg.width);
                    ps.step(gui.get_delta_time());
                    ps.refresh_vertices();
                    renderer.render(cfg.height, cfg.width, view_matrix, view_persp_matrix, vp_matrix, cuda_pbo_resource);
                    auto end = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> elapsed = end - begin;
                    std::cout << "time cost: " << elapsed.count() << " seconds" << std::endl;
                }
                else
                {
                    (*gui.current_step()) = 0;
                    stopped = true;
                }
                last_frame = std::chrono::high_resolution_clock::now();
            }
        }
        else if (rendered)
        {
            auto begin = std::chrono::high_resolution_clock::now();
            std::chrono::duration<float> delta = begin - last_frame;
            if (delta.count() > 1.0f / gui.get_fps())
            {
                auto view_matrix = get_view_matrix(gui.get_eye_pos_vec(), gui.get_lookat_vec(), gui.get_up_vec());
                auto persp_matrix = get_persp_matrix(cfg.near, cfg.far, cfg.ar, cfg.fov_y * (float)M_PI / 180);
                auto view_persp_matrix = get_view_persp_matrix(view_matrix, persp_matrix);
                auto vp_matrix = get_vp_matrix(cfg.height, cfg.width);
                renderer.render(cfg.height, cfg.width, view_matrix, view_persp_matrix, vp_matrix, cuda_pbo_resource);

                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed = end - begin;
                std::cout << "time cost: " << elapsed.count() << " seconds" << std::endl;
                last_frame = std::chrono::high_resolution_clock::now();
            }
            
        }
        if (rendered)
        {
            glBindTexture(GL_TEXTURE_2D, textureID);
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, cfg.width, cfg.height, GL_RGB, GL_UNSIGNED_BYTE, 0);
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
            glBindTexture(GL_TEXTURE_2D, 0);
            glBindVertexArray(VAO);
            glBindTexture(GL_TEXTURE_2D, textureID);
            glDrawArrays(GL_TRIANGLES, 0, 6);
            glBindTexture(GL_TEXTURE_2D, 0);
        }
        glfwSwapBuffers(window);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    }
    ps.free();
    // s.free();
    renderer.free();
    CHECK_CUDA_ERROR(cudaFree(d_lights_pos));
    CHECK_CUDA_ERROR(cudaFree(d_lights_emit));

}