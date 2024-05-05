cmake_minimum_required(VERSION 3.14.0)
project(CudaParticleSystem VERSION 0.1.0 LANGUAGES C CXX CUDA)

include(CTest)
enable_testing()

find_package ( OpenGL REQUIRED )
include_directories(
    "include"
    "include/fields"
    "ext"
    "ext/imgui"
    "ext/imgui/backends"
    "ext/glad/include"
    "ext/glfw-3.3.9.bin.WIN64/include"
    )
include_directories(${CUDA_INCLUDE_DIRS})

set ( CPACK_PROJECT_NAME ${PROJECT_NAME} )
set ( CPACK_PROJECT_VERSION ${PROJECT_VERSION} )
set ( GLFW_HOME "ext/glfw-3.3.9.bin.WIN64" )
link_directories( "${GLFW_HOME}/lib-vc2022" )
set ( GLAD_HOME "ext/glad" )
set ( IMGUI_HOME "ext/imgui" )
set ( GLFW_INC "${GLFW_HOME}/include" )
set ( GLFW_LIB "glfw3" )

set ( GLAD_INC "${GLAD_HOME}/include" )
set ( GLAD_SRC "${GLAD_HOME}/src" )

file ( GLOB glad_src_files "${GLAD_SRC}/glad.c" )
file ( GLOB glad_header_files "${GLAD_INC}/glad/glad.h" "${GLAD_INC}/KHR/khrplatform.h" )
file ( GLOB imgui_src_files "${IMGUI_HOME}/imgui.cpp" 
                            "${IMGUI_HOME}/imgui_draw.cpp"
                            "${IMGUI_HOME}/imgui_tables.cpp"
                            "${IMGUI_HOME}/imgui_widgets.cpp"
                            "${IMGUI_HOME}/backends/imgui_impl_glfw.cpp"
                            "${IMGUI_HOME}/backends/imgui_impl_opengl3.cpp"
                            )
add_executable(CudaParticleSystem 
    src/main.cu
    src/particle_system_kernel.cu
    src/rasterizer_kernel.cu
    ${glad_header_files}
    ${glad_src_files}
    ${imgui_src_files}
)

set_target_properties(CudaParticleSystem PROPERTIES CUDA_ARCHITECTURES "86;75;70;61;60")

target_link_libraries(CudaParticleSystem ${CUDA_LIBRARIES} ${GLFW_LIB} ${OPENGL_LIBRARIES})
target_compile_features(CudaParticleSystem PRIVATE cxx_std_17)
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
