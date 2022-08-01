cmake_minimum_required(VERSION 3.16)
project(reactiondiffusion)

list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Find Polyscope
find_package(Polyscope REQUIRED)

# Libigl
option(LIBIGL_WITH_OPENGL            "Use OpenGL"         ON)
option(LIBIGL_WITH_OPENGL_GLFW       "Use GLFW"           ON)
option(LIBIGL_WITH_TRIANGLE          "Use Triangle"       ON)
include(libigl)


# Add your project files
file(GLOB SRC_FILES src/*.cpp)
add_executable(${PROJECT_NAME} ${SRC_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC igl::core igl::opengl_glfw igl::triangle polyscope)
