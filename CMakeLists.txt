cmake_minimum_required(VERSION 3.20)
project(tracking_demo LANGUAGES CXX)

# C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_VERBOSE_MAKEFILE ON)

# Required packages
find_package(OpenCV REQUIRED)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

file(GLOB SOURCES "${CMAKE_SOURCE_DIR}/src/*.cpp")

# Executable
add_executable(tracking_demo ${SOURCES})

# Link OpenCV only — no need to link nlohmann_json
target_link_libraries(tracking_demo
    PRIVATE
    ${OpenCV_LIBS}
)

# MSVC debug info policy
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()
