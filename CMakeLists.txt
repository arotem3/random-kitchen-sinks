cmake_minimum_required(VERSION 3.13)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenMP)

include_directories(include)

add_library(random_sinks SHARED
        include/random_kitchen_sinks.hpp

        src/RFF.cpp
        src/fastfood.cpp
        src/sorf.cpp
        src/fwht.cpp
        src/PCA.cpp
        src/nystrom.cpp
)

if (OpenMP_FOUND)
    target_link_libraries(random_sinks PRIVATE OpenMP::OpenMP_CXX)
endif()

install(TARGETS random_sinks DESTINATION lib)
install(FILES include/random_kitchen_sinks.hpp DESTINATION include)