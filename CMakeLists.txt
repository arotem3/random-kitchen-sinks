set(CMAKE_CXX_STANDARD 11)

find_package(OpenMP)

include_directories(include)

add_library(random_sinks SHARED
        include/random_kitchen_sinks.hpp

        src/RFF.cpp
        src/fastfood.cpp
        src/fwht.cpp
)

if (OpenMP_FOUND)
    target_link_libraries(random_sinks PRIVATE OpenMP::OpenMP_CXX)
endif()

install(TARGETS random_sinks DESTINATION lib)
install(FILES include/random_kitchen_sinks.hpp DESTINATION include)