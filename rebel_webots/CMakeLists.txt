cmake_minimum_required(VERSION 3.5)
project(rebel_webots)

find_package(ament_cmake REQUIRED)
find_package(rebel_description REQUIRED)

# Install files
install(DIRECTORY launch
                  worlds
                  proto
                  urdf
                  config
                  controllers
        DESTINATION share/${PROJECT_NAME}
)

# Install driver python package
ament_python_install_package(${PROJECT_NAME})


# Create a symbolic link to rebel_description meshes
execute_process(
    COMMAND ${CMAKE_COMMAND} -E create_symlink
    ${rebel_description_DIR}/../meshes
    ${CMAKE_INSTALL_PREFIX}/share/${PROJECT_NAME}/meshes
)

ament_package()

