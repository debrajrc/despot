cmake_minimum_required(VERSION 2.8.3)

add_executable("${PROJECT_NAME}_drn"
  src/drn.cpp
  src/main.cpp
        src/drn.h
)
target_link_libraries("${PROJECT_NAME}_drn"
  "${PROJECT_NAME}"
)
install(TARGETS "${PROJECT_NAME}_drn"
  RUNTIME DESTINATION "${BINARY_INSTALL_PATH}"
)