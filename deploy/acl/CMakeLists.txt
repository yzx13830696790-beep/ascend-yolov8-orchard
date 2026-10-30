cmake_minimum_required(VERSION 3.10)
project(orchard_acl CXX)

set(CMAKE_CXX_STANDARD 11)

# CANN 环境变量（安装昇腾驱动/工具包后自动设置）
# 若无，请 source /usr/local/Ascend/ascend-toolkit/set_env.sh
if(NOT DEFINED ENV{ASCEND_HOME_PATH})
  message(FATAL_ERROR "ASCEND_HOME_PATH not set, source ascend-toolkit set_env.sh first")
endif()

set(ASCEND $ENV{ASCEND_HOME_PATH})
include_directories(${ASCEND}/runtime/include ${ASCEND}/compiler/op_host ../postprocess)

add_executable(orchard_acl
  main.cpp
  ../postprocess/nms.cpp
)

target_link_libraries(orchard_acl
  ${ASCEND}/runtime/lib64/stub/libascendcl.so
  ${ASCEND}/runtime/lib64/stub/libruntime.so
)
