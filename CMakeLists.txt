cmake_minimum_required(VERSION 2.8)

set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

project(neural_network_test)

find_package( OpenCV REQUIRED )

FIND_PACKAGE( Boost 1.66 COMPONENTS program_options REQUIRED )


find_package(PkgConfig)

INCLUDE(FindPkgConfig)



pkg_check_modules(GTKMM gtkmm-2.4)



link_directories(
    ${GTKMM_LIBRARY_DIRS}  )

include_directories(header ${GTKMM_INCLUDE_DIRS})

INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} )

add_executable(
	neural_network_test
	source/main.cpp
	source/nnbp.cpp
	source/acfunc.cpp
	source/cu.cpp
	source/mytraining.cpp
	
)

target_link_libraries( neural_network_test ${OpenCV_LIBS} ${GTKMM_LIBRARIES}  ${Boost_LIBRARIES}  -O2 -larmadillo -llapack -lblas -lpthread)
