cmake_minimum_required(VERSION 2.8.12)

PROJECT(DocumentScanner)
SET(CMAKE_CXX_STANDARD 17)



######################## EDIT IF REQUIRED ####################
set(OpenCV_DIR /opt/opencv/4.4.0/installation/lib/cmake/opencv4)
# ###Uncomment the line below and specify the path to OpenCV directory i.e. the path to the OpenCVConfig.cmake file. Check the examples given below.
#SET(OpenCV_DIR Enter-the-path-of-OpenCV-installation-on-your-system)
#SET(OpenCV_DIR /home/hp/workfolder/OpenCV-Installation/installation/OpenCV-master/lib/cmake/opencv4)



if(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
endif()


find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(doscan 
	main.cpp 
	abstractquaddetector.h 
	abstractquaddetector.cpp 
	abstractpapersheetdetector.h 
	abstractpapersheetdetector.cpp 
	ithreshpapersheetdetector.h
	ithreshpapersheetdetector.cpp
	savaldopapersheetdetector.h
	savaldopapersheetdetector.cpp
	documentscanner.h
	documentscanner.cpp
)

target_link_libraries(doscan ${OpenCV_LIBS} )
  
if(MSVC)
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
  set_property(TARGET doscan PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
endif()
