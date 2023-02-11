# CMakeList file for component.

idf_component_register(SRCS "BriandMatrix.cpp" "BriandCNN.cpp" "BriandImage.cpp" "BriandMath.cpp" "BriandMatrix.cpp" "BriandNN.cpp" "BriandPorting.cpp"
                    INCLUDE_DIRS "include"
                    REQUIRES esp_timer)
