# CMakeList file for component.

idf_component_register(SRCS "BriandFCNN.cpp" "BriandSimpleNN.cpp" "BriandMatrix.cpp" "BriandCNN.cpp" "BriandImage.cpp" "BriandMath.cpp" "BriandMatrix.cpp" "BriandPorting.cpp"
                    INCLUDE_DIRS "include"
                    REQUIRES esp_timer)
