# CMakeList file for component.

idf_component_register(SRCS "BriandCNN.cpp" "BriandImage.cpp" "BriandMath.cpp" "BriandNN.cpp" "BriandPorting.cpp"
                    INCLUDE_DIRS "include"
                    REQUIRES esp_timer)
