#pragma once
#include "opencv2/core.hpp"
#include <string>

namespace cv {
namespace hed {

class CV_EXPORTS_W HEDDetector {
public:
    CV_WRAP static cv::Ptr<HEDDetector> create(
        const std::string& modelPath,
        const std::string& protoPath
    );
    CV_WRAP virtual cv::Mat detectEdges(cv::InputArray image) = 0;
};

} // hed
} // cv