#include "test_precomp.hpp"

namespace opencv_test { namespace {

static bool dataFileExists(const std::string& relativePath)
{
    // 1. Search in TS data_search_path
    const std::vector<std::string>& search_path = cvtest::TS::ptr()->data_search_path;
    for (size_t i = search_path.size(); i > 0; i--)
    {
        std::string path = search_path[i - 1] + "/" + relativePath;
        FILE* f = fopen(path.c_str(), "rb");
        if (f)
        {
            fclose(f);
            return true;
        }
    }

    // 2. Search in OPENCV_TEST_DATA_PATH
    const std::vector<std::string>& search_subdir = cvtest::TS::ptr()->data_search_subdir;
    std::string datapath_dir = cv::utils::getConfigurationParameterString("OPENCV_TEST_DATA_PATH");
    if (!datapath_dir.empty())
    {
        for (size_t i = search_subdir.size(); i > 0; i--)
        {
            std::string path = datapath_dir + "/" + search_subdir[i - 1] + "/" + relativePath;
            FILE* f = fopen(path.c_str(), "rb");
            if (f)
            {
                fclose(f);
                return true;
            }
        }
    }
    return false;
}

TEST(HED, LoadModel)
{
    if (!dataFileExists("hed_pretrained_bsds.caffemodel") || !dataFileExists("deploy.prototxt"))
        throw SkipTestException("HED model files not found");

    std::string model = cvtest::findDataFile("hed_pretrained_bsds.caffemodel", true);
    std::string proto = cvtest::findDataFile("deploy.prototxt", true);

    auto detector = cv::hed::HEDDetector::create(model, proto);
    ASSERT_FALSE(detector.empty());
}

TEST(HED, OutputShape)
{
    if (!dataFileExists("hed_pretrained_bsds.caffemodel") || !dataFileExists("deploy.prototxt"))
        throw SkipTestException("HED model files not found");

    std::string model = cvtest::findDataFile("hed_pretrained_bsds.caffemodel", true);
    std::string proto = cvtest::findDataFile("deploy.prototxt", true);

    cv::Mat img = cv::Mat::zeros(100, 100, CV_8UC3);
    auto detector = cv::hed::HEDDetector::create(model, proto);
    cv::Mat edges = detector->detectEdges(img);

    EXPECT_EQ(edges.rows, 100);
    EXPECT_EQ(edges.cols, 100);
    EXPECT_EQ(edges.type(), CV_32F);
}

}} // namespace