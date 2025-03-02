//
// Created by icys on 25-2-10.
//

#ifndef MODEL_Latex_H
#define MODEL_Latex_H

namespace NBCapture {

class LatexOcr {
public:
    LatexOcr();
    std::string forward(const cv::Mat& image);

    ncnn::Net decoder;
    ncnn::Net encoder;
    std::vector<std::string> token_list;
};

} // NBCapture

#endif //MODEL_Latex_H
