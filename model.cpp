//
// Created by icys on 25-2-10.
//

#include "model.h"

namespace NBCapture {
    LatexOcr::LatexOcr() {
        // TODO: Replace ugly lower op with InstanceNorm to speed up

        decoder.load_param("./assets/Simple-LaTeX-OCR_Decoder_fp16.param");
        decoder.load_model("./assets/Simple-LaTeX-OCR_Decoder_fp16.bin");

        encoder.load_param("./assets/Simple-LaTeX-OCR_Encoder_fp16.param");
        encoder.load_model("./assets/Simple-LaTeX-OCR_Encoder_fp16.bin");

        // simple_latex_ocr_vocab.txt
        std::ifstream ifs("./assets/simple_latex_ocr_vocab.txt");
        std::string line;
        while (std::getline(ifs, line)) {
            token_list.push_back(line);
        }
    }


    std::string post_process(std::string str) {
        // 替换 Ġ Ċ 为 ' '
        // 正则替换
        str = std::regex_replace(str, std::regex("Ġ"), " ");
        str = std::regex_replace(str, std::regex("Ċ"), " ");

        return str;
    }

    std::string LatexOcr::forward(const cv::Mat& image) {
        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        // Canny
        // cv::Canny(gray, gray, 50, 150, 3);
        // 转白底黑字
        // gray = 255 - gray;

        float scale = 1.f;
        int target_width, target_height;
        if (image.cols * 128.0f / 640.0f > image.rows) {
            scale = 640 / static_cast<float>(image.cols);
            target_width = 640;
            target_height = static_cast<int>(image.rows * scale);
        } else {
            scale = 128 / static_cast<float>(image.rows);
            target_height = 128;
            target_width = static_cast<int>(image.cols * scale);
        }

        int wpad = (640 - target_width) / 2;
        int hpad = (128 - target_height) / 2;

        // resize
        cv::resize(gray, gray, cv::Size(target_width, target_height));
        const int pad_color = 114;

        cv::Mat pad_img = cv::Mat(128, 640, CV_8UC1, cv::Scalar(pad_color));
        gray.copyTo(pad_img(cv::Rect(wpad, hpad, target_width, target_height)));

        auto in = ncnn::Mat::from_pixels(pad_img.data, ncnn::Mat::PIXEL_GRAY, 640, 128);
        const float mean_vals[1] = {0.7931 * 255};
        const float norm_vals[1] = {1.0 / 0.1738 / 255.0};
        in.substract_mean_normalize(mean_vals, norm_vals);

        auto ex = encoder.create_extractor();
        ex.input("in0", in.clone());
        ncnn::Mat feat;
        ex.extract("out0", feat);

        const int max_step = 1024;
        int step = 0;

        // output, feat, mask, pos

        std::vector<int32_t> output;
        output.push_back(1); // <sos>
        std::vector<int32_t> pos;

        while (step < max_step) {
            auto ex2 = decoder.create_extractor();
            pos.push_back(step);

            ncnn::Mat mask(step+1,step+1,1);
            mask.fill(0.0f);
            for (int i = 0; i < step+1; i++) {
                for (int j = i + 1; j < step+1; j++) {
                    mask.row(i)[j] = -1e30f;
                }
            }

            ex2.input("in0", ncnn::Mat(output.size(),output.data()).clone());
            ex2.input("in1", feat.clone());
            ex2.input("in2", mask.clone());
            ex2.input("in3", ncnn::Mat(pos.size(),pos.data()).clone());

            ncnn::Mat out0;
            ex2.extract("out0", out0);

            const int len_token = 1200;

            output.resize(step+2);
            for (int i = 0; i < step+1; i++) {
                int maxarg = 0;
                float maxval = -1e30f;
                for (int j = 0; j < len_token; j++) {
                    if (out0.row(i)[j] > maxval) {
                        maxval = out0.row(i)[j];
                        maxarg = j;
                    }
                }
                output[i+1] = maxarg;

                if (maxarg == 2) {
                    output.pop_back();
                    goto Over;
                }
            }
            step++;
        }
        Over:
        std::string ret;
        for (int i = 1; i < output.size(); i++) {
            ret += token_list[output[i]];
        }
        return post_process(ret);

    }
} // NBCapture