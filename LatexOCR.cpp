//
// Created by icys on 25-2-10.
//
#include <NBLatex/model.h>
#include <iostream>

int WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine,
            int nShowCmd)
{
    cv::Mat image = cv::imread("./assets/latex.png");
    NBCapture::LatexOcr model;
    std::cout << model.forward(image);
}