#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    std::string videoPath = (argc > 1) ? argv[1] : "./input/1.mp4";
    cv::VideoCapture cap(videoPath);

    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open video: " << videoPath << std::endl;
        return -1;
    }

    fs::path p(videoPath);
    std::string videoName = p.stem().string();

    std::string dirMotion = "recorded_motion";

    if (!fs::exists(dirMotion)) fs::create_directory(dirMotion);

    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0) fps = 30.0;
    int framesFor3Sec = static_cast<int>(fps * 3.0);
    int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');

    const double SCALE = 0.5;
    const int BLUR_KSIZE = 3;
    const int DIFF_THRESHOLD = 30;

    const double MOTION_RATIO_THRESH = 0.002;
    const double HUGE_MOTION_RATIO_THRESH = 0.02;

    const int IDLE_SKIP = 4;

    cv::Mat frame, gray, smallGray, prevSmallGray;
    cv::Mat diffRoi, maskRoi;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));

    bool firstFrame = true;
    int frameIndex = 0;
    cv::Rect roiRect;
    bool roiInitialized = false;

    bool detectionActive = false;
    int framesSinceLastEvent = 0;
    int overlayFramesLeft = 0;

    cv::VideoWriter recorder;

    cv::namedWindow("Original", cv::WINDOW_NORMAL);
    cv::namedWindow("Detection", cv::WINDOW_NORMAL);

    while (true) {
        if (!detectionActive) {
            for (int i = 0; i < IDLE_SKIP; ++i) {
                if (!cap.grab()) goto exit_loop;
                ++frameIndex;
            }
        }

        if (!cap.read(frame)) break;
        ++frameIndex;

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        if (SCALE != 1.0) {
            cv::resize(gray, smallGray, cv::Size(), SCALE, SCALE, cv::INTER_AREA);
        } else {
            smallGray = gray;
        }

        if (!roiInitialized) {
            int h = smallGray.rows;
            int w = smallGray.cols;
            int roiY = static_cast<int>(h * 0.7);
            roiRect = cv::Rect(0, roiY, w, h - roiY);
            roiInitialized = true;
        }

        if (BLUR_KSIZE > 1) {
            cv::GaussianBlur(smallGray, smallGray, cv::Size(BLUR_KSIZE, BLUR_KSIZE), 0);
        }

        if (firstFrame) {
            prevSmallGray = smallGray.clone();
            firstFrame = false;
            cv::imshow("Original", frame);
            cv::imshow("Detection", cv::Mat::zeros(frame.size(), frame.type()));
            if (cv::waitKey(1) == 27) break;
            continue;
        }

        cv::Mat roiCurr = smallGray(roiRect);
        cv::Mat roiPrev = prevSmallGray(roiRect);
        cv::absdiff(roiCurr, roiPrev, diffRoi);
        cv::threshold(diffRoi, maskRoi, DIFF_THRESHOLD, 255, cv::THRESH_BINARY);
        cv::morphologyEx(maskRoi, maskRoi, cv::MORPH_OPEN, kernel);

        int roiPixels = maskRoi.rows * maskRoi.cols;
        int roiMotionPixels = cv::countNonZero(maskRoi);
        double roiRatio = (double)roiMotionPixels / roiPixels;

        std::string statusText = "NO MOTION";
        cv::Scalar statusColor = cv::Scalar(0, 255, 0);

        bool eventTriggered = false;

        if (roiRatio > HUGE_MOTION_RATIO_THRESH) {
            statusText = "HUGE MOTION";
            statusColor = cv::Scalar(0, 0, 255);
            eventTriggered = true;
        } else if (roiRatio > MOTION_RATIO_THRESH) {
            statusText = "MOTION";
            statusColor = cv::Scalar(0, 255, 255);
            eventTriggered = true;
        }

        if (eventTriggered) {
            framesSinceLastEvent = 0;

            if (!detectionActive) {
                detectionActive = true;
                overlayFramesLeft = framesFor3Sec;

                std::stringstream ss;
                ss << dirMotion << "/" << videoName << "_" << frameIndex << ".mp4";
                std::string outputFilename = ss.str();

                recorder.open(outputFilename, fourcc, fps, frame.size(), true);

                if (recorder.isOpened()) {
                    std::cout << ">>> Start Recording: " << outputFilename << std::endl;
                }
            } else {
                if (overlayFramesLeft <= 0) overlayFramesLeft = framesFor3Sec;
            }
        } else {
            if (detectionActive) {
                framesSinceLastEvent++;
                if (framesSinceLastEvent > framesFor3Sec) {
                    detectionActive = false;
                    overlayFramesLeft = 0;

                    if (recorder.isOpened()) {
                        recorder.release();
                        std::cout << ">>> Stop Recording." << std::endl;
                    }
                }
            }
        }

        if (overlayFramesLeft > 0) overlayFramesLeft--;

        if (detectionActive && recorder.isOpened()) {
            recorder.write(frame);
        }

        cv::Mat originalDisplay = frame.clone();
        cv::putText(originalDisplay, statusText, cv::Point(20, 40),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, statusColor, 2);

        cv::imshow("Original", originalDisplay);

        cv::Mat detectDisplay;
        if (detectionActive) {
            detectDisplay = frame.clone();
            std::string alertMsg = "MOTION DETECT";
            cv::Scalar alertColor = cv::Scalar(0, 0, 255);

            if (overlayFramesLeft > 0) {
                cv::putText(detectDisplay, alertMsg,
                            cv::Point(50, 80),
                            cv::FONT_HERSHEY_SIMPLEX,
                            1.5, alertColor, 3);
            }
        } else {
            detectDisplay = cv::Mat::zeros(frame.size(), frame.type());
        }
        cv::imshow("Detection", detectDisplay);

        char key = (char)cv::waitKey(1);
        if (key == 27 || key == 'q' || key == 'Q') break;

        prevSmallGray = smallGray.clone();
    }

exit_loop:
    if (recorder.isOpened()) recorder.release();
    cap.release();
    cv::destroyAllWindows();
    return 0;
}