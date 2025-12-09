#include <filesystem>  // C++17 (폴더 생성용)
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    // ===== 1. 영상 열기 =====
    std::string videoPath = (argc > 1) ? argv[1] : "./input/2.mp4";
    cv::VideoCapture cap(videoPath);

    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open video: " << videoPath << std::endl;
        return -1;
    }

    // 파일명 추출 (확장자 제외)
    fs::path p(videoPath);
    std::string videoName = p.stem().string();

    // ===== 2. 저장 폴더 설정 (Motion / Shock 분리) =====
    std::string dirMotion = "recorded_motion";
    std::string dirShock = "recorded_shock";

    // 폴더가 없으면 생성
    if (!fs::exists(dirMotion)) fs::create_directory(dirMotion);
    if (!fs::exists(dirShock)) fs::create_directory(dirShock);

    // FPS 및 코덱 설정
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0) fps = 30.0;
    int framesFor3Sec = static_cast<int>(fps * 3.0);
    int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');

    // ===== 3. 알고리즘 파라미터 =====
    const double SCALE = 0.5;
    const int BLUR_KSIZE = 3;
    const int DIFF_THRESHOLD = 30;

    // 모션 임계값 (ROI 기준)
    const double MOTION_RATIO_THRESH = 0.002;
    const double HUGE_MOTION_RATIO_THRESH = 0.02;

    // 충격 임계값 (전체 화면 기준)
    const double SHOCK_RATIO_THRESH = 0.30;

    const int IDLE_SKIP = 4;

    cv::Mat frame, gray, smallGray, prevSmallGray;
    cv::Mat diffRoi, maskRoi;
    cv::Mat diffWhole, maskWhole;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));

    // 상태 변수
    bool firstFrame = true;
    int frameIndex = 0;
    cv::Rect roiRect;
    bool roiInitialized = false;

    // 감지 상태
    bool detectionActive = false;
    int framesSinceLastEvent = 0;
    int overlayFramesLeft = 0;
    bool isShockDetected = false;

    cv::VideoWriter recorder;

    cv::namedWindow("Original", cv::WINDOW_NORMAL);
    cv::namedWindow("Detection", cv::WINDOW_NORMAL);

    while (true) {
        // ===== Idle 모드 스킵 =====
        if (!detectionActive) {
            for (int i = 0; i < IDLE_SKIP; ++i) {
                if (!cap.grab()) goto exit_loop;
                ++frameIndex;
            }
        }

        if (!cap.read(frame)) break;
        ++frameIndex;

        // 전처리
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        if (SCALE != 1.0) {
            cv::resize(gray, smallGray, cv::Size(), SCALE, SCALE, cv::INTER_AREA);
        } else {
            smallGray = gray;
        }

        // ROI 설정 (하단 30%)
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

        // ===== 1. 전체 화면 차분 (충격 감지) =====
        cv::absdiff(smallGray, prevSmallGray, diffWhole);
        cv::threshold(diffWhole, maskWhole, DIFF_THRESHOLD, 255, cv::THRESH_BINARY);
        int wholePixels = maskWhole.rows * maskWhole.cols;
        int wholeMotionPixels = cv::countNonZero(maskWhole);
        double wholeRatio = (double)wholeMotionPixels / wholePixels;

        // ===== 2. ROI 차분 (일반 모션 감지) =====
        cv::Mat roiCurr = smallGray(roiRect);
        cv::Mat roiPrev = prevSmallGray(roiRect);
        cv::absdiff(roiCurr, roiPrev, diffRoi);
        cv::threshold(diffRoi, maskRoi, DIFF_THRESHOLD, 255, cv::THRESH_BINARY);
        cv::morphologyEx(maskRoi, maskRoi, cv::MORPH_OPEN, kernel);

        int roiPixels = maskRoi.rows * maskRoi.cols;
        int roiMotionPixels = cv::countNonZero(maskRoi);
        double roiRatio = (double)roiMotionPixels / roiPixels;

        // ===== 판정 =====
        std::string statusText = "NO MOTION";
        cv::Scalar statusColor = cv::Scalar(0, 255, 0);

        bool eventTriggered = false;
        isShockDetected = false;

        // 충격 우선 순위
        if (wholeRatio > SHOCK_RATIO_THRESH) {
            statusText = "SHOCK DETECTED!";
            statusColor = cv::Scalar(255, 0, 255);  // 보라색
            eventTriggered = true;
            isShockDetected = true;
        } else if (roiRatio > HUGE_MOTION_RATIO_THRESH) {
            statusText = "HUGE MOTION";
            statusColor = cv::Scalar(0, 0, 255);  // 빨강
            eventTriggered = true;
        } else if (roiRatio > MOTION_RATIO_THRESH) {
            statusText = "MOTION";
            statusColor = cv::Scalar(0, 255, 255);  // 노랑
            eventTriggered = true;
        }

        // ===== 상태 머신 및 녹화 로직 =====
        if (eventTriggered) {
            framesSinceLastEvent = 0;

            if (!detectionActive) {
                // [상태 전환] Idle -> Active (녹화 시작)
                detectionActive = true;
                overlayFramesLeft = framesFor3Sec;

                // 저장할 폴더 및 파일명 결정
                // 현재 이벤트가 '충격'이면 shock 폴더, 아니면 motion 폴더
                std::string targetDir = isShockDetected ? dirShock : dirMotion;

                std::stringstream ss;
                ss << targetDir << "/" << videoName << "_" << frameIndex << ".mp4";
                std::string outputFilename = ss.str();

                recorder.open(outputFilename, fourcc, fps, frame.size(), true);

                if (recorder.isOpened()) {
                    std::cout << ">>> Start Recording (" << (isShockDetected ? "SHOCK" : "MOTION")
                              << "): " << outputFilename << std::endl;
                }
            } else {
                // 이미 녹화 중일 때 (Active)
                if (overlayFramesLeft <= 0) overlayFramesLeft = framesFor3Sec;

                // (선택사항) 만약 일반 모션 녹화 중에 충격이 발생하면 로그만 출력
                if (isShockDetected) {
                    std::cout << "!!! Shock occurred during recording !!!" << std::endl;
                }
            }
        } else {
            // 이벤트 없음
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

        // ===== 녹화 (원본 저장) =====
        if (detectionActive && recorder.isOpened()) {
            recorder.write(frame);
        }

        // ===== 화면 출력 =====
        cv::Mat originalDisplay = frame.clone();
        cv::putText(originalDisplay, statusText, cv::Point(20, 40),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, statusColor, 2);

        if (isShockDetected) {
            cv::rectangle(originalDisplay, cv::Rect(0, 0, frame.cols, frame.rows), statusColor, 10);
        }
        cv::imshow("Original", originalDisplay);

        cv::Mat detectDisplay;
        if (detectionActive) {
            detectDisplay = frame.clone();
            std::string alertMsg = isShockDetected ? "SHOCK DETECTED" : "MOTION DETECT";
            // 녹화 중일 때 현재 어떤 폴더에 저장 중인지 구분하기 위해 색상을 다르게 할 수도 있습니다.
            // 여기서는 현재 프레임 상태(isShockDetected)에 따라 표시합니다.
            cv::Scalar alertColor = isShockDetected ? cv::Scalar(255, 0, 255) : cv::Scalar(0, 0, 255);

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