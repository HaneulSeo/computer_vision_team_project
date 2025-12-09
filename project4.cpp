#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char** argv) {
    // ===== 1. 영상 열기 =====
    std::string videoPath = (argc > 1) ? argv[1] : "./input/2.mp4";
    cv::VideoCapture cap(videoPath);

    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open video: " << videoPath << std::endl;
        return -1;
    }

    // FPS 가져오기 (3초 기준 계산에 사용)
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0) fps = 30.0;  // 못 가져오면 기본 30fps 가정
    int framesFor3Sec = static_cast<int>(fps * 3.0);

    // ===== 2. 파라미터 설정 =====
    const double SCALE = 0.5;       // 해상도 축소 비율
    const int BLUR_KSIZE = 3;       // GaussianBlur 커널 크기
    const int DIFF_THRESHOLD = 30;  // 이진화 threshold (0~255)

    const double MOTION_RATIO_THRESH = 0.002;      // 일반 모션 기준 (0.2%)
    const double HUGE_MOTION_RATIO_THRESH = 0.02;  // huge motion 기준 (2%)

    // idle 모드에서 건너뛸 프레임 수 (예: 4면 5프레임 중 1개만 분석)
    const int IDLE_SKIP = 4;

    cv::Mat frame, gray, smallGray, prevSmallGray;
    cv::Mat diff, mask, kernel;

    // morphology용 커널
    kernel = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));

    bool firstFrame = true;
    int frameIndex = 0;

    // ===== ROI (하단 30%) 설정용 =====
    cv::Rect roiRect;
    bool roiInitialized = false;

    // ===== Detection window 상태 관리 =====
    bool detectionActive = false;   // 실제 구현 화면이 영상 출력 중인지 여부
    int framesSinceLastMotion = 0;  // 마지막 모션 이후 지난 프레임 수
    int overlayFramesLeft = 0;      // "MOTION DETECT" 텍스트를 더 보여줄 프레임 수

    // 창 두 개 생성
    cv::namedWindow("Original", cv::WINDOW_NORMAL);
    cv::namedWindow("Detection", cv::WINDOW_NORMAL);

    while (true) {
        // ===== idle 모드에서는 프레임 일부만 분석 =====
        if (!detectionActive) {
            // IDLE_SKIP 개수만큼은 decode 없이 grab()으로 그냥 넘김
            for (int i = 0; i < IDLE_SKIP; ++i) {
                if (!cap.grab()) {
                    std::cout << "End of video (during idle skip)." << std::endl;
                    cap.release();
                    cv::destroyAllWindows();
                    return 0;
                }
                ++frameIndex;
            }
        }

        // ===== 실제 분석할 프레임 읽기 =====
        if (!cap.read(frame)) {
            std::cout << "End of video." << std::endl;
            break;
        }
        ++frameIndex;

        // ===== 4. 그레이스케일 변환 =====
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // ===== 5. 해상도 축소 =====
        if (SCALE != 1.0) {
            cv::resize(gray, smallGray, cv::Size(), SCALE, SCALE, cv::INTER_AREA);
        } else {
            smallGray = gray;
        }

        // ROI 초기화 (하단 30%)
        if (!roiInitialized) {
            int h = smallGray.rows;
            int w = smallGray.cols;
            int roiY = static_cast<int>(h * 0.7);  // 위 70% 제외, 아래 30% 사용
            int roiH = h - roiY;
            roiRect = cv::Rect(0, roiY, w, roiH);
            roiInitialized = true;
        }

        // ===== 6. 블러 =====
        if (BLUR_KSIZE > 1) {
            cv::GaussianBlur(smallGray, smallGray, cv::Size(BLUR_KSIZE, BLUR_KSIZE), 0);
        }

        // 첫 프레임이면 이전 프레임만 세팅하고 다음으로
        if (firstFrame) {
            prevSmallGray = smallGray.clone();
            firstFrame = false;

            // Original 창은 첫 프레임부터 보여줌
            cv::imshow("Original", frame);
            cv::Mat black(frame.rows, frame.cols, frame.type(), cv::Scalar::all(0));
            cv::imshow("Detection", black);

            char key = static_cast<char>(cv::waitKey(1));
            if (key == 27 || key == 'q' || key == 'Q') {
                break;
            }
            continue;
        }

        // ===== 7. ROI에서 이전 프레임과 차이 =====
        cv::Mat roiCurr = smallGray(roiRect);
        cv::Mat roiPrev = prevSmallGray(roiRect);

        cv::absdiff(roiCurr, roiPrev, diff);  // diff는 ROI 크기

        // ===== 8. 이진화 =====
        cv::threshold(diff, mask, DIFF_THRESHOLD, 255, cv::THRESH_BINARY);

        // ===== 9. Morphology (노이즈 제거) =====
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);

        // ===== 10. 모션 비율 계산 (ROI 기준) =====
        int motionPixels = cv::countNonZero(mask);
        int totalPixels = mask.rows * mask.cols;
        double motionRatio = 0.0;
        if (totalPixels > 0) {
            motionRatio = static_cast<double>(motionPixels) / totalPixels;
        }

        // ===== 11. 상태 판정 (NO / MOTION / HUGE MOTION) =====
        std::string statusText;
        cv::Scalar statusColor;

        if (motionRatio > HUGE_MOTION_RATIO_THRESH) {
            statusText = "HUGE MOTION";
            statusColor = cv::Scalar(0, 0, 255);  // 빨강
        } else if (motionRatio > MOTION_RATIO_THRESH) {
            statusText = "MOTION";
            statusColor = cv::Scalar(0, 255, 255);  // 노랑
        } else {
            statusText = "NO MOTION";
            statusColor = cv::Scalar(0, 255, 0);  // 초록
        }

        bool hasMotion = (motionRatio > MOTION_RATIO_THRESH);

        // 콘솔 로그
        std::cout << "[Frame " << frameIndex << "] "
                  << "motionPixels=" << motionPixels
                  << ", totalPixels=" << totalPixels
                  << ", ratio=" << motionRatio * 100.0 << "% -> "
                  << statusText
                  << " | detectionActive=" << detectionActive
                  << std::endl;

        // ===== Detection window 상태 업데이트 =====
        if (hasMotion) {
            framesSinceLastMotion = 0;

            // 모션 처음 감지되면:
            if (!detectionActive) {
                // idle → active로 전환
                detectionActive = true;
                overlayFramesLeft = framesFor3Sec;  // 약 3초간 "MOTION DETECT"
            } else {
                // 이미 active 상태인데 텍스트를 연장하고 싶으면:
                if (overlayFramesLeft <= 0)
                    overlayFramesLeft = framesFor3Sec;
            }
        } else {
            if (detectionActive) {
                framesSinceLastMotion++;
                // active 모드에서 3초 이상 모션 없으면 다시 idle 모드
                if (framesSinceLastMotion > framesFor3Sec) {
                    detectionActive = false;
                    overlayFramesLeft = 0;
                }
            }
        }

        if (overlayFramesLeft > 0)
            overlayFramesLeft--;

        // ===== 12. 화면 출력 =====

        // (1) Original: 항상 원본 + 상태 텍스트 표시
        cv::Mat originalDisplay = frame.clone();
        cv::putText(originalDisplay, statusText,
                    cv::Point(20, 40),
                    cv::FONT_HERSHEY_SIMPLEX,
                    1.0, statusColor, 2);
        cv::imshow("Original", originalDisplay);

        // (2) Detection: idle이면 검은 화면, active면 영상 + "MOTION DETECT"(3초)
        cv::Mat detectDisplay;
        if (detectionActive) {
            detectDisplay = frame.clone();
            if (overlayFramesLeft > 0) {
                cv::putText(detectDisplay, "MOTION DETECT",
                            cv::Point(50, 80),
                            cv::FONT_HERSHEY_SIMPLEX,
                            1.5, cv::Scalar(0, 0, 255), 3);  // 빨간색 굵게
            }
        } else {
            detectDisplay = cv::Mat(frame.rows, frame.cols, frame.type(), cv::Scalar::all(0));
        }

        cv::imshow("Detection", detectDisplay);

        // ESC 또는 q 누르면 종료
        char key = static_cast<char>(cv::waitKey(1));
        if (key == 27 || key == 'q' || key == 'Q') {
            break;
        }

        // ===== 13. 이전 프레임 업데이트 =====
        prevSmallGray = smallGray.clone();
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
