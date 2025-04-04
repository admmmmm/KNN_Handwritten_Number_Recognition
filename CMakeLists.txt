#include <iostream>
#include <vector>
#include <unordered_map>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <random>
#include <algorithm>
#include <cmath>

namespace fs = std::filesystem;

enum DistanceType {
    EUCLIDEAN,
    MANHATTAN,
    COSINE
};

float euclidean_distance(const cv::Mat& a, const cv::Mat& b) {
    float sum = 0.0f;
    for (int i = 0; i < a.total(); ++i) {
        float diff = a.at<float>(i) - b.at<float>(i);
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

float manhattan_distance(const cv::Mat& a, const cv::Mat& b) {
    float sum = 0.0f;
    for (int i = 0; i < a.total(); ++i) {
        sum += std::abs(a.at<float>(i) - b.at<float>(i));
    }
    return sum;
}

float cosine_distance(const cv::Mat& a, const cv::Mat& b) {
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (int i = 0; i < a.total(); ++i) {
        float va = a.at<float>(i), vb = b.at<float>(i);
        dot += va * vb;
        norm_a += va * va;
        norm_b += vb * vb;
    }
    return 1.0f - dot / (std::sqrt(norm_a) * std::sqrt(norm_b) + 1e-6f);
}

float compute_distance(const cv::Mat& a, const cv::Mat& b, DistanceType type) {
    switch (type) {
    case EUCLIDEAN: return euclidean_distance(a, b);
    case MANHATTAN: return manhattan_distance(a, b);
    case COSINE: return cosine_distance(a, b);
    default: return euclidean_distance(a, b);
    }
}

class MNISTLoader {
public:
    static std::vector<std::pair<cv::Mat, int>> loadImages(const std::string& dataset_path) {
        std::vector<std::pair<cv::Mat, int>> data;

        try {
            for (const auto& class_dir : fs::directory_iterator(dataset_path)) {
                if (!class_dir.is_directory()) continue;
                int label = std::stoi(class_dir.path().filename().string());

                for (const auto& img_path : fs::directory_iterator(class_dir)) {
                    cv::Mat img = cv::imread(img_path.path().string(), cv::IMREAD_GRAYSCALE);
                    if (img.empty()) continue;

                    img.convertTo(img, CV_32F, 1.0 / 255.0);
                    data.emplace_back(img, label);
                }
            }
        }
        catch (const std::exception& e) {
            std::cerr << "数据加载错误: " << e.what() << std::endl;
        }

        std::cout << "已加载 " << data.size() << " 张图像" << std::endl;
        return data;
    }
};

int knn_classify(const cv::Mat& query,
    const std::vector<std::pair<cv::Mat, int>>& train_data,
    int k, DistanceType dist_type)
{
    std::vector<std::pair<float, int>> distances;

#pragma omp parallel for
    for (int i = 0; i < train_data.size(); ++i) {
        float dist = compute_distance(query, train_data[i].first, dist_type);
#pragma omp critical
        distances.emplace_back(dist, train_data[i].second);
    }

    std::sort(distances.begin(), distances.end());

    std::unordered_map<int, int> counts;
    for (int i = 0; i < k && i < distances.size(); ++i) {
        counts[distances[i].second]++;
    }

    return std::max_element(counts.begin(), counts.end(),
        [](auto& a, auto& b) { return a.second < b.second; })->first;
}

void show_prediction(const cv::Mat& img, int pred, int true_label, const std::string& dist_name) {
    cv::Mat gray_img_8u;
    img.convertTo(gray_img_8u, CV_8U, 255.0);
    cv::imshow("MNIST预测(" + dist_name + ")", gray_img_8u);
    std::cout << "【" << dist_name << "】预测: " << pred << "，真实: " << true_label << std::endl;
    cv::waitKey(30);  
}

std::string distance_type_name(DistanceType type) {
    switch (type) {
    case EUCLIDEAN: return "欧几里得";
    case MANHATTAN: return "曼哈顿";
    case COSINE: return "余弦";
    default: return "未知";
    }
}

int main() {
    const std::string train_path = "D:\\dev\\Number_Recognition\\small_training";
    const std::string test_path = "D:\\dev\\Number_Recognition\\mnist_png\\mnist_png\\mnist_png\\testing";

    if (!fs::exists(train_path) || !fs::exists(test_path)) {
        std::cerr << "路径不存在，请确认路径正确。" << std::endl;
        return -1;
    }

    auto train_data = MNISTLoader::loadImages(train_path);
    auto test_data = MNISTLoader::loadImages(test_path);

    if (train_data.empty() || test_data.empty()) {
        std::cerr << "加载数据失败！" << std::endl;
        return -1;
    }

    // 随机打乱 test_data
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(test_data.begin(), test_data.end(), g);

    int K;
    std::cout << "请输入 K 值：";
    std::cin >> K;
    if (K <= 0) {
        std::cerr << "K值必须大于0" << std::endl;
        return -1;
    }

    for (int dist = 0; dist <= 2; ++dist) {
        DistanceType dist_type = static_cast<DistanceType>(dist);
        std::string dist_name = distance_type_name(dist_type);

        std::cout << "\n=== [" << dist_name << " 距离] 开始测试随机100张样本 ===\n";

        int correct = 0;
        for (int i = 0; i < 100 && i < test_data.size(); ++i) {
            const auto& [img, true_label] = test_data[i];
            int prediction = knn_classify(img, train_data, K, dist_type);
            if (prediction == true_label) ++correct;
            show_prediction(img, prediction, true_label, dist_name);
        }

        float accuracy = 100.0f * correct / 100.0f;
        std::cout << "【" << dist_name << " 距离】准确率: " << accuracy << "%" << std::endl;
        cv::destroyAllWindows();
    }

    return 0;
}
