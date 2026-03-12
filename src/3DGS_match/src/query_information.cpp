#include "3DGS_match/query_information.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <bitset>
#include <filesystem>
#include <chrono>
#include <stdexcept>
#include <algorithm>
#include <unordered_map>

namespace fs = std::filesystem;
using std::chrono::steady_clock;
using std::chrono::duration;

// 外部进度条函数
extern void updateProgressBar(size_t processed, size_t total, steady_clock::time_point startTime);

// ============================================================================
// VisualQueryData 实现
// ============================================================================

// 默认构造函数
VisualQueryData::VisualQueryData() 
    : frame_id(-1), x(0.0), y(0.0), z(0.0),
      rotation{{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}}} {}

// 完整构造函数
VisualQueryData::VisualQueryData(
    int frame_id,
    double x, double y, double z,
    const std::array<std::array<double, 3>, 3>& rot,
    const std::string& desc1,
    const std::string& desc2,
    const std::string& desc4
) : frame_id(frame_id), x(x), y(y), z(z), rotation(rot),
    descriptors_size1(desc1), descriptors_size2(desc2), descriptors_size4(desc4) {}

// 从旋转矩阵提取Yaw角（弧度）
double VisualQueryData::getYawAngle() const {
    return std::atan2(rotation[1][0], rotation[0][0]);
}

// 从旋转矩阵提取Yaw角（角度）
double VisualQueryData::getYawDegrees() const {
    return getYawAngle() * 180.0 / M_PI;
}

// 提取ORB特征
std::string VisualQueryData::extractOrbFeatures(const std::string& image_path, int n) {
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        throw std::runtime_error("无法读取图像: " + image_path);
    }

    int rows = image.rows;
    int cols = image.cols;
    
    if (rows < n || cols < n) {
        throw std::runtime_error("图像尺寸过小: " + image_path);
    }

    int partRows = rows / n;
    int partCols = cols / n;

    std::string binaryString;
    binaryString.reserve(n * n * 256);

    cv::Ptr<cv::ORB> orb = cv::ORB::create();

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int startX = j * partCols;
            int startY = i * partRows;
            int width = (j == n - 1) ? (cols - startX) : partCols;
            int height = (i == n - 1) ? (rows - startY) : partRows;
            
            cv::Mat subImage = image(cv::Rect(startX, startY, width, height));
            cv::Mat imageResized;
            cv::resize(subImage, imageResized, cv::Size(63, 63));

            cv::KeyPoint keypoint(31.0f, 31.0f, 31.0f);
            std::vector<cv::KeyPoint> keypoints{keypoint};
            cv::Mat descriptors;

            orb->compute(imageResized, keypoints, descriptors);

            if (descriptors.empty()) {
                throw std::runtime_error("ORB特征提取失败: " + image_path);
            }

            for (int row = 0; row < descriptors.rows; row++) {
                for (int col = 0; col < descriptors.cols; col++) {
                    int value = static_cast<int>(descriptors.at<uchar>(row, col));
                    std::bitset<8> bits(value);
                    binaryString += bits.to_string();
                }
            }
        }
    }

    return binaryString;
}

// ✅ 修改：从CSV构建查询序列（添加范围过滤支持）
std::vector<VisualQueryData> VisualQueryData::buildQueryFromCSV(
    const std::string& csv_path,
    const std::string& images_folder,
    bool enable_range_filter,
    int start_frame_id,
    int end_frame_id) {
    
    std::cout << "========================================" << std::endl;
    std::cout << "开始处理查询数据" << std::endl;
    std::cout << "CSV文件: " << csv_path << std::endl;
    std::cout << "图片目录: " << images_folder << std::endl;
    
    // ✅ 显示范围过滤信息
    if (enable_range_filter) {
        std::cout << "范围过滤: 启用" << std::endl;
        std::cout << "  frame_id范围: " << start_frame_id << " ~ " << end_frame_id << std::endl;
    } else {
        std::cout << "范围过滤: 禁用（处理所有帧）" << std::endl;
    }
    std::cout << "========================================" << std::endl;

    // ==================== 步骤1：读取CSV文件 ====================
    std::ifstream csv_file(csv_path);
    if (!csv_file.is_open()) {
        throw std::runtime_error("无法打开CSV文件: " + csv_path);
    }

    std::string header;
    if (!std::getline(csv_file, header)) {
        throw std::runtime_error("CSV文件为空: " + csv_path);
    }

    std::cout << "CSV表头: " << header << std::endl;

    std::vector<VisualQueryData> query_vec;
    std::string line;
    size_t line_number = 1;
    size_t skipped_count = 0;  // ✅ 记录跳过的帧数

    while (std::getline(csv_file, line)) {
        line_number++;
        
        if (line.empty() || line.find_first_not_of(" \t\r\n") == std::string::npos) {
            continue;
        }

        std::istringstream ss(line);
        std::string token;
        std::vector<std::string> tokens;

        while (std::getline(ss, token, ',')) {
            size_t start = token.find_first_not_of(" \t\r\n");
            size_t end = token.find_last_not_of(" \t\r\n");
            if (start != std::string::npos && end != std::string::npos) {
                tokens.push_back(token.substr(start, end - start + 1));
            } else {
                tokens.push_back("");
            }
        }

        // CSV格式: frame_id, x, y, z, r00, r01, r02, r10, r11, r12, r20, r21, r22
        if (tokens.size() < 13) {
            throw std::runtime_error(
                "CSV第" + std::to_string(line_number) + "行格式错误: 需要13列，实际" + 
                std::to_string(tokens.size()) + "列"
            );
        }

        try {
            int frame_id = std::stoi(tokens[0]);
            
            // ✅ 范围过滤逻辑
            if (enable_range_filter) {
                if (frame_id < start_frame_id || frame_id > end_frame_id) {
                    skipped_count++;
                    continue;  // 跳过范围外的帧
                }
            }
            
            double x = std::stod(tokens[1]);
            double y = std::stod(tokens[2]);
            double z = std::stod(tokens[3]);

            std::array<std::array<double, 3>, 3> rot;
            rot[0][0] = std::stod(tokens[4]);
            rot[0][1] = std::stod(tokens[5]);
            rot[0][2] = std::stod(tokens[6]);
            rot[1][0] = std::stod(tokens[7]);
            rot[1][1] = std::stod(tokens[8]);
            rot[1][2] = std::stod(tokens[9]);
            rot[2][0] = std::stod(tokens[10]);
            rot[2][1] = std::stod(tokens[11]);
            rot[2][2] = std::stod(tokens[12]);

            query_vec.emplace_back(frame_id, x, y, z, rot, "", "", "");
        } catch (const std::exception& e) {
            throw std::runtime_error(
                "CSV第" + std::to_string(line_number) + "行数据转换失败: " + e.what()
            );
        }
    }
    csv_file.close();

    if (query_vec.empty()) {
        throw std::runtime_error("CSV文件无有效数据（可能被范围过滤器全部过滤）");
    }

    // ✅ 显示过滤统计
    if (enable_range_filter) {
        std::cout << "范围过滤统计:" << std::endl;
        std::cout << "  跳过的帧数: " << skipped_count << std::endl;
        std::cout << "  保留的帧数: " << query_vec.size() << std::endl;
    } else {
        std::cout << "成功读取 " << query_vec.size() << " 条查询记录" << std::endl;
    }

    // ==================== 步骤2：验证图片文件夹 ====================
    if (!fs::exists(images_folder) || !fs::is_directory(images_folder)) {
        throw std::runtime_error("图片文件夹不存在: " + images_folder);
    }

    // ==================== 步骤3：建立frame_id到索引的映射 ====================
    std::unordered_map<int, size_t> frame_map;
    for (size_t i = 0; i < query_vec.size(); ++i) {
        int fid = query_vec[i].frame_id;
        if (frame_map.count(fid) > 0) {
            throw std::runtime_error("CSV存在重复的frame_id: " + std::to_string(fid));
        }
        frame_map[fid] = i;
    }

    // ==================== 步骤4：遍历图片并提取特征 ====================
    size_t total_count = query_vec.size();
    size_t processed_count = 0;
    auto start_time = steady_clock::now();

    std::cout << "开始提取ORB特征 (总共 " << total_count << " 帧)..." << std::endl;

    std::vector<std::string> valid_extensions = {
        ".jpg", ".jpeg", ".png", ".bmp",
        ".JPG", ".JPEG", ".PNG", ".BMP"
    };

    for (const auto& entry : fs::directory_iterator(images_folder)) {
        if (!entry.is_regular_file()) continue;

        std::string img_path = entry.path().string();
        std::string img_extension = entry.path().extension().string();

        bool is_valid = false;
        for (const auto& ext : valid_extensions) {
            if (img_extension == ext) {
                is_valid = true;
                break;
            }
        }
        if (!is_valid) continue;

        std::string frame_id_str = entry.path().stem().string();
        int frame_id;
        
        try {
            frame_id = std::stoi(frame_id_str);
        } catch (...) {
            // 跳过无法解析的文件名
            continue;
        }

        // ✅ 检查该图片是否在过滤后的frame_map中
        auto it = frame_map.find(frame_id);
        if (it == frame_map.end()) {
            // 该图片不在需要处理的范围内，跳过
            continue;
        }

        size_t idx = it->second;
        
        if (!query_vec[idx].descriptors_size1.empty()) {
            std::cerr << "警告：frame_id " << frame_id << " 存在多个图片文件" << std::endl;
            continue;
        }

        try {
            query_vec[idx].descriptors_size1 = extractOrbFeatures(img_path, 1);
            query_vec[idx].descriptors_size2 = extractOrbFeatures(img_path, 2);
            query_vec[idx].descriptors_size4 = extractOrbFeatures(img_path, 4);
            query_vec[idx].image_path = img_path;  // 保存图像路径
        } catch (const std::exception& e) {
            throw std::runtime_error(
                "图片特征提取失败: " + std::string(e.what())
            );
        }

        processed_count++;
        updateProgressBar(processed_count, total_count, start_time);
    }

    // ==================== 步骤5：验证完整性 ====================
    if (processed_count != total_count) {
        std::vector<int> missing_frames;
        for (const auto& data : query_vec) {
            if (data.descriptors_size1.empty()) {
                missing_frames.push_back(data.frame_id);
            }
        }

        std::string missing_str;
        size_t show_count = std::min(missing_frames.size(), size_t(10));
        for (size_t i = 0; i < show_count; ++i) {
            missing_str += std::to_string(missing_frames[i]);
            if (i < show_count - 1) missing_str += ", ";
        }
        if (missing_frames.size() > 10) {
            missing_str += " ...";
        }

        throw std::runtime_error(
            "图片数量与CSV记录数量不匹配\n缺失的frame_id: " + missing_str
        );
    }

    std::cout << "========================================" << std::endl;
    std::cout << "查询数据处理完成！总帧数: " << query_vec.size() << std::endl;
    std::cout << "========================================" << std::endl;

    return query_vec;
}