#include "3DGS_match/visual_map.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <bitset>
#include <filesystem>
#include <stdexcept>
#include <algorithm>
#include <unordered_map>

#include <thread>
#include <mutex>
#include <atomic>

namespace fs = std::filesystem;
using std::chrono::steady_clock;
using std::chrono::duration;

// ============================================================================
// 进度条函数
// ============================================================================

void updateProgressBar(size_t processed, size_t total, steady_clock::time_point startTime) {
    if (total == 0) return;

    double progress = static_cast<double>(processed) / total * 100.0;
    auto elapsedTime = steady_clock::now() - startTime;
    double elapsedSeconds = duration<double>(elapsedTime).count();
    
    size_t safeProcesed = (processed == 0) ? 1 : processed;
    double estimatedTotalTime = elapsedSeconds / safeProcesed * total;
    double remainingTime = estimatedTotalTime - elapsedSeconds;
    if (remainingTime < 0) remainingTime = 0;

    int remainingMinutes = static_cast<int>(remainingTime) / 60;
    int remainingSeconds = static_cast<int>(remainingTime) % 60;

    std::cout << "\r[";
    int barWidth = 50;
    int completed = static_cast<int>(progress / 100.0 * barWidth);
    
    for (int i = 0; i < barWidth; ++i) {
        std::cout << (i < completed ? "=" : " ");
    }
    std::cout << "] " << std::fixed << std::setprecision(1) << progress << "% "
              << "(" << processed << "/" << total << ") "
              << "剩余: " << remainingMinutes << "m " << remainingSeconds << "s   ";
    
    std::cout.flush();
    if (processed == total) {
        std::cout << std::endl;
    }
}

// ============================================================================
// VisualMapData 实现
// ============================================================================

// 默认构造函数
VisualMapData::VisualMapData() 
    : frame_id(-1), x(0.0), y(0.0), z(0.0), 
      rotation{{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}}} {}

// 完整构造函数
VisualMapData::VisualMapData(
    int frame_id,
    double x, double y, double z,
    const std::array<std::array<double, 3>, 3>& rot,
    const std::string& desc1,
    const std::string& desc2,
    const std::string& desc4
) : frame_id(frame_id), x(x), y(y), z(z), rotation(rot),
    descriptors_size1(desc1), descriptors_size2(desc2), descriptors_size4(desc4) {}

// 从旋转矩阵提取Yaw角（弧度）
double VisualMapData::getYawAngle() const {
    // yaw = atan2(r10, r00)
    return std::atan2(rotation[1][0], rotation[0][0]);
}

// 从旋转矩阵提取Yaw角（角度）
double VisualMapData::getYawDegrees() const {
    return getYawAngle() * 180.0 / M_PI;
}

std::string VisualMapData::extractOrbFeatures(const std::string& image_path, int n) {
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

// 从CSV构建视觉地图
std::vector<VisualMapData> VisualMapData::buildVisualMapFromCSV(
    const std::string& csv_path,
    const std::string& images_folder) {
    
    std::cout << "========================================" << std::endl;
    std::cout << "开始构建视觉地图" << std::endl;
    std::cout << "CSV文件: " << csv_path << std::endl;
    std::cout << "图片目录: " << images_folder << std::endl;
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

    std::vector<VisualMapData> map_vec;
    std::string line;
    size_t line_number = 1;

    while (std::getline(csv_file, line)) {
        line_number++;
        
        // 跳过空行
        if (line.empty() || line.find_first_not_of(" \t\r\n") == std::string::npos) {
            continue;
        }

        std::istringstream ss(line);
        std::string token;
        std::vector<std::string> tokens;

        while (std::getline(ss, token, ',')) {
            // 去除首尾空白
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
                std::to_string(tokens.size()) + "列\n内容: " + line
            );
        }

        try {
            int frame_id = std::stoi(tokens[0]);
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

            map_vec.emplace_back(frame_id, x, y, z, rot, "", "", "");
        } catch (const std::exception& e) {
            throw std::runtime_error(
                "CSV第" + std::to_string(line_number) + "行数据转换失败: " + e.what() + 
                "\n内容: " + line
            );
        }
    }
    csv_file.close();

    if (map_vec.empty()) {
        throw std::runtime_error("CSV文件无有效数据: " + csv_path);
    }

    std::cout << "成功读取 " << map_vec.size() << " 条位姿记录" << std::endl;

    // ==================== 步骤2：验证图片文件夹 ====================
    if (!fs::exists(images_folder)) {
        throw std::runtime_error("图片文件夹不存在: " + images_folder);
    }
    if (!fs::is_directory(images_folder)) {
        throw std::runtime_error("路径不是文件夹: " + images_folder);
    }

    // ==================== 步骤3：建立frame_id到索引的映射 ====================
    std::unordered_map<int, size_t> frame_map;
    for (size_t i = 0; i < map_vec.size(); ++i) {
        int fid = map_vec[i].frame_id;
        if (frame_map.count(fid) > 0) {
            throw std::runtime_error("CSV存在重复的frame_id: " + std::to_string(fid));
        }
        frame_map[fid] = i;
    }


    // ==================== 步骤4：并行提取特征 ====================
    size_t total_count = map_vec.size();
    std::atomic<size_t> processed_count(0);
    auto start_time = steady_clock::now();

    std::cout << "开始提取ORB特征..." << std::endl;

    // 支持的图片格式
    std::vector<std::string> valid_extensions = {
        ".jpg", ".jpeg", ".png", ".bmp",
        ".JPG", ".JPEG", ".PNG", ".BMP"
    };

    // 收集所有图片任务
    struct ImageTask {
        std::string img_path;
        int frame_id;
        size_t map_idx;
    };

    std::vector<ImageTask> image_tasks;
    image_tasks.reserve(total_count);

    for (const auto& entry : fs::directory_iterator(images_folder)) {
        if (!entry.is_regular_file()) {
            continue;
        }

        std::string img_path = entry.path().string();
        std::string img_extension = entry.path().extension().string();

        // 检查是否为支持的图片格式
        bool is_valid = false;
        for (const auto& ext : valid_extensions) {
            if (img_extension == ext) {
                is_valid = true;
                break;
            }
        }
        if (!is_valid) {
            continue;
        }

        // 提取文件名作为frame_id
        std::string frame_id_str = entry.path().stem().string();
        int frame_id;
        
        try {
            frame_id = std::stoi(frame_id_str);
        } catch (const std::exception& e) {
            std::cerr << "警告：无法解析图片frame_id: " << frame_id_str 
                    << " (跳过此文件)" << std::endl;
            continue;
        }

        // 查找对应的数据结构
        auto it = frame_map.find(frame_id);
        if (it == frame_map.end()) {
            throw std::runtime_error(
                "图片 " + entry.path().filename().string() + 
                " 在CSV中找不到对应位姿 (frame_id=" + std::to_string(frame_id) + ")"
            );
        }

        size_t idx = it->second;

        // 检查是否已处理过
        if (!map_vec[idx].descriptors_size1.empty()) {
            std::cerr << "警告：frame_id " << frame_id << " 存在多个图片文件" << std::endl;
            continue;
        }

        image_tasks.push_back({img_path, frame_id, idx});
    }

    if (image_tasks.size() != total_count) {
        throw std::runtime_error(
            "图片数量(" + std::to_string(image_tasks.size()) + 
            ")与CSV记录数量(" + std::to_string(total_count) + ")不匹配"
        );
    }

    // 多线程处理
    std::mutex map_mutex;
    std::mutex progress_mutex;
    std::mutex error_mutex;

    // 获取CPU核心数
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;  // 默认4线程
    if (num_threads > 16) num_threads = 16; // 限制最大线程数

    std::cout << "使用 " << num_threads << " 个线程进行并行处理" << std::endl;

    // 工作线程函数
    auto worker = [&](size_t start_idx,size_t end_idx) {
        for (size_t i = start_idx; i < end_idx; ++i) {
            const auto& task = image_tasks[i];
            
            try {
                // 提取三种尺度的特征（一次性读取图片）
                auto [desc1, desc2, desc4] = extractAllOrbFeatures(task.img_path);

                // 加锁写入结果
                {
                    std::lock_guard<std::mutex> lock(map_mutex);
                    map_vec[task.map_idx].descriptors_size1 = std::move(desc1);
                    map_vec[task.map_idx].descriptors_size2 = std::move(desc2);
                    map_vec[task.map_idx].descriptors_size4 = std::move(desc4);
                    map_vec[task.map_idx].image_path = task.img_path;  // 保存图像路径
                }

            } catch (const std::exception& e) {
                std::lock_guard<std::mutex> lock(error_mutex);
                std::cerr << "\n错误：图片 " << task.img_path 
                        << " (frame_id=" << task.frame_id << ") 处理失败: " 
                        << e.what() << std::endl;
                throw;  // 重新抛出异常
            }

            // 更新进度（每处理一个就更新）
            size_t current = ++processed_count;
            
            // 定期更新进度条（避免过于频繁的锁竞争）
            if (current % 10 == 0 || current == total_count) {
                std::lock_guard<std::mutex> lock(progress_mutex);
                updateProgressBar(current, total_count, start_time);
            }
        }
    };

    // 启动线程
    std::vector<std::thread> threads;
    size_t tasks_per_thread = image_tasks.size() / num_threads;
    size_t remaining_tasks = image_tasks.size() % num_threads;

    size_t start_idx = 0;
    for (unsigned int t = 0; t < num_threads; ++t) {
        // 均匀分配任务，最后一个线程处理剩余任务
        size_t current_tasks = tasks_per_thread + (t < remaining_tasks ? 1 : 0);
        size_t end_idx = start_idx + current_tasks;
        
        threads.emplace_back(worker, start_idx, end_idx);
        start_idx = end_idx;
    }

    // 等待所有线程完成
    for (auto& thread : threads) {
        thread.join();
    }

    // 确保进度条显示100%
    {
        std::lock_guard<std::mutex> lock(progress_mutex);
        updateProgressBar(total_count, total_count, start_time);
    }

    std::cout << std::endl;

    // ==================== 步骤5：验证完整性 ====================
    if (processed_count != total_count) {
        // 找出缺失的frame_id
        std::vector<int> missing_frames;
        for (const auto& data : map_vec) {
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
            missing_str += " ... (共" + std::to_string(missing_frames.size()) + "个)";
        }

        throw std::runtime_error(
            "图片数量(" + std::to_string(processed_count) + 
            ")与CSV记录数量(" + std::to_string(total_count) + ")不匹配\n" +
            "缺失的frame_id: " + missing_str
        );
    }

    std::cout << "========================================" << std::endl;
    std::cout << "视觉地图构建完成！" << std::endl;
    std::cout << "总帧数: " << map_vec.size() << std::endl;
    std::cout << "========================================" << std::endl;

    return map_vec;
}

// ============================================================================
// 地图缓存功能
// ============================================================================

// 保存地图到二进制文件（高效，文件较小）
void VisualMapData::saveMapToFile(
    const std::vector<VisualMapData>& map_vec,
    const std::string& output_path) {
    
    std::cout << "\n保存视觉地图到: " << output_path << std::endl;
    
    std::ofstream out(output_path, std::ios::binary);
    if (!out.is_open()) {
        throw std::runtime_error("无法创建文件: " + output_path);
    }

    // 写入魔数和版本号
    uint32_t magic = 0x564D4150;  // "VMAP"
    uint32_t version = 1;
    out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    out.write(reinterpret_cast<const char*>(&version), sizeof(version));

    // 写入地图大小
    size_t map_size = map_vec.size();
    out.write(reinterpret_cast<const char*>(&map_size), sizeof(map_size));

    // 逐帧写入
    for (const auto& data : map_vec) {
        // frame_id
        out.write(reinterpret_cast<const char*>(&data.frame_id), sizeof(data.frame_id));
        
        // 位置
        out.write(reinterpret_cast<const char*>(&data.x), sizeof(data.x));
        out.write(reinterpret_cast<const char*>(&data.y), sizeof(data.y));
        out.write(reinterpret_cast<const char*>(&data.z), sizeof(data.z));
        
        // 旋转矩阵
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                out.write(reinterpret_cast<const char*>(&data.rotation[i][j]), 
                         sizeof(data.rotation[i][j]));
            }
        }
        
        // 描述符（写入字符串长度+内容）
        auto writeString = [&](const std::string&str) {
            size_t len = str.length();
            out.write(reinterpret_cast<const char*>(&len), sizeof(len));
            out.write(str.c_str(), len);
        };
        
        writeString(data.descriptors_size1);
        writeString(data.descriptors_size2);
        writeString(data.descriptors_size4);
    }

    out.close();
    
    // 获取文件大小
    size_t file_size = fs::file_size(output_path);
    double size_mb = file_size / (1024.0 * 1024.0);
    
    std::cout << "保存完成！" << std::endl;
    std::cout << "  文件大小: " << std::fixed << std::setprecision(2) << size_mb << " MB" << std::endl;
    std::cout << "  总帧数: " << map_vec.size() << std::endl;
}

// 从二进制文件加载地图
std::vector<VisualMapData> VisualMapData::loadMapFromFile(const std::string& input_path) {
    std::cout << "\n从文件加载视觉地图: " << input_path << std::endl;

    std::ifstream in(input_path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("无法打开文件: " + input_path);
    }

    // 读取并验证魔数
    uint32_t magic, version;
    in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    in.read(reinterpret_cast<char*>(&version), sizeof(version));

    if (magic != 0x564D4150) {
        throw std::runtime_error("文件格式错误：不是有效的地图文件");
    }
    if (version != 1) {
        throw std::runtime_error("文件版本不支持: " + std::to_string(version));
    }

    // 读取地图大小
    size_t map_size;
    in.read(reinterpret_cast<char*>(&map_size), sizeof(map_size));

    std::cout << "加载 " << map_size << " 帧数据..." << std::endl;

    // 尝试从文件路径推断图像目录
    std::string images_folder;
    fs::path cache_path(input_path);
    fs::path parent_dir = cache_path.parent_path();

    // 尝试几个可能的图像目录
    std::vector<std::string> possible_dirs = {
        (parent_dir / "PVM" / "images_sparse").string(),
        (parent_dir / "PVM" / "images").string(),
        (parent_dir / "images_sparse").string(),
        (parent_dir / "images").string()
    };

    for (const auto& dir : possible_dirs) {
        if (fs::exists(dir) && fs::is_directory(dir)) {
            images_folder = dir;
            std::cout << "  检测到图像目录: " << images_folder << std::endl;
            break;
        }
    }

    std::vector<VisualMapData> map_vec;
    map_vec.reserve(map_size);

    auto start_time = steady_clock::now();

    // 逐帧读取
    for (size_t i = 0; i < map_size; ++i) {
        VisualMapData data;

        // frame_id
        in.read(reinterpret_cast<char*>(&data.frame_id), sizeof(data.frame_id));

        // 位置
        in.read(reinterpret_cast<char*>(&data.x), sizeof(data.x));
        in.read(reinterpret_cast<char*>(&data.y), sizeof(data.y));
        in.read(reinterpret_cast<char*>(&data.z), sizeof(data.z));

        // 旋转矩阵
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                in.read(reinterpret_cast<char*>(&data.rotation[r][c]),
                       sizeof(data.rotation[r][c]));
            }
        }

        // 描述符
        auto readString = [&]() -> std::string {
            size_t len;
            in.read(reinterpret_cast<char*>(&len), sizeof(len));
            std::string str(len, '\0');
            in.read(&str[0], len);
            return str;
        };

        data.descriptors_size1 = readString();
        data.descriptors_size2 = readString();
        data.descriptors_size4 = readString();

        // 重建image_path（如果找到了图像目录）
        if (!images_folder.empty()) {
            // 尝试常见的图像文件扩展名
            std::vector<std::string> extensions = {".png", ".jpg", ".jpeg"};
            for (const auto& ext : extensions) {
                std::ostringstream oss;
                oss << std::setw(5) << std::setfill('0') << data.frame_id << ext;
                std::string img_path = (fs::path(images_folder) / oss.str()).string();

                if (fs::exists(img_path)) {
                    data.image_path = img_path;
                    break;
                }
            }
        }

        map_vec.push_back(std::move(data));

        // 进度显示
        if ((i + 1) % 100 == 0 || i == map_size - 1) {
            updateProgressBar(i + 1, map_size, start_time);
        }
    }

    in.close();

    std::cout << "\n加载完成！共 " << map_vec.size() << " 帧" << std::endl;

    // 统计有多少帧找到了图像路径
    size_t found_images = 0;
    for (const auto& data : map_vec) {
        if (!data.image_path.empty()) {
            found_images++;
        }
    }

    if (found_images > 0) {
        std::cout << "  成功关联图像路径: " << found_images << "/" << map_vec.size() << " 帧" << std::endl;
    } else {
        std::cout << "  警告: 未找到图像文件，PnP功能将不可用" << std::endl;
    }

    return map_vec;
}

// 保存地图到文本文件（可读，包含完整描述符前100帧）
void VisualMapData::saveMapToText(
    const std::vector<VisualMapData>& map_vec,
    const std::string& output_path) {
    
    std::cout << "\n保存地图到文本文件: " << output_path << std::endl;
    
    std::ofstream out(output_path);
    if (!out.is_open()) {
        throw std::runtime_error("无法创建文件: " + output_path);
    }

    // ==================== 文件头 ====================
    out << "========================================" << std::endl;
    out << "Visual Map Data (Text Format)" << std::endl;
    out << "========================================" << std::endl;
    out << "[Header]" << std::endl;
    out << "  Magic: 0x564D4150 (VMAP)" << std::endl;
    out << "  Version: 1" << std::endl;
    out << "  Total Frames: " << map_vec.size() << std::endl;
    out << "========================================" << std::endl;
    out << std::endl;

    // ==================== 前100帧详细信息 ====================
    size_t detail_count = std::min(size_t(100), map_vec.size());
    
    out << "# 详细信息（前 " << detail_count << " 帧，包含完整描述符）" << std::endl;
    out << std::endl;

    for (size_t i = 0; i < detail_count; ++i) {
        const auto& data = map_vec[i];
        
        out << "----------------------------------------" << std::endl;
        out << "[Frame " << i << "]" << std::endl;
        out << "  frame_id: " << data.frame_id << std::endl;
        
        // 位置信息
        out << "  x, y, z: " 
            << std::fixed << std::setprecision(6)
            << data.x << ", " << data.y << ", " << data.z << std::endl;
        
        // 旋转矩阵（格式化为3×3矩阵）
        out << "  rotation:" << std::endl;
        for (int r = 0; r < 3; ++r) {
            out << "    [";
            for (int c = 0; c < 3; ++c) {
                out << std::setw(12) << std::setprecision(6) << data.rotation[r][c];
                if (c < 2) out << ", ";
            }
            out << "]" << std::endl;
        }
        
        // Yaw角（额外信息）
        out << "  yaw_angle: " << std::fixed << std::setprecision(2) 
            << data.getYawDegrees() << "°" << std::endl;
        
        // 描述符详情
        out << "  desc1_len: " << data.descriptors_size1.length() << std::endl;
        out << "  desc1: \"" << data.descriptors_size1 << "\"" << std::endl;
        
        out << "  desc2_len: " << data.descriptors_size2.length() << std::endl;
        out << "  desc2: \"" << data.descriptors_size2 << "\"" << std::endl;
        
        out << "  desc4_len: " << data.descriptors_size4.length() << std::endl;
        out << "  desc4: \"" << data.descriptors_size4 << "\"" << std::endl;
        
        out << std::endl;
    }

    // ==================== 剩余帧概要信息 ====================
    if (map_vec.size() > detail_count) {
        out << "========================================" << std::endl;
        out << "# 概要信息（第 " << detail_count << " 到 " << map_vec.size() - 1 
            << " 帧，不含描述符）" << std::endl;
        out << "# Format: frame_id x y z r00 r01 r02 r10 r11 r12 r20 r21 r22 "
            << "desc1_len desc2_len desc4_len yaw_deg" << std::endl;
        out << "========================================" << std::endl;
        out << std::endl;

        for (size_t i = detail_count; i < map_vec.size(); ++i) {
            const auto& data = map_vec[i];
            
            out << data.frame_id << " "
                << std::fixed << std::setprecision(6)
                << data.x << " " << data.y << " " << data.z << " ";
            
            // 旋转矩阵（单行）
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    out << data.rotation[r][c] << " ";
                }
            }
            
            // 描述符长度
            out << data.descriptors_size1.length() << " "
                << data.descriptors_size2.length() << " "
                << data.descriptors_size4.length() << " ";
            
            // Yaw角
            out << std::setprecision(2) << data.getYawDegrees();
            
            out << std::endl;
        }
    }

    out.close();
    std::cout << "文本文件保存完成！" << std::endl;
    std::cout << "  详细信息: 前 " << detail_count << " 帧（包含完整描述符）" << std::endl;
    std::cout << "  概要信息: 第 " << detail_count << " 到 " << map_vec.size() - 1 << " 帧" << std::endl;
}

// ============================================================================
// 一次性提取三种尺度的ORB特征
// ============================================================================

std::tuple<std::string, std::string, std::string> VisualMapData::extractAllOrbFeatures(
    const std::string& image_path) {
    
    // 读取图片（只读一次）
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        throw std::runtime_error("无法读取图像: " + image_path);
    }

    // 如果图片太大，先缩小（可选优化）
    if (image.cols > 1920 || image.rows > 1080) {
        double scale = std::min(1920.0 / image.cols, 1080.0 / image.rows);
        cv::resize(image, image, cv::Size(), scale, scale, cv::INTER_AREA);
    }

    // 提取三种尺度的特征
    std::string desc1 = extractOrbFeaturesFromMat(image, 1);
    std::string desc2 = extractOrbFeaturesFromMat(image, 2);
    std::string desc4 = extractOrbFeaturesFromMat(image, 4);

    return {desc1, desc2, desc4};
}

// ============================================================================
// 从已加载的图像提取ORB特征
// ============================================================================

std::string VisualMapData::extractOrbFeaturesFromMat(const cv::Mat& image, int n) {
    int rows = image.rows;
    int cols = image.cols;
    
    if (rows < n || cols < n) {
        throw std::runtime_error("图像尺寸过小");
    }

    int partRows = rows / n;
    int partCols = cols / n;

    std::string binaryString;
    binaryString.reserve(n * n * 256);

    // ✅ 只创建一次ORB实例（优化1已完成）
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
                throw std::runtime_error("ORB特征提取失败");
            }

            // 转换为二进制字符串
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