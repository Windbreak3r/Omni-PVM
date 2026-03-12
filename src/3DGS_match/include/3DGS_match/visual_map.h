#ifndef VISUAL_MAP_H
#define VISUAL_MAP_H

#include <string>
#include <vector>
#include <array>
#include <cmath>
#include <chrono>
#include <tuple>      // ✅ 添加这个
#include <opencv2/opencv.hpp>

struct VisualMapData {
    int frame_id;
    double x, y, z;
    std::array<std::array<double, 3>, 3> rotation;
    std::string descriptors_size1;
    std::string descriptors_size2;
    std::string descriptors_size4;
    std::string image_path;  // 图像路径（用于PnP求解）

    VisualMapData();
    VisualMapData(
        int frame_id,
        double x, double y, double z,
        const std::array<std::array<double, 3>, 3>& rot,
        const std::string& desc1,
        const std::string& desc2,
        const std::string& desc4
    );

    double getYawAngle() const;
    double getYawDegrees() const;

    // ========== 公开接口 ==========
    
    // 从CSV构建视觉地图
    static std::vector<VisualMapData> buildVisualMapFromCSV(
        const std::string& csv_path,
        const std::string& images_folder
    );

    // 保存地图到二进制文件
    static void saveMapToFile(
        const std::vector<VisualMapData>& map_vec,
        const std::string& output_path
    );

    // 从二进制文件加载地图
    static std::vector<VisualMapData> loadMapFromFile(
        const std::string& input_path
    );

    // 保存地图到文本文件（人类可读，便于调试）
    static void saveMapToText(
        const std::vector<VisualMapData>& map_vec,
        const std::string& output_path
    );

private:
    // ========== 私有辅助函数 ==========
    
    // 从图片路径提取ORB特征（旧版本，单独处理）
    static std::string extractOrbFeatures(const std::string& image_path, int n);
    
    // 从已加载的图像提取ORB特征（新版本，避免重复读取）
    static std::string extractOrbFeaturesFromMat(const cv::Mat& image, int n);
    
    // 一次性提取三种尺度的特征（优化版本）
    static std::tuple<std::string, std::string, std::string> extractAllOrbFeatures(
        const std::string& image_path
    );
};

// ========== 全局辅助函数 ==========
void updateProgressBar(
    size_t processed, 
    size_t total, 
    std::chrono::steady_clock::time_point startTime
);

#endif // VISUAL_MAP_H