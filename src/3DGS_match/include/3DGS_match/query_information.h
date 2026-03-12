#ifndef VISUAL_QUERY_DATA_H
#define VISUAL_QUERY_DATA_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <array>

class VisualQueryData {
public:
    // 数据成员
    int frame_id;
    double x, y, z;
    std::array<std::array<double, 3>, 3> rotation;
    std::string descriptors_size1;
    std::string descriptors_size2;
    std::string descriptors_size4;
    std::string image_path;  // 图像路径（用于PnP求解）

    // 构造函数
    VisualQueryData();
    VisualQueryData(
        int frame_id,
        double x, double y, double z,
        const std::array<std::array<double, 3>, 3>& rot,
        const std::string& desc1 = "",
        const std::string& desc2 = "",
        const std::string& desc4 = ""
    );

    // 辅助函数
    double getYawAngle() const;
    double getYawDegrees() const;

    // ✅ 修改：添加范围过滤参数（默认值保持向后兼容）
    static std::vector<VisualQueryData> buildQueryFromCSV(
        const std::string& csv_path,
        const std::string& images_folder,
        bool enable_range_filter = false,
        int start_frame_id = 0,
        int end_frame_id = 0
    );

    static std::string extractOrbFeatures(const std::string& image_path, int n);
};

#endif // VISUAL_QUERY_DATA_H