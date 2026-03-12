#ifndef MATCH_H
#define MATCH_H

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "3DGS_match/HMM_2.h"
#include "3DGS_match/visual_map.h"
#include "3DGS_match/query_information.h"
#include "3DGS_match/pnp_solver.h"

// 全局变量声明
extern Viterbi viterbiInstance;
extern std::vector<VisualMapData> visualMapVec;
extern std::vector<VisualQueryData> queryDataVec;

// PnP配置结构
struct PnPConfig {
    bool enable = false;                    // 是否启用PnP
    int min_inliers = 20;                   // 最小内点数
    double ransac_reproj_error = 8.0;       // RANSAC重投影误差阈值
    cv::Mat camera_matrix;                  // 相机内参矩阵
    cv::Mat dist_coeffs;                    // 畸变系数
};

// 主匹配函数（使用已加载的数据）
void ViterbiMatch(
    const std::string& result_path,
    const std::string& backtrack_txt,
    const std::vector<VisualMapData>& mapData,
    const std::vector<VisualQueryData>& queryData,
    int target_frames = -1,           // 目标帧数，-1表示自动检测
    bool use_motion_prediction = true, // true: 运动预测模式, false: 完整CSV查询模式
    const PnPConfig* pnp_config = nullptr // PnP配置（可选）
);

// 主匹配函数（从CSV加载数据）
void ViterbiMatch(
    const std::string& result_path,
    const std::string& backtrack_txt,
    const std::string& map_csv_path,
    const std::string& map_images_folder,
    const std::string& query_csv_path,
    const std::string& query_images_folder
);

// 工具函数
double angleSimilarity(double angle1, double angle2);
double calculateMixedDistance(
    double map_x, double map_y, double map_yaw,
    double query_x, double query_y, double query_yaw
);


#endif // MATCH_H