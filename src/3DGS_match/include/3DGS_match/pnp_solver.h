#ifndef PNP_SOLVER_H
#define PNP_SOLVER_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <array>
#include "3DGS_match/visual_map.h"
#include "3DGS_match/query_information.h"

/**
 * @brief PnP位姿求解器
 *
 * 在稀疏地图场景下，通过特征匹配和PnP算法进一步优化位姿估计
 */
class PnPSolver {
public:
    /**
     * @brief 构造函数
     * @param min_inliers 最小内点数量阈值（默认20）
     * @param ransac_reproj_error RANSAC重投影误差阈值（默认8.0像素）
     * @param ransac_confidence RANSAC置信度（默认0.99）
     * @param ransac_iterations RANSAC最大迭代次数（默认100）
     */
    PnPSolver(
        int min_inliers = 20,
        double ransac_reproj_error = 8.0,
        double ransac_confidence = 0.99,
        int ransac_iterations = 100
    );

    /**
     * @brief 使用PnP求解位姿
     *
     * @param query_data 查询帧数据
     * @param matched_map_data 匹配的地图帧数据
     * @param camera_matrix 相机内参矩阵 (3x3)
     * @param dist_coeffs 畸变系数 (可选，默认为空)
     * @param refined_x 输出：优化后的x坐标
     * @param refined_y 输出：优化后的y坐标
     * @param refined_z 输出：优化后的z坐标
     * @param refined_rotation 输出：优化后的旋转矩阵 (3x3)
     * @param num_inliers 输出：内点数量
     * @return true 如果PnP求解成功，false 否则
     */
    bool solvePnP(
        const VisualQueryData& query_data,
        const VisualMapData& matched_map_data,
        const cv::Mat& camera_matrix,
        const cv::Mat& dist_coeffs,
        double& refined_x,
        double& refined_y,
        double& refined_z,
        std::array<std::array<double, 3>, 3>& refined_rotation,
        int& num_inliers
    );

    /**
     * @brief 设置是否启用详细输出
     */
    void setVerbose(bool verbose) { verbose_ = verbose; }

    /**
     * @brief 获取最小内点数阈值
     */
    int getMinInliers() const { return min_inliers_; }

private:
    /**
     * @brief 提取并匹配ORB特征
     *
     * @param query_image 查询图像
     * @param map_image 地图图像
     * @param query_keypoints 输出：查询图像关键点
     * @param map_keypoints 输出：地图图像关键点
     * @param good_matches 输出：良好的匹配对
     * @return 匹配数量
     */
    int extractAndMatchFeatures(
        const cv::Mat& query_image,
        const cv::Mat& map_image,
        std::vector<cv::KeyPoint>& query_keypoints,
        std::vector<cv::KeyPoint>& map_keypoints,
        std::vector<cv::DMatch>& good_matches
    );

    /**
     * @brief 从地图帧的关键点恢复3D点
     *
     * 假设地图帧的关键点深度为固定值（例如1.0米）
     * 在实际应用中，如果有深度信息可以使用真实深度
     *
     * @param map_keypoints 地图关键点
     * @param camera_matrix 相机内参
     * @param map_pose_rotation 地图帧旋转矩阵
     * @param map_pose_translation 地图帧平移向量
     * @param assumed_depth 假设的深度值（默认1.0米）
     * @return 3D点云
     */
    std::vector<cv::Point3f> reconstruct3DPoints(
        const std::vector<cv::KeyPoint>& map_keypoints,
        const cv::Mat& camera_matrix,
        const std::array<std::array<double, 3>, 3>& map_pose_rotation,
        const cv::Point3d& map_pose_translation,
        double assumed_depth = 1.0
    );

    /**
     * @brief 将旋转向量和平移向量转换为位姿
     */
    void convertRvecTvecToPose(
        const cv::Mat& rvec,
        const cv::Mat& tvec,
        double& x, double& y, double& z,
        std::array<std::array<double, 3>, 3>& rotation
    );

private:
    int min_inliers_;              // 最小内点数阈值
    double ransac_reproj_error_;   // RANSAC重投影误差阈值
    double ransac_confidence_;     // RANSAC置信度
    int ransac_iterations_;        // RANSAC最大迭代次数
    bool verbose_;                 // 是否输出详细信息

    cv::Ptr<cv::ORB> orb_detector_; // ORB特征检测器
    cv::Ptr<cv::BFMatcher> matcher_; // 暴力匹配器
};

#endif // PNP_SOLVER_H
