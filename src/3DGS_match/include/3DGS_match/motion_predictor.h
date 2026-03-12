#ifndef MOTION_PREDICTOR_H
#define MOTION_PREDICTOR_H

#include <vector>
#include <deque>
#include <Eigen/Dense>
#include "3DGS_match/query_information.h"

/**
 * @brief 运动学预测器
 *
 * 基于历史位姿预测当前帧的位姿，用于粗匹配
 */
class MotionPredictor {
public:
    /**
     * @brief 构造函数
     * @param window_size 预测窗口大小（使用多少历史帧）
     * @param init_frames 初始化帧数（前N帧直接使用原始位姿）
     */
    MotionPredictor(int window_size = 5, int init_frames = 5);

    /**
     * @brief 添加新的位姿观测
     * @param query_data 查询帧数据
     */
    void addPose(const VisualQueryData& query_data);

    /**
     * @brief 预测下一帧位姿
     * @param query_data 当前查询帧（包含原始位姿）
     * @return 预测的位姿（如果在初始化阶段，返回原始位姿）
     */
    VisualQueryData predictPose(const VisualQueryData& query_data);

    /**
     * @brief 重置预测器
     */
    void reset();

    /**
     * @brief 获取当前帧索引
     */
    int getCurrentFrameIndex() const { return frame_count_; }

    /**
     * @brief 是否处于初始化阶段
     */
    bool isInitializing() const { return frame_count_ < init_frames_; }

private:
    struct Pose {
        double x, y, z;
        Eigen::Matrix3d rotation;
        int frame_id;
    };

    int window_size_;      // 预测窗口大小
    int init_frames_;      // 初始化帧数
    int frame_count_;      // 当前帧计数

    std::deque<Pose> pose_history_;  // 位姿历史

    /**
     * @brief 从旋转矩阵提取位姿
     */
    Pose extractPose(const VisualQueryData& query_data);

    /**
     * @brief 计算平均平移速度
     */
    Eigen::Vector3d computeAverageVelocity();

    /**
     * @brief 计算平均旋转速度
     */
    Eigen::Matrix3d computeAverageRotationDelta();

    /**
     * @brief 应用运动模型预测
     */
    Pose applyMotionModel(const Pose& last_pose,
                         const Eigen::Vector3d& velocity,
                         const Eigen::Matrix3d& rotation_delta);
};

#endif // MOTION_PREDICTOR_H
