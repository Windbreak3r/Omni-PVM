#include "3DGS_match/motion_predictor.h"
#include <iostream>
#include <cmath>

MotionPredictor::MotionPredictor(int window_size, int init_frames)
    : window_size_(window_size)
    , init_frames_(init_frames)
    , frame_count_(0) {
}

void MotionPredictor::reset() {
    pose_history_.clear();
    frame_count_ = 0;
}

MotionPredictor::Pose MotionPredictor::extractPose(const VisualQueryData& query_data) {
    Pose pose;
    pose.x = query_data.x;
    pose.y = query_data.y;
    pose.z = query_data.z;
    pose.frame_id = query_data.frame_id;

    // 从query_data的旋转矩阵构建Eigen矩阵
    pose.rotation << query_data.rotation[0][0], query_data.rotation[0][1], query_data.rotation[0][2],
                     query_data.rotation[1][0], query_data.rotation[1][1], query_data.rotation[1][2],
                     query_data.rotation[2][0], query_data.rotation[2][1], query_data.rotation[2][2];

    return pose;
}

void MotionPredictor::addPose(const VisualQueryData& query_data) {
    Pose pose = extractPose(query_data);
    pose_history_.push_back(pose);

    // 保持窗口大小
    if (pose_history_.size() > static_cast<size_t>(window_size_)) {
        pose_history_.pop_front();
    }

    frame_count_++;
}

Eigen::Vector3d MotionPredictor::computeAverageVelocity() {
    if (pose_history_.size() < 2) {
        return Eigen::Vector3d::Zero();
    }

    Eigen::Vector3d total_velocity = Eigen::Vector3d::Zero();
    int count = 0;

    for (size_t i = 1; i < pose_history_.size(); ++i) {
        Eigen::Vector3d delta(
            pose_history_[i].x - pose_history_[i-1].x,
            pose_history_[i].y - pose_history_[i-1].y,
            pose_history_[i].z - pose_history_[i-1].z
        );
        total_velocity += delta;
        count++;
    }

    if (count > 0) {
        return total_velocity / count;
    }
    return Eigen::Vector3d::Zero();
}

Eigen::Matrix3d MotionPredictor::computeAverageRotationDelta() {
    if (pose_history_.size() < 2) {
        return Eigen::Matrix3d::Identity();
    }

    // 简化方法：计算相对旋转矩阵的平均
    // 更精确的方法应该使用李代数或四元数平均
    std::vector<Eigen::Matrix3d> rotation_deltas;

    for (size_t i = 1; i < pose_history_.size(); ++i) {
        // 相对旋转: R_delta = R_i * R_{i-1}^T
        Eigen::Matrix3d delta = pose_history_[i].rotation *
                                pose_history_[i-1].rotation.transpose();
        rotation_deltas.push_back(delta);
    }

    if (rotation_deltas.empty()) {
        return Eigen::Matrix3d::Identity();
    }

    // 简单平均（近似）
    // 更好的方法是使用四元数平均或李代数
    Eigen::Matrix3d avg_delta = Eigen::Matrix3d::Zero();
    for (const auto& delta : rotation_deltas) {
        avg_delta += delta;
    }
    avg_delta /= rotation_deltas.size();

    // 正交化（确保是有效的旋转矩阵）
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(avg_delta, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    Eigen::Matrix3d orthogonal = U * V.transpose();

    // 确保行列式为1（右手坐标系）
    if (orthogonal.determinant() < 0) {
        U.col(2) *= -1;
        orthogonal = U * V.transpose();
    }

    return orthogonal;
}

MotionPredictor::Pose MotionPredictor::applyMotionModel(
    const Pose& last_pose,
    const Eigen::Vector3d& velocity,
    const Eigen::Matrix3d& rotation_delta) {

    Pose predicted_pose;
    predicted_pose.frame_id = last_pose.frame_id + 1;

    // 预测位置
    predicted_pose.x = last_pose.x + velocity(0);
    predicted_pose.y = last_pose.y + velocity(1);
    predicted_pose.z = last_pose.z + velocity(2);

    // 预测旋转
    predicted_pose.rotation = rotation_delta * last_pose.rotation;

    // 正交化（确保数值稳定性）
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(predicted_pose.rotation,
                                          Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    predicted_pose.rotation = U * V.transpose();

    if (predicted_pose.rotation.determinant() < 0) {
        U.col(2) *= -1;
        predicted_pose.rotation = U * V.transpose();
    }

    return predicted_pose;
}

VisualQueryData MotionPredictor::predictPose(const VisualQueryData& query_data) {
    // 如果在初始化阶段，直接返回原始位姿
    if (isInitializing()) {
        std::cout << "Frame " << query_data.frame_id
                  << ": 初始化阶段，使用原始位姿" << std::endl;
        return query_data;
    }

    // 如果历史不足，返回原始位姿
    if (pose_history_.size() < 2) {
        std::cout << "Frame " << query_data.frame_id
                  << ": 历史不足，使用原始位姿" << std::endl;
        return query_data;
    }

    // 计算运动模型
    Eigen::Vector3d velocity = computeAverageVelocity();
    Eigen::Matrix3d rotation_delta = computeAverageRotationDelta();

    // 预测位姿
    Pose last_pose = pose_history_.back();
    Pose predicted_pose = applyMotionModel(last_pose, velocity, rotation_delta);

    // 计算预测误差（用于调试）
    double pred_error = std::sqrt(
        std::pow(predicted_pose.x - query_data.x, 2) +
        std::pow(predicted_pose.y - query_data.y, 2) +
        std::pow(predicted_pose.z - query_data.z, 2)
    );

    std::cout << "Frame " << query_data.frame_id
              << ": 预测误差=" << pred_error << "m, "
              << "预测位置=(" << predicted_pose.x << ", "
              << predicted_pose.y << ", " << predicted_pose.z << ")" << std::endl;

    // 创建预测的查询数据
    VisualQueryData predicted_query = query_data;  // 复制其他字段
    predicted_query.x = predicted_pose.x;
    predicted_query.y = predicted_pose.y;
    predicted_query.z = predicted_pose.z;

    // 更新旋转矩阵
    predicted_query.rotation[0][0] = predicted_pose.rotation(0, 0);
    predicted_query.rotation[0][1] = predicted_pose.rotation(0, 1);
    predicted_query.rotation[0][2] = predicted_pose.rotation(0, 2);
    predicted_query.rotation[1][0] = predicted_pose.rotation(1, 0);
    predicted_query.rotation[1][1] = predicted_pose.rotation(1, 1);
    predicted_query.rotation[1][2] = predicted_pose.rotation(1, 2);
    predicted_query.rotation[2][0] = predicted_pose.rotation(2, 0);
    predicted_query.rotation[2][1] = predicted_pose.rotation(2, 1);
    predicted_query.rotation[2][2] = predicted_pose.rotation(2, 2);

    return predicted_query;
}
