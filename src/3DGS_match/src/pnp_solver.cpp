#include "3DGS_match/pnp_solver.h"
#include <iostream>

PnPSolver::PnPSolver(
    int min_inliers,
    double ransac_reproj_error,
    double ransac_confidence,
    int ransac_iterations
) : min_inliers_(min_inliers),
    ransac_reproj_error_(ransac_reproj_error),
    ransac_confidence_(ransac_confidence),
    ransac_iterations_(ransac_iterations),
    verbose_(false)
{
    // 初始化ORB检测器（使用较多的特征点以提高匹配质量）
    orb_detector_ = cv::ORB::create(2000);

    // 初始化暴力匹配器（使用汉明距离）
    matcher_ = cv::BFMatcher::create(cv::NORM_HAMMING, true);
}

bool PnPSolver::solvePnP(
    const VisualQueryData& query_data,
    const VisualMapData& matched_map_data,
    const cv::Mat& camera_matrix,
    const cv::Mat& dist_coeffs,
    double& refined_x,
    double& refined_y,
    double& refined_z,
    std::array<std::array<double, 3>, 3>& refined_rotation,
    int& num_inliers
) {
    // 1. 加载图像
    cv::Mat query_image = cv::imread(query_data.image_path, cv::IMREAD_GRAYSCALE);
    cv::Mat map_image = cv::imread(matched_map_data.image_path, cv::IMREAD_GRAYSCALE);

    if (query_image.empty() || map_image.empty()) {
        if (verbose_) {
            std::cerr << "PnP错误: 无法加载图像" << std::endl;
        }
        return false;
    }

    // 2. 提取并匹配特征
    std::vector<cv::KeyPoint> query_keypoints, map_keypoints;
    std::vector<cv::DMatch> good_matches;

    int num_matches = extractAndMatchFeatures(
        query_image, map_image,
        query_keypoints, map_keypoints,
        good_matches
    );

    if (verbose_) {
        std::cout << "  特征匹配数量: " << num_matches << std::endl;
    }

    // 检查匹配数量是否足够
    if (num_matches < min_inliers_) {
        if (verbose_) {
            std::cout << "  PnP跳过: 匹配数量不足 (" << num_matches
                      << " < " << min_inliers_ << ")" << std::endl;
        }
        return false;
    }

    // 3. 重建地图帧的3D点
    cv::Point3d map_translation(
        matched_map_data.x,
        matched_map_data.y,
        matched_map_data.z
    );

    std::vector<cv::Point3f> object_points = reconstruct3DPoints(
        map_keypoints,
        camera_matrix,
        matched_map_data.rotation,
        map_translation,
        1.0  // 假设深度为1米
    );

    // 4. 提取查询帧的2D点
    std::vector<cv::Point2f> image_points;
    std::vector<cv::Point3f> matched_object_points;

    for (const auto& match : good_matches) {
        image_points.push_back(query_keypoints[match.queryIdx].pt);
        matched_object_points.push_back(object_points[match.trainIdx]);
    }

    // 5. 使用PnP RANSAC求解位姿
    cv::Mat rvec, tvec, inliers_mask;

    bool success = cv::solvePnPRansac(
        matched_object_points,  // 3D点
        image_points,           // 2D点
        camera_matrix,          // 相机内参
        dist_coeffs,            // 畸变系数
        rvec,                   // 输出：旋转向量
        tvec,                   // 输出：平移向量
        false,                  // useExtrinsicGuess
        ransac_iterations_,     // 迭代次数
        ransac_reproj_error_,   // 重投影误差阈值
        ransac_confidence_,     // 置信度
        inliers_mask            // 输出：内点掩码
    );

    if (!success) {
        if (verbose_) {
            std::cout << "  PnP求解失败" << std::endl;
        }
        return false;
    }

    // 6. 统计内点数量
    num_inliers = cv::countNonZero(inliers_mask);

    if (verbose_) {
        std::cout << "  PnP内点数量: " << num_inliers << "/" << num_matches << std::endl;
    }

    // 检查内点数量是否足够
    if (num_inliers < min_inliers_) {
        if (verbose_) {
            std::cout << "  PnP跳过: 内点数量不足 (" << num_inliers
                      << " < " << min_inliers_ << ")" << std::endl;
        }
        return false;
    }

    // 7. 转换为位姿
    convertRvecTvecToPose(rvec, tvec, refined_x, refined_y, refined_z, refined_rotation);

    if (verbose_) {
        std::cout << "  PnP求解成功!" << std::endl;
        std::cout << "    位置: (" << refined_x << ", " << refined_y << ", " << refined_z << ")" << std::endl;
    }

    return true;
}

int PnPSolver::extractAndMatchFeatures(
    const cv::Mat& query_image,
    const cv::Mat& map_image,
    std::vector<cv::KeyPoint>& query_keypoints,
    std::vector<cv::KeyPoint>& map_keypoints,
    std::vector<cv::DMatch>& good_matches
) {
    // 提取特征
    cv::Mat query_descriptors, map_descriptors;

    orb_detector_->detectAndCompute(query_image, cv::noArray(),
                                     query_keypoints, query_descriptors);
    orb_detector_->detectAndCompute(map_image, cv::noArray(),
                                     map_keypoints, map_descriptors);

    if (query_descriptors.empty() || map_descriptors.empty()) {
        return 0;
    }

    // 匹配特征
    std::vector<cv::DMatch> matches;
    matcher_->match(query_descriptors, map_descriptors, matches);

    if (matches.empty()) {
        return 0;
    }

    // 筛选良好的匹配（使用距离阈值）
    double min_dist = 100.0;
    for (const auto& match : matches) {
        if (match.distance < min_dist) {
            min_dist = match.distance;
        }
    }

    double distance_threshold = std::max(2.0 * min_dist, 30.0);

    for (const auto& match : matches) {
        if (match.distance <= distance_threshold) {
            good_matches.push_back(match);
        }
    }

    return good_matches.size();
}

std::vector<cv::Point3f> PnPSolver::reconstruct3DPoints(
    const std::vector<cv::KeyPoint>& map_keypoints,
    const cv::Mat& camera_matrix,
    const std::array<std::array<double, 3>, 3>& map_pose_rotation,
    const cv::Point3d& map_pose_translation,
    double assumed_depth
) {
    std::vector<cv::Point3f> points_3d;
    points_3d.reserve(map_keypoints.size());

    // 提取相机内参
    double fx = camera_matrix.at<double>(0, 0);
    double fy = camera_matrix.at<double>(1, 1);
    double cx = camera_matrix.at<double>(0, 2);
    double cy = camera_matrix.at<double>(1, 2);

    // 构建旋转矩阵
    cv::Mat R = (cv::Mat_<double>(3, 3) <<
        map_pose_rotation[0][0], map_pose_rotation[0][1], map_pose_rotation[0][2],
        map_pose_rotation[1][0], map_pose_rotation[1][1], map_pose_rotation[1][2],
        map_pose_rotation[2][0], map_pose_rotation[2][1], map_pose_rotation[2][2]
    );

    cv::Mat t = (cv::Mat_<double>(3, 1) <<
        map_pose_translation.x,
        map_pose_translation.y,
        map_pose_translation.z
    );

    // 对每个关键点进行反投影
    for (const auto& kp : map_keypoints) {
        // 像素坐标转相机坐标（归一化平面）
        double x_norm = (kp.pt.x - cx) / fx;
        double y_norm = (kp.pt.y - cy) / fy;

        // 相机坐标系下的3D点（假设深度）
        cv::Mat point_camera = (cv::Mat_<double>(3, 1) <<
            x_norm * assumed_depth,
            y_norm * assumed_depth,
            assumed_depth
        );

        // 转换到世界坐标系
        cv::Mat point_world = R * point_camera + t;

        points_3d.emplace_back(
            point_world.at<double>(0),
            point_world.at<double>(1),
            point_world.at<double>(2)
        );
    }

    return points_3d;
}

void PnPSolver::convertRvecTvecToPose(
    const cv::Mat& rvec,
    const cv::Mat& tvec,
    double& x, double& y, double& z,
    std::array<std::array<double, 3>, 3>& rotation
) {
    // 旋转向量转旋转矩阵
    cv::Mat R_cam2world;
    cv::Rodrigues(rvec, R_cam2world);

    // solvePnP返回的是世界坐标系到相机坐标系的变换
    // 我们需要相机在世界坐标系中的位姿

    // 相机在世界坐标系中的旋转矩阵 = R^T
    cv::Mat R_world2cam = R_cam2world.t();

    // 相机在世界坐标系中的位置 = -R^T * t
    cv::Mat camera_position = -R_world2cam * tvec;

    // 提取旋转矩阵
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            rotation[i][j] = R_world2cam.at<double>(i, j);
        }
    }

    // 提取相机位置
    x = camera_position.at<double>(0);
    y = camera_position.at<double>(1);
    z = camera_position.at<double>(2);
}
