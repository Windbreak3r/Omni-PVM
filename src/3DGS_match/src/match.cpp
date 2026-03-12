#include "3DGS_match/match.h"
#include "3DGS_match/motion_predictor.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <iomanip>

// ============================================================================
// 工具函数
// ============================================================================

double angleSimilarity(double angle1, double angle2) {
    auto normalize = [](double a) -> double {
        a = std::fmod(a, 360.0);
        return a < 0 ? a + 360.0 : a;
    };
    double delta = std::abs(normalize(angle1) - normalize(angle2));
    return 1.0 - std::min(delta, 360.0 - delta) / 180.0;
}

double calculateMixedDistance(
    double map_x, double map_y, double map_yaw,
    double query_x, double query_y, double query_yaw) {
    
    double dx = map_x - query_x;
    double dy = map_y - query_y;
    double xy_dist = std::sqrt(dx * dx + dy * dy);
    double theta_scaled = 0.9 + 0.1 * angleSimilarity(map_yaw, query_yaw);
    return xy_dist / theta_scaled;
}

// ============================================================================
// 读取回溯数据
// ============================================================================

std::vector<std::vector<BacktrackData>> readBacktrackDataFromFile(const std::string& filename) {
    std::vector<std::vector<BacktrackData>> data;
    std::ifstream file(filename);
    if (!file.is_open()) return data;

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        std::vector<BacktrackData> row;
        BacktrackData item;
        while (iss >> item.last_state_index >> item.probability) {
            row.push_back(item);
        }
        if (!row.empty()) data.push_back(row);
    }
    return data;
}

// ============================================================================
// 回溯转移概率计算
// ============================================================================

std::vector<double> calcBacktrackT1(
    int last_state,
    const std::vector<BacktrackData>& candidates,
    const std::vector<VisualMapData>& mapData) {
    
    std::vector<double> probs(candidates.size());
    double px = mapData[last_state].x;
    double py = mapData[last_state].y;
    double pyaw = mapData[last_state].getYawDegrees();

    double sum = 0.0;
    for (size_t i = 0; i < candidates.size(); ++i) {
        int idx = candidates[i].last_state_index;
        double dist = calculateMixedDistance(
            mapData[idx].x, mapData[idx].y, mapData[idx].getYawDegrees(),
            px, py, pyaw
        );
        probs[i] = std::exp(-std::pow(dist / 5.0, 2) / 2.0);
        sum += probs[i];
    }

    if (sum > 1e-10) {
        for (double& p : probs) p /= sum;
    }
    return probs;
}

std::vector<std::vector<std::vector<double>>> calcBacktrackT2(
    const std::vector<BacktrackData>& t2,
    const std::vector<BacktrackData>& t1,
    const std::vector<BacktrackData>& t0,
    const std::vector<VisualMapData>& mapData) {
    
    std::vector<std::vector<std::vector<double>>> probs(
        t2.size(), std::vector<std::vector<double>>(t1.size(), std::vector<double>(t0.size(), 0.0))
    );

    for (size_t i = 0; i < t2.size(); ++i) {
        int idx2 = t2[i].last_state_index;
        double sum = 0.0;
        
        for (size_t j = 0; j < t1.size(); ++j) {
            int idx1 = t1[j].last_state_index;
            double px = 2.0 * mapData[idx1].x - mapData[idx2].x;
            double py = 2.0 * mapData[idx1].y - mapData[idx2].y;
            double pyaw = 2.0 * mapData[idx1].getYawDegrees() - mapData[idx2].getYawDegrees();

            for (size_t k = 0; k < t0.size(); ++k) {
                int idx0 = t0[k].last_state_index;
                double dist = calculateMixedDistance(
                    mapData[idx0].x, mapData[idx0].y, mapData[idx0].getYawDegrees(),
                    px, py, pyaw
                );
                probs[i][j][k] = std::exp(-std::pow(dist / 5.0, 2) / 2.0);
                sum += probs[i][j][k];
            }
        }

        if (sum > 1e-10) {
            for (auto& row : probs[i]) {
                for (double& p : row) p /= sum;
            }
        }
    }
    return probs;
}

// ============================================================================
// 主匹配函数（使用已加载的数据）
// ============================================================================

void ViterbiMatch(
    const std::string& result_path,
    const std::string& backtrack_txt,
    const std::vector<VisualMapData>& mapData,
    const std::vector<VisualQueryData>& queryData,
    int target_frames,
    bool use_motion_prediction,
    const PnPConfig* pnp_config) {

    if (mapData.empty()) {
        throw std::runtime_error("地图数据为空");
    }
    if (queryData.empty()) {
        throw std::runtime_error("查询数据为空");
    }

    // 记录总处理时间
    auto total_start = std::chrono::steady_clock::now();

    std::cout << "\n开始Viterbi匹配..." << std::endl;
    std::cout << "  地图帧数: " << mapData.size() << std::endl;
    std::cout << "  初始查询帧数: " << queryData.size() << std::endl;
    std::cout << "  查询模式: " << (use_motion_prediction ? "运动预测模式" : "完整CSV查询模式") << std::endl;

    // 初始化PnP求解器（如果启用）
    PnPSolver* pnp_solver = nullptr;
    if (pnp_config && pnp_config->enable) {
        std::cout << "  PnP优化: 启用（最小内点数=" << pnp_config->min_inliers << "）" << std::endl;
        pnp_solver = new PnPSolver(
            pnp_config->min_inliers,
            pnp_config->ransac_reproj_error
        );
        pnp_solver->setVerbose(false); // 关闭详细输出以避免刷屏
    } else {
        std::cout << "  PnP优化: 禁用" << std::endl;
    }

    // 创建本地Viterbi实例
    Viterbi viterbi;

    // 创建运动预测器（窗口大小=5，初始化帧数=5）
    MotionPredictor motion_predictor(5, 5);

    // 准备查询数据：只使用前5帧作为初始数据
    std::vector<VisualQueryData> initial_query_data = queryData;

    // 如果未指定目标帧数，使用初始查询数据的大小
    if (target_frames <= 0) {
        target_frames = static_cast<int>(queryData.size());
    }

    std::cout << "  初始查询帧数: " << initial_query_data.size() << std::endl;
    std::cout << "  目标帧数: " << target_frames << std::endl;

    // ==================== 前向Viterbi ====================
    std::cout << "\n[前向Viterbi计算]" << std::endl;
    if (use_motion_prediction) {
        std::cout << "  策略：运动预测模式 - 用Viterbi匹配结果更新运动预测器，实现闭环反馈" << std::endl;
    } else {
        std::cout << "  策略：完整CSV查询模式 - 使用CSV中的完整位姿信息" << std::endl;
    }
    int last_hidden_state = -1;

    auto forward_start = std::chrono::steady_clock::now();

    // 存储所有帧的匹配结果（用于后续保存）
    std::vector<VisualQueryData> all_query_poses;
    all_query_poses.reserve(target_frames);

    for (int i = 0; i < target_frames; ++i) {
        int time_idx = (i == 0) ? 0 : (i == 1) ? 1 : 2;

        // 获取当前帧的query数据
        VisualQueryData current_query;

        if (use_motion_prediction) {
            // ========== 运动预测模式 ==========
            if (i < static_cast<int>(initial_query_data.size())) {
                // 前5帧：使用CSV中的真实数据
                current_query = initial_query_data[i];
            } else {
                // 第6帧及以后：使用运动预测
                current_query.frame_id = i;
                // 预测位姿（基于之前的匹配结果）
                VisualQueryData predicted = motion_predictor.predictPose(current_query);
                current_query.x = predicted.x;
                current_query.y = predicted.y;
                current_query.z = predicted.z;
                current_query.rotation = predicted.rotation;
                // 没有实际图片，描述符为空
                current_query.descriptors_size1 = "";
                current_query.descriptors_size2 = "";
                current_query.descriptors_size4 = "";
            }
        } else {
            // ========== 完整CSV查询模式 ==========
            if (i < static_cast<int>(initial_query_data.size())) {
                // 直接使用CSV中的数据
                current_query = initial_query_data[i];
            } else {
                throw std::runtime_error("完整CSV查询模式下，查询帧数超出CSV数据范围");
            }
        }

        // 使用当前位姿进行粗定位（确定候选集）
        VisualQueryData query_for_coarse = use_motion_prediction
            ? motion_predictor.predictPose(current_query)  // 运动预测模式：使用预测位姿
            : current_query;                                // 完整CSV模式：使用CSV位姿
        auto candidates = viterbi.determineSequence(mapData, query_for_coarse);

        // Viterbi精确匹配
        viterbi.init(mapData, candidates);
        viterbi.uptrans_prob(current_query, candidates, mapData, time_idx);
        viterbi.upobr_prob(current_query, candidates, mapData);
        int result = viterbi.calculate_prob(time_idx, candidates, backtrack_txt);

        // 更新运动预测器（仅在运动预测模式下）
        if (use_motion_prediction) {
            // 关键：用Viterbi匹配得到的地图位姿更新预测器（闭环反馈）
            VisualQueryData matched_pose;
            matched_pose.frame_id = i;
            matched_pose.x = mapData[result].x;
            matched_pose.y = mapData[result].y;
            matched_pose.z = mapData[result].z;
            matched_pose.rotation = mapData[result].rotation;
            matched_pose.descriptors_size1 = "";
            matched_pose.descriptors_size2 = "";
            matched_pose.descriptors_size4 = "";

            // 用匹配结果更新预测器
            motion_predictor.addPose(matched_pose);
        }

        // 保存当前帧的位姿（用于最终结果输出）
        all_query_poses.push_back(current_query);

        if (i == target_frames - 1) {
            last_hidden_state = result;
        }

        // 进度显示
        if ((i + 1) % 50 == 0 || i == target_frames - 1) {
            double progress = 100.0 * (i + 1) / target_frames;
            std::cout << "\r  前向进度: " << std::fixed << std::setprecision(1)
                      << progress << "% (" << (i + 1) << "/" << target_frames << ")   ";
            std::cout.flush();
        }
    }
    std::cout << std::endl;

    auto forward_end = std::chrono::steady_clock::now();
    double forward_time = std::chrono::duration<double>(forward_end - forward_start).count();
    std::cout << "  前向完成，耗时: " << std::fixed << std::setprecision(2)
              << forward_time << " 秒" << std::endl;
    std::cout << "  最终状态索引: " << last_hidden_state << std::endl;

    // ==================== 回溯 ====================
    std::cout << "\n[回溯计算]" << std::endl;

    auto backtrack_data = readBacktrackDataFromFile(backtrack_txt);
    if (backtrack_data.empty()) {
        throw std::runtime_error("回溯数据为空");
    }

    std::cout << "  读取回溯数据: " << backtrack_data.size() << " 行" << std::endl;

    // 反转顺序进行回溯
    std::reverse(backtrack_data.begin(), backtrack_data.end());

    std::vector<int> results(all_query_poses.size());
    results[0] = last_hidden_state;

    auto backtrack_start = std::chrono::steady_clock::now();

    for (size_t i = 1; i < backtrack_data.size(); ++i) {
        double max_prob = -1.0;
        int max_idx = 0;

        if (i == 1) {
            // 一阶回溯
            auto trans = calcBacktrackT1(last_hidden_state, backtrack_data[i], mapData);
            for (size_t j = 0; j < backtrack_data[i].size(); ++j) {
                double prob = backtrack_data[i][j].probability * trans[j];
                if (prob > max_prob) {
                    max_prob = prob;
                    max_idx = static_cast<int>(j);
                }
            }
        } else {
            // 二阶回溯
            auto trans = calcBacktrackT2(
                backtrack_data[i-2], backtrack_data[i-1], backtrack_data[i], mapData
            );
            for (size_t a = 0; a < backtrack_data[i-2].size(); ++a) {
                for (size_t b = 0; b < backtrack_data[i-1].size(); ++b) {
                    for (size_t c = 0; c < backtrack_data[i].size(); ++c) {
                        double prob = backtrack_data[i][c].probability * trans[a][b][c];
                        if (prob > max_prob) {
                            max_prob = prob;
                            max_idx = static_cast<int>(c);
                        }
                    }
                }
            }
        }
        results[i] = backtrack_data[i][max_idx].last_state_index;

        // 进度显示
        if ((i + 1) % 100 == 0 || i == backtrack_data.size() - 1) {
            double progress = 100.0 * (i + 1) / backtrack_data.size();
            std::cout << "\r  回溯进度: " << std::fixed << std::setprecision(1) 
                      << progress << "% (" << (i + 1) << "/" << backtrack_data.size() << ")   ";
            std::cout.flush();
        }
    }
    std::cout << std::endl;

    // 恢复正序
    std::reverse(results.begin(), results.end());

    auto backtrack_end = std::chrono::steady_clock::now();
    double backtrack_time = std::chrono::duration<double>(backtrack_end - backtrack_start).count();
    std::cout << "  回溯完成，耗时: " << std::fixed << std::setprecision(2) 
              << backtrack_time << " 秒" << std::endl;

    // ==================== PnP位姿优化 ====================
    std::vector<bool> pnp_success_flags(results.size(), false);
    std::vector<int> pnp_inliers_count(results.size(), 0);
    int pnp_success_count = 0;

    if (pnp_solver) {
        std::cout << "\n[PnP位姿优化]" << std::endl;
        std::cout << "  开始对匹配结果进行PnP优化..." << std::endl;

        auto pnp_start = std::chrono::steady_clock::now();

        // 调试：检查前几帧的image_path
        std::cout << "  调试信息 - 检查image_path:" << std::endl;
        for (size_t i = 0; i < std::min(size_t(3), results.size()); ++i) {
            int matched_idx = results[i];
            std::cout << "    Query[" << i << "].image_path: "
                      << (all_query_poses[i].image_path.empty() ? "空" : all_query_poses[i].image_path) << std::endl;
            std::cout << "    Map[" << matched_idx << "].image_path: "
                      << (mapData[matched_idx].image_path.empty() ? "空" : mapData[matched_idx].image_path) << std::endl;
        }

        for (size_t i = 0; i < results.size(); ++i) {
            int matched_idx = results[i];

            // 尝试PnP求解
            double refined_x, refined_y, refined_z;
            std::array<std::array<double, 3>, 3> refined_rotation;
            int num_inliers;

            bool success = pnp_solver->solvePnP(
                all_query_poses[i],
                mapData[matched_idx],
                pnp_config->camera_matrix,
                pnp_config->dist_coeffs,
                refined_x, refined_y, refined_z,
                refined_rotation,
                num_inliers
            );

            if (success) {
                // 更新位姿
                all_query_poses[i].x = refined_x;
                all_query_poses[i].y = refined_y;
                all_query_poses[i].z = refined_z;
                all_query_poses[i].rotation = refined_rotation;

                pnp_success_flags[i] = true;
                pnp_inliers_count[i] = num_inliers;
                pnp_success_count++;
            }

            // 每100帧显示一次进度
            if ((i + 1) % 100 == 0 || i == results.size() - 1) {
                std::cout << "  进度: " << (i + 1) << "/" << results.size()
                          << " (成功: " << pnp_success_count << ")" << std::endl;
            }
        }

        auto pnp_end = std::chrono::steady_clock::now();
        double pnp_time = std::chrono::duration<double>(pnp_end - pnp_start).count();

        std::cout << "  PnP优化完成，耗时: " << std::fixed << std::setprecision(2)
                  << pnp_time << " 秒" << std::endl;
        std::cout << "  成功率: " << pnp_success_count << "/" << results.size()
                  << " (" << std::fixed << std::setprecision(1)
                  << (100.0 * pnp_success_count / results.size()) << "%)" << std::endl;
    }

    // ==================== 保存结果 ====================
    std::cout << "\n[保存结果]" << std::endl;
    
    std::ofstream out(result_path);
    if (!out.is_open()) {
        throw std::runtime_error("无法创建输出文件: " + result_path);
    }

    // 写入表头
    if (pnp_solver) {
        out << "query_frame_id,matched_map_frame_id,query_x,query_y,query_z,query_yaw,"
            << "map_x,map_y,map_z,map_yaw,distance_error,pnp_refined,pnp_inliers" << std::endl;
    } else {
        out << "query_frame_id,matched_map_frame_id,query_x,query_y,query_z,query_yaw,"
            << "map_x,map_y,map_z,map_yaw,distance_error" << std::endl;
    }

    double total_error = 0.0;
    for (size_t i = 0; i < results.size(); ++i) {
        int matched_idx = results[i];

        double dx = all_query_poses[i].x - mapData[matched_idx].x;
        double dy = all_query_poses[i].y - mapData[matched_idx].y;
        double error = std::sqrt(dx * dx + dy * dy);
        total_error += error;

        out << all_query_poses[i].frame_id << ","
            << mapData[matched_idx].frame_id << ","
            << std::fixed << std::setprecision(6)
            << all_query_poses[i].x << ","
            << all_query_poses[i].y << ","
            << all_query_poses[i].z << ","
            << all_query_poses[i].getYawDegrees() << ","
            << mapData[matched_idx].x << ","
            << mapData[matched_idx].y << ","
            << mapData[matched_idx].z << ","
            << mapData[matched_idx].getYawDegrees() << ","
            << error;

        if (pnp_solver) {
            out << "," << (pnp_success_flags[i] ? "1" : "0")
                << "," << pnp_inliers_count[i];
        }

        out << std::endl;
    }
    out.close();

    double avg_error = total_error / results.size();

    std::cout << "  结果已保存到: " << result_path << std::endl;
    std::cout << "  平均定位误差: " << std::fixed << std::setprecision(3)
              << avg_error << " 米" << std::endl;

    // 显示部分匹配结果
    std::cout << "\n[匹配结果预览 (前10帧)]" << std::endl;
    for (size_t i = 0; i < std::min(size_t(10), results.size()); ++i) {
        int matched_idx = results[i];
        double dx = all_query_poses[i].x - mapData[matched_idx].x;
        double dy = all_query_poses[i].y - mapData[matched_idx].y;
        double error = std::sqrt(dx * dx + dy * dy);

        std::cout << "  Query " << std::setw(4) << all_query_poses[i].frame_id
                  << " -> Map " << std::setw(4) << mapData[matched_idx].frame_id
                  << " | 误差: " << std::fixed << std::setprecision(3) << error << " m"
                  << std::endl;
    }

    // 释放PnP求解器
    if (pnp_solver) {
        delete pnp_solver;
    }

    // 计算并输出总处理时间
    auto total_end = std::chrono::steady_clock::now();
    double total_time = std::chrono::duration<double>(total_end - total_start).count();

    std::cout << "\n匹配完成!" << std::endl;
    std::cout << "  总处理时间: " << std::fixed << std::setprecision(2)
              << total_time << " 秒" << std::endl;
    std::cout << "  平均每帧耗时: " << std::fixed << std::setprecision(3)
              << (total_time / results.size() * 1000.0) << " 毫秒" << std::endl;
}

// ============================================================================
// 主匹配函数（从CSV加载数据）
// ============================================================================

void ViterbiMatch(
    const std::string& result_path,
    const std::string& backtrack_txt,
    const std::string& map_csv_path,
    const std::string& map_images_folder,
    const std::string& query_csv_path,
    const std::string& query_images_folder) {
    
    // 加载地图数据
    std::vector<VisualMapData> mapData = VisualMapData::buildVisualMapFromCSV(
        map_csv_path, map_images_folder
    );

    // 加载查询数据
    std::vector<VisualQueryData> queryData = VisualQueryData::buildQueryFromCSV(
        query_csv_path, query_images_folder
    );

    // 清空回溯文件
    std::ofstream(backtrack_txt, std::ios::trunc).close();

    // 执行匹配
    ViterbiMatch(result_path, backtrack_txt, mapData, queryData);
}