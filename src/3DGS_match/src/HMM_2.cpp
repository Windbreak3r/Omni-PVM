#include "3DGS_match/HMM_2.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cmath>

// ============================================================================
// 工具函数
// ============================================================================

int hammingDistance(const std::string& str1, const std::string& str2) {
    if (str1.length() != str2.length()) {
        return -1;
    }
    int distance = 0;
    for (size_t i = 0; i < str1.length(); ++i) {
        if (str1[i] != str2[i]) {
            ++distance;
        }
    }
    return distance;
}

void writeBacktrackDataToFile(
    const std::vector<BacktrackData>& backtrack_data,
    const std::string& filename) {
    
    std::ofstream file(filename, std::ios::app);
    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return;
    }

    for (const auto& data : backtrack_data) {
        file << data.last_state_index << " " << data.probability << " ";
    }
    file << std::endl;
    file.close();
}

// ============================================================================
// Viterbi 类实现
// ============================================================================

Viterbi::Viterbi() 
    : distance_scale_factor(5.0),
      gaussian_sigma(1.0),
      gaussian_mu(0.0),
      real_size(216) {}

void Viterbi::normalizeProbabilities(std::vector<double>& probs) {
    double sum = 0.0;
    for (double p : probs) sum += p;
    if (sum > 1e-10) {
        for (double& p : probs) p /= sum;
    }
}

double Viterbi::angleSimilarity(double angle1, double angle2) {
    auto normalize = [](double a) -> double {
        a = std::fmod(a, 360.0);
        return a < 0.0 ? a + 360.0 : a;
    };
    double norm1 = normalize(angle1);
    double norm2 = normalize(angle2);
    double delta = std::abs(norm1 - norm2);
    double minDelta = std::min(delta, 360.0 - delta);
    return 1.0 - (minDelta / 180.0);
}

double Viterbi::calculateMixedDistance(
    double map_x, double map_y, double map_yaw,
    double query_x, double query_y, double query_yaw) {
    
    double dx = map_x - query_x;
    double dy = map_y - query_y;
    double xy_dist = std::sqrt(dx * dx + dy * dy);
    
    double theta_sim = angleSimilarity(map_yaw, query_yaw);
    double theta_scaled = 0.9 + 0.1 * theta_sim;
    
    return xy_dist / theta_scaled;
}

// 粗定位：筛选最近的N个候选点（固定数量策略）
std::vector<int> Viterbi::determineSequence(
    const std::vector<VisualMapData>& visualMapVec,
    const VisualQueryData& queryData) {

    struct DistanceWithIndex {
        int index;
        double distance;
    };

    // 候选数量：取地图大小和64的较小值
    const size_t FIXED_CANDIDATES = std::min(size_t(64), visualMapVec.size());

    // 计算所有地图点到查询点的距离
    std::vector<DistanceWithIndex> distanceVec;
    distanceVec.reserve(visualMapVec.size());

    for (size_t i = 0; i < visualMapVec.size(); ++i) {
        double dx = visualMapVec[i].x - queryData.x;
        double dy = visualMapVec[i].y - queryData.y;
        double xy_dist = std::sqrt(dx * dx + dy * dy);
        distanceVec.push_back({static_cast<int>(i), xy_dist});
    }

    // 按距离排序
    std::sort(distanceVec.begin(), distanceVec.end(),
        [](const DistanceWithIndex& a, const DistanceWithIndex& b) {
            return a.distance < b.distance;
        }
    );

    // 提取最近的FIXED_CANDIDATES个索引
    std::vector<int> result;
    size_t take = std::min(FIXED_CANDIDATES, distanceVec.size());
    result.reserve(take);

    for (size_t i = 0; i < take; ++i) {
        result.push_back(distanceVec[i].index);
    }

    // 输出调试信息
    if (take < FIXED_CANDIDATES) {
        std::cout << "  警告: 查询位置 (" << queryData.x << ", " << queryData.y
                  << ") 地图点总数不足，只找到 " << take << " 个候选点（目标: "
                  << FIXED_CANDIDATES << "）" << std::endl;
    }

    return result;
}

// 初始化：提取候选帧的特征
void Viterbi::init(
    const std::vector<VisualMapData>& visualMapVec,
    const std::vector<int>& Index_array) {

    map_vec_descriptors_size1.clear();
    map_vec_descriptors_size2.clear();
    map_vec_descriptors_size4.clear();

    map_vec_descriptors_size1.reserve(Index_array.size());
    map_vec_descriptors_size2.reserve(Index_array.size());
    map_vec_descriptors_size4.reserve(Index_array.size());

    for (int idx : Index_array) {
        // 添加边界检查，防止访问越界
        if (idx < 0 || idx >= static_cast<int>(visualMapVec.size())) {
            std::cerr << "警告: 索引越界 idx=" << idx
                      << ", visualMapVec.size()=" << visualMapVec.size() << std::endl;
            // 使用空描述符作为后备
            map_vec_descriptors_size1.push_back("");
            map_vec_descriptors_size2.push_back("");
            map_vec_descriptors_size4.push_back("");
            continue;
        }
        map_vec_descriptors_size1.push_back(visualMapVec[idx].descriptors_size1);
        map_vec_descriptors_size2.push_back(visualMapVec[idx].descriptors_size2);
        map_vec_descriptors_size4.push_back(visualMapVec[idx].descriptors_size4);
    }

    obr_prob.assign(Index_array.size(), 0.0);
}

// 发射概率：基于ORB特征汉明距离
void Viterbi::upobr_prob(
    const VisualQueryData& queryData,
    const std::vector<int>& Index_array,
    const std::vector<VisualMapData>& visualMapVec) {

    obr_prob.assign(Index_array.size(), 0.0);

    // 检查查询数据是否有描述符（预测的帧可能没有）
    bool has_descriptors = !queryData.descriptors_size1.empty() &&
                          !queryData.descriptors_size2.empty() &&
                          !queryData.descriptors_size4.empty();

    if (!has_descriptors) {
        // 没有描述符时，使用均匀分布（只依赖位置信息）
        double uniform_prob = 1.0 / Index_array.size();
        for (size_t i = 0; i < Index_array.size(); ++i) {
            obr_prob[i] = uniform_prob;
        }
        return;
    }

    // 有描述符时，正常计算汉明距离
    for (size_t i = 0; i < Index_array.size(); ++i) {
        int dist1 = hammingDistance(queryData.descriptors_size1, map_vec_descriptors_size1[i]);
        int dist2 = hammingDistance(queryData.descriptors_size2, map_vec_descriptors_size2[i]);
        int dist4 = hammingDistance(queryData.descriptors_size4, map_vec_descriptors_size4[i]);

        if (dist1 < 0 || dist2 < 0 || dist4 < 0) {
            obr_prob[i] = 0.0;
            continue;
        }

        int len1 = static_cast<int>(queryData.descriptors_size1.length());
        int len2 = static_cast<int>(queryData.descriptors_size2.length());
        int len4 = static_cast<int>(queryData.descriptors_size4.length());

        double sim1 = (len1 > 0) ? 1.0 - static_cast<double>(dist1) / len1 : 0.0;
        double sim2 = (len2 > 0) ? 1.0 - static_cast<double>(dist2) / len2 : 0.0;
        double sim4 = (len4 > 0) ? 1.0 - static_cast<double>(dist4) / len4 : 0.0;

        // 加权融合：越细粒度权重越高
        obr_prob[i] = 0.2 * sim1 + 0.3 * sim2 + 0.5 * sim4;
    }

    normalizeProbabilities(obr_prob);
}

// 转移概率：基于位姿预测
void Viterbi::uptrans_prob(
    const VisualQueryData& queryData,
    const std::vector<int>& Index_array,
    const std::vector<VisualMapData>& visualMapVec,
    int time_Index) {
    
    size_t n = Index_array.size();

    if (time_Index == 0) {
        // 首帧：无转移概率，只保存当前索引
        last_Index_array = Index_array;
        return;
    } 
    else if (time_Index == 1) {
        // 第二帧：一阶马尔可夫转移
        trans_prob_t1.assign(last_Index_array.size(), std::vector<double>(n, 0.0));

        for (size_t i = 0; i < last_Index_array.size(); ++i) {
            int last_idx = last_Index_array[i];

            // 边界检查
            if (last_idx < 0 || last_idx >= static_cast<int>(visualMapVec.size())) {
                std::cerr << "警告: uptrans_prob中索引越界 last_idx=" << last_idx << std::endl;
                continue;
            }

            double predict_x = visualMapVec[last_idx].x;
            double predict_y = visualMapVec[last_idx].y;
            double predict_yaw = visualMapVec[last_idx].getYawDegrees();

            for (size_t j = 0; j < n; ++j) {
                int curr_idx = Index_array[j];

                // 边界检查
                if (curr_idx < 0 || curr_idx >= static_cast<int>(visualMapVec.size())) {
                    std::cerr << "警告: uptrans_prob中索引越界 curr_idx=" << curr_idx << std::endl;
                    trans_prob_t1[i][j] = 0.0;
                    continue;
                }

                double dist = calculateMixedDistance(
                    visualMapVec[curr_idx].x,
                    visualMapVec[curr_idx].y,
                    visualMapVec[curr_idx].getYawDegrees(),
                    predict_x, predict_y, predict_yaw
                );

                trans_prob_t1[i][j] = (1.0 / (gaussian_sigma * std::sqrt(2 * M_PI))) *
                    std::exp(-std::pow((dist / distance_scale_factor - gaussian_mu), 2) /
                             (2 * std::pow(gaussian_sigma, 2)));
            }
            normalizeProbabilities(trans_prob_t1[i]);
        }

        // 更新历史状态
        last_two_Index_array = last_Index_array;
        last_Index_array = Index_array;
    } 
    else {
        // 第三帧及以后：二阶马尔可夫转移
        trans_prob_t2.assign(
            last_two_Index_array.size(),
            std::vector<std::vector<double>>(last_Index_array.size(), std::vector<double>(n, 0.0))
        );

        for (size_t i = 0; i < last_two_Index_array.size(); ++i) {
            int idx_t2 = last_two_Index_array[i];

            // 边界检查
            if (idx_t2 < 0 || idx_t2 >= static_cast<int>(visualMapVec.size())) {
                std::cerr << "警告: uptrans_prob中索引越界 idx_t2=" << idx_t2 << std::endl;
                continue;
            }

            for (size_t j = 0; j < last_Index_array.size(); ++j) {
                int idx_t1 = last_Index_array[j];

                // 边界检查
                if (idx_t1 < 0 || idx_t1 >= static_cast<int>(visualMapVec.size())) {
                    std::cerr << "警告: uptrans_prob中索引越界 idx_t1=" << idx_t1 << std::endl;
                    continue;
                }

                // 二阶线性预测：基于前两帧外推
                double predict_x = 2.0 * visualMapVec[idx_t1].x - visualMapVec[idx_t2].x;
                double predict_y = 2.0 * visualMapVec[idx_t1].y - visualMapVec[idx_t2].y;
                double yaw_t2 = visualMapVec[idx_t2].getYawDegrees();
                double yaw_t1 = visualMapVec[idx_t1].getYawDegrees();
                double predict_yaw = 2.0 * yaw_t1 - yaw_t2;

                for (size_t k = 0; k < n; ++k) {
                    int curr_idx = Index_array[k];

                    // 边界检查
                    if (curr_idx < 0 || curr_idx >= static_cast<int>(visualMapVec.size())) {
                        std::cerr << "警告: uptrans_prob中索引越界 curr_idx=" << curr_idx << std::endl;
                        trans_prob_t2[i][j][k] = 0.0;
                        continue;
                    }

                    double dist = calculateMixedDistance(
                        visualMapVec[curr_idx].x,
                        visualMapVec[curr_idx].y,
                        visualMapVec[curr_idx].getYawDegrees(),
                        predict_x, predict_y, predict_yaw
                    );

                    trans_prob_t2[i][j][k] = (1.0 / (gaussian_sigma * std::sqrt(2 * M_PI))) *
                        std::exp(-std::pow((dist / distance_scale_factor - gaussian_mu), 2) /
                                 (2 * std::pow(gaussian_sigma, 2)));
                }
                normalizeProbabilities(trans_prob_t2[i][j]);
            }
        }

        // 更新历史状态
        last_two_Index_array = last_Index_array;
        last_Index_array = Index_array;
    }
}

// Viterbi概率计算
int Viterbi::calculate_prob(
    int time_Index,
    std::vector<int>& Index_array,
    const std::string& backtrack_txt) {
    
    size_t n = Index_array.size();
    std::vector<BacktrackData> backtrack_data(n);

    if (time_Index == 0) {
        // 首帧：只用发射概率
        laststates_prob_t1.assign(n, 0.0);
        
        for (size_t i = 0; i < n; ++i) {
            laststates_prob_t1[i] = obr_prob[i];
            backtrack_data[i] = {Index_array[i], laststates_prob_t1[i]};
        }
        normalizeProbabilities(laststates_prob_t1);
    } 
    else if (time_Index == 1) {
        // 第二帧：一阶转移
        std::vector<double> current_prob(n, 0.0);
        laststates_prob_t2.assign(last_two_Index_array.size(), std::vector<double>(n, 0.0));

        for (size_t j = 0; j < n; ++j) {
            double max_prob = 0.0;
            for (size_t i = 0; i < last_two_Index_array.size(); ++i) {
                double prob = laststates_prob_t1[i] * trans_prob_t1[i][j] * obr_prob[j];
                laststates_prob_t2[i][j] = prob;
                if (prob > max_prob) {
                    max_prob = prob;
                }
            }
            current_prob[j] = max_prob;
            backtrack_data[j] = {Index_array[j], max_prob};
        }
        
        normalizeProbabilities(current_prob);
        for (size_t i = 0; i < last_two_Index_array.size(); ++i) {
            normalizeProbabilities(laststates_prob_t2[i]);
        }
    } 
    else {
        // 第三帧及以后：二阶转移
        std::vector<double> current_prob(n, 0.0);
        std::vector<std::vector<double>> new_laststates_prob_t2(
            last_two_Index_array.size(), std::vector<double>(n, 0.0)
        );

        // 检查维度是否匹配
        bool dimension_mismatch = false;
        if (laststates_prob_t2.size() != last_two_Index_array.size()) {
            std::cerr << "警告: laststates_prob_t2维度不匹配 ("
                      << laststates_prob_t2.size() << " vs " << last_two_Index_array.size() << ")" << std::endl;
            dimension_mismatch = true;
        }
        if (!dimension_mismatch && !laststates_prob_t2.empty() &&
            laststates_prob_t2[0].size() != last_Index_array.size()) {
            std::cerr << "警告: laststates_prob_t2[0]维度不匹配 ("
                      << laststates_prob_t2[0].size() << " vs " << last_Index_array.size() << ")" << std::endl;
            dimension_mismatch = true;
        }

        if (dimension_mismatch) {
            // 维度不匹配时，使用简化的一阶转移
            std::cerr << "使用简化的一阶转移作为后备" << std::endl;
            for (size_t k = 0; k < n; ++k) {
                current_prob[k] = obr_prob[k];
                backtrack_data[k] = {Index_array[k], obr_prob[k]};
            }
        } else {
            // 正常的二阶转移
            for (size_t k = 0; k < n; ++k) {
                double max_prob = 0.0;
                for (size_t i = 0; i < last_two_Index_array.size(); ++i) {
                    for (size_t j = 0; j < last_Index_array.size(); ++j) {
                        double prob = laststates_prob_t2[i][j] * trans_prob_t2[i][j][k] * obr_prob[k];
                        new_laststates_prob_t2[j][k] += prob;
                        if (prob > max_prob) {
                            max_prob = prob;
                        }
                    }
                }
                current_prob[k] = max_prob;
                backtrack_data[k] = {Index_array[k], max_prob};
            }
        }

        laststates_prob_t2 = new_laststates_prob_t2;
        normalizeProbabilities(current_prob);
        for (size_t j = 0; j < last_two_Index_array.size(); ++j) {
            normalizeProbabilities(laststates_prob_t2[j]);
        }
    }

    // 写入回溯数据
    writeBacktrackDataToFile(backtrack_data, backtrack_txt);

    // 找最大概率的索引
    int best_idx = 0;
    double best_prob = backtrack_data[0].probability;
    for (size_t i = 1; i < n; ++i) {
        if (backtrack_data[i].probability > best_prob) {
            best_prob = backtrack_data[i].probability;
            best_idx = static_cast<int>(i);
        }
    }

    return Index_array[best_idx];
}