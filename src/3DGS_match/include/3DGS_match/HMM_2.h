#ifndef HMM_2_H
#define HMM_2_H

#include <string>
#include <vector>
#include <cmath>
#include "3DGS_match/visual_map.h"
#include "3DGS_match/query_information.h"

struct BacktrackData {
    int last_state_index;
    double probability;
};

class Viterbi {
public:
    Viterbi();

    // 粗定位：返回最近的候选索引
    std::vector<int> determineSequence(
        const std::vector<VisualMapData>& visualMapVec,
        const VisualQueryData& queryData
    );

    // 初始化
    void init(
        const std::vector<VisualMapData>& visualMapVec,
        const std::vector<int>& Index_array
    );

    // 发射概率
    void upobr_prob(
        const VisualQueryData& queryData,
        const std::vector<int>& Index_array,
        const std::vector<VisualMapData>& visualMapVec
    );

    // 转移概率
    void uptrans_prob(
        const VisualQueryData& queryData,
        const std::vector<int>& Index_array,
        const std::vector<VisualMapData>& visualMapVec,
        int time_Index
    );

    // Viterbi计算
    int calculate_prob(
        int time_Index,
        std::vector<int>& Index_array,
        const std::string& backtrack_txt
    );

private:
    double distance_scale_factor;
    double gaussian_sigma;
    double gaussian_mu;
    int real_size;

    std::vector<std::string> map_vec_descriptors_size1;
    std::vector<std::string> map_vec_descriptors_size2;
    std::vector<std::string> map_vec_descriptors_size4;

    std::vector<double> obr_prob;
    std::vector<std::vector<double>> trans_prob_t1;
    std::vector<std::vector<std::vector<double>>> trans_prob_t2;

    std::vector<double> laststates_prob_t1;
    std::vector<std::vector<double>> laststates_prob_t2;
    std::vector<int> last_Index_array;
    std::vector<int> last_two_Index_array;

    void normalizeProbabilities(std::vector<double>& probs);
    double angleSimilarity(double angle1, double angle2);
    double calculateMixedDistance(
        double map_x, double map_y, double map_yaw,
        double query_x, double query_y, double query_yaw
    );
};

// 工具函数
int hammingDistance(const std::string& str1, const std::string& str2);

void writeBacktrackDataToFile(
    const std::vector<BacktrackData>& backtrack_data,
    const std::string& filename
);

#endif // HMM_2_H