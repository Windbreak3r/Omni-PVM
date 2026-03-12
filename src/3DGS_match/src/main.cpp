#include <iostream>
#include <string>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "3DGS_match/visual_map.h"
#include "3DGS_match/query_information.h"
#include "3DGS_match/match.h"

namespace fs = std::filesystem;


struct Config {
    // ==================== 视觉地图构建 ====================
    // bool step1_build_map = true;      // 步骤1：构建视觉地图
    // bool step2_run_matching = false;    // 步骤2：执行匹配

    // bool use_motion_prediction = false; // true: 运动预测模式, false: 完整CSV查询模式

    // bool load_from_cache = false;       // 是否从缓存加载地图（跳过图片处理）
    // bool save_map_cache = true;       // 构建地图后是否保存缓存

    // ==================== 匹配算法 ====================
    bool step1_build_map = false;      // 步骤1：构建视觉地图
    bool step2_run_matching = true;    // 步骤2：执行匹配

    bool use_motion_prediction = false; // true: 运动预测模式, false: 完整CSV查询模式

    bool load_from_cache = true;       // 是否从缓存加载地图（跳过图片处理）
    bool save_map_cache = false;       // 构建地图后是否保存缓存
    
    // ==================== PnP位姿优化 ====================
    bool enable_pnp_refinement = true; // 是否启用PnP位姿优化（稀疏地图场景推荐）
    int pnp_min_inliers = 20;          // PnP最小内点数阈值
    double pnp_ransac_reproj_error = 8.0; // PnP RANSAC重投影误差阈值（像素）

    // 相机内参（根据实际相机标定结果）
    // Calibration: projection=[1432.69, 1429.36, 964.59, 772.05]
    double camera_fx = 1432.69236698;  // 焦距 fx
    double camera_fy = 1429.36076169;  // 焦距 fy
    double camera_cx = 964.58584504;   // 主点 cx
    double camera_cy = 772.05318599;   // 主点 cy



    // ==================== 地图数据路径 ====================
    std::string map_csv_path = "/home/dragonwu/experiments/compara2/HMM+PnP/PVM/sampled_cameras_for_mapping_sparse.csv";
    std::string map_images_folder = "/home/dragonwu/experiments/compara2/HMM+PnP/PVM/images_sparse";
    std::string map_cache_binary = "/home/dragonwu/experiments/compara2/HMM+PnP/visual_map.bin";
    std::string map_cache_text = "/home/dragonwu/experiments/compara2/HMM+PnP/visual_map_info.txt";

    // ==================== 查询数据路径 ====================
    std::string query_csv_path = "/home/dragonwu/experiments/L1/query_poses.csv";
    std::string query_images_folder = "/home/dragonwu/experiments/L1/query_images";

    // ==================== 输出路径 ====================
    std::string result_csv_path = "/home/dragonwu/experiments/compara2/HMM+PnP/results/match_results.csv";
    std::string backtrack_file = "/home/dragonwu/experiments/compara2/HMM+PnP/results/backtrack_temp.txt";

    // ==================== 算法参数 ====================
    int candidate_size = 16;           // 粗定位候选数量（固定）
    double distance_scale = 5.0;       // 距离缩放因子
    double gaussian_sigma = 1.0;       // 高斯分布标准差

    // ==================== 查询范围限制 ====================
    bool enable_query_range = false;    // 是否启用查询范围限制
    int query_start_image_id = 0;      // 起始图片ID（对应 0.jpg）
    int query_end_image_id = 476;      // 结束图片ID

};

// ============================================================================
// 全局变量
// ============================================================================

Viterbi viterbiInstance;
std::vector<VisualMapData> visualMapVec;
std::vector<VisualQueryData> queryDataVec;

// ============================================================================
// 辅助函数
// ============================================================================

void printSeparator(const std::string& title = "") {
    std::cout << "\n";
    std::cout << "========================================================" << std::endl;
    if (!title.empty()) {
        std::cout << "  " << title << std::endl;
        std::cout << "========================================================" << std::endl;
    }
}

void printConfig(const Config& cfg) {
    printSeparator("当前配置");

    std::cout << "\n[执行步骤]" << std::endl;
    std::cout << "  步骤1 - 构建视觉地图: " << (cfg.step1_build_map ? "是" : "否") << std::endl;
    std::cout << "  步骤2 - 执行匹配:     " << (cfg.step2_run_matching ? "是" : "否") << std::endl;

    std::cout << "\n[查询模式]" << std::endl;
    std::cout << "  模式: " << (cfg.use_motion_prediction ? "运动预测模式" : "完整CSV查询模式") << std::endl;
    if (cfg.use_motion_prediction) {
        std::cout << "  说明: 前5帧使用CSV真实位姿，后续帧使用运动预测+闭环反馈" << std::endl;
    } else {
        std::cout << "  说明: 所有帧都使用CSV中的完整位姿信息" << std::endl;
    }

    std::cout << "\n[缓存控制]" << std::endl;
    std::cout << "  从缓存加载: " << (cfg.load_from_cache ? "是" : "否") << std::endl;
    std::cout << "  保存缓存:   " << (cfg.save_map_cache ? "是" : "否") << std::endl;

    std::cout << "\n[地图数据路径]" << std::endl;
    if (!cfg.load_from_cache && cfg.step1_build_map) {
        std::cout << "  位姿CSV:    " << cfg.map_csv_path << std::endl;
        std::cout << "  图片文件夹: " << cfg.map_images_folder << std::endl;
    }
    std::cout << "  缓存文件:   " << cfg.map_cache_binary << std::endl;
    if (cfg.save_map_cache) {
        std::cout << "  信息文件:   " << cfg.map_cache_text << std::endl;
    }

    if (cfg.step2_run_matching) {
        std::cout << "\n[查询数据路径]" << std::endl;
        std::cout << "  位姿CSV:    " << cfg.query_csv_path << std::endl;
        std::cout << "  图片文件夹: " << cfg.query_images_folder << std::endl;

        std::cout << "\n[输出路径]" << std::endl;
        std::cout << "  匹配结果:   " << cfg.result_csv_path << std::endl;
        std::cout << "  回溯文件:   " << cfg.backtrack_file << std::endl;

        std::cout << "\n[查询范围限制]" << std::endl;
        if (cfg.enable_query_range) {
            std::cout << "  启用: 是" << std::endl;
            std::cout << "  图片范围: " << cfg.query_start_image_id << ".jpg ~ " 
                      << cfg.query_end_image_id << ".jpg" << std::endl;
            std::cout << "  总查询帧数: " << (cfg.query_end_image_id - cfg.query_start_image_id + 1) << std::endl;
        } else {
            std::cout << "  启用: 否（匹配所有查询帧）" << std::endl;
        }
    }

    std::cout << "\n[算法参数]" << std::endl;
    std::cout << "  候选数量:   " << cfg.candidate_size << std::endl;
    std::cout << "  距离缩放:   " << cfg.distance_scale << std::endl;
    std::cout << "  高斯sigma:  " << cfg.gaussian_sigma << std::endl;

    if (cfg.step2_run_matching) {
        std::cout << "\n[PnP位姿优化]" << std::endl;
        std::cout << "  启用PnP:    " << (cfg.enable_pnp_refinement ? "是" : "否") << std::endl;
        if (cfg.enable_pnp_refinement) {
            std::cout << "  最小内点数: " << cfg.pnp_min_inliers << std::endl;
            std::cout << "  重投影误差: " << cfg.pnp_ransac_reproj_error << " 像素" << std::endl;
            std::cout << "  相机内参:   fx=" << cfg.camera_fx << ", fy=" << cfg.camera_fy
                      << ", cx=" << cfg.camera_cx << ", cy=" << cfg.camera_cy << std::endl;
        }
    }

    std::cout << std::endl;
}

bool validatePaths(const Config& cfg) {
    bool valid = true;

    if (cfg.step1_build_map) {
        if (cfg.load_from_cache) {
            // 检查缓存文件
            if (!fs::exists(cfg.map_cache_binary)) {
                std::cerr << "错误: 地图缓存文件不存在: " << cfg.map_cache_binary << std::endl;
                valid = false;
            }
        } else {
            // 检查源文件
            if (!fs::exists(cfg.map_csv_path)) {
                std::cerr << "错误: 地图CSV文件不存在: " << cfg.map_csv_path << std::endl;
                valid = false;
            }
            if (!fs::exists(cfg.map_images_folder)) {
                std::cerr << "错误: 地图图片文件夹不存在: " << cfg.map_images_folder << std::endl;
                valid = false;
            }
        }
    }

    if (cfg.step2_run_matching) {
        if (visualMapVec.empty() && !cfg.step1_build_map && !cfg.load_from_cache) {
            std::cerr << "错误: 未构建地图且内存中无地图数据" << std::endl;
            valid = false;
        }
        if (!fs::exists(cfg.query_csv_path)) {
            std::cerr << "错误: 查询CSV文件不存在: " << cfg.query_csv_path << std::endl;
            valid = false;
        }
        if (!fs::exists(cfg.query_images_folder)) {
            std::cerr << "错误: 查询图片文件夹不存在: " << cfg.query_images_folder << std::endl;
            valid = false;
        }

        // 验证查询范围
        if (cfg.enable_query_range) {
            if (cfg.query_start_image_id < 0 || cfg.query_end_image_id < 0) {
                std::cerr << "错误: 查询范围ID不能为负数" << std::endl;
                valid = false;
            }
            if (cfg.query_start_image_id > cfg.query_end_image_id) {
                std::cerr << "错误: 起始图片ID不能大于结束图片ID" << std::endl;
                valid = false;
            }
        }
    }

    // 检查输出目录
    if (cfg.step2_run_matching || cfg.save_map_cache) {
        std::vector<std::string> output_paths;
        if (cfg.step2_run_matching) {
            output_paths.push_back(cfg.result_csv_path);
            output_paths.push_back(cfg.backtrack_file);
        }
        if (cfg.save_map_cache) {
            output_paths.push_back(cfg.map_cache_binary);
        }

        for (const auto& path : output_paths) {
            fs::path dir = fs::path(path).parent_path();
            if (!dir.empty() && !fs::exists(dir)) {
                try {
                    fs::create_directories(dir);
                    std::cout << "创建输出目录: " << dir << std::endl;
                } catch (const std::exception& e) {
                    std::cerr << "错误: 无法创建目录: " << dir << std::endl;
                    valid = false;
                }
            }
        }
    }

    return valid;
}

// ============================================================================
// 步骤1：构建视觉地图
// ============================================================================

bool buildVisualMap(const Config& cfg) {
    printSeparator("步骤1: 构建视觉地图");

    try {
        // 如果指定从缓存加载
        if (cfg.load_from_cache && fs::exists(cfg.map_cache_binary)) {
            std::cout << "从缓存加载地图..." << std::endl;
            visualMapVec = VisualMapData::loadMapFromFile(cfg.map_cache_binary);
        } else {
            // 从头构建
            std::cout << "从CSV和图片构建地图..." << std::endl;
            visualMapVec = VisualMapData::buildVisualMapFromCSV(
                cfg.map_csv_path,
                cfg.map_images_folder
            );

            // 保存缓存
            if (cfg.save_map_cache) {
                VisualMapData::saveMapToFile(visualMapVec, cfg.map_cache_binary);
                
                // 同时保存文本信息文件（可选）
                if (!cfg.map_cache_text.empty()) {
                    VisualMapData::saveMapToText(visualMapVec, cfg.map_cache_text);
                }
            }
        }

        std::cout << "\n视觉地图准备完成!" << std::endl;
        std::cout << "  总帧数: " << visualMapVec.size() << std::endl;

        // 数据预览
        std::cout << "\n数据预览 (前5帧):" << std::endl;
        for (size_t i = 0; i < std::min(size_t(5), visualMapVec.size()); ++i) {
            const auto& data = visualMapVec[i];
            std::cout << "  Frame " << data.frame_id 
                      << ": pos=(" << data.x << ", " << data.y << ", " << data.z << ")"
                      << ", yaw=" << data.getYawDegrees() << "°"
                      << ", desc_len=" << data.descriptors_size1.length() << std::endl;
        }

        return true;
    } catch (const std::exception& e) {
        std::cerr << "\n视觉地图构建/加载失败: " << e.what() << std::endl;
        return false;
    }
}

// ============================================================================
// 过滤查询数据（基于frame_id范围）
// ============================================================================

std::vector<VisualQueryData> filterQueryDataByRange(
    const std::vector<VisualQueryData>& query_data,
    const Config& cfg) {
    
    if (!cfg.enable_query_range) {
        return query_data;  // 不启用过滤，返回全部数据
    }

    std::vector<VisualQueryData> filtered;
    size_t skipped_count = 0;

    for (const auto& data : query_data) {
        if (data.frame_id >= cfg.query_start_image_id && 
            data.frame_id <= cfg.query_end_image_id) {
            filtered.push_back(data);
        } else {
            skipped_count++;
        }
    }

    std::cout << "\n查询范围过滤结果:" << std::endl;
    std::cout << "  原始查询帧数: " << query_data.size() << std::endl;
    std::cout << "  有效帧数:     " << filtered.size() << std::endl;
    std::cout << "  跳过帧数:     " << skipped_count << std::endl;
    std::cout << "  有效范围:     " << cfg.query_start_image_id 
              << " ~ " << cfg.query_end_image_id << std::endl;

    if (filtered.empty()) {
        throw std::runtime_error("查询范围过滤后无有效数据！请检查范围设置。");
    }

    return filtered;
}

// ============================================================================
// 步骤2：执行Viterbi匹配
// ============================================================================

bool runMatching(const Config& cfg) {
    printSeparator("步骤2: 执行Viterbi匹配");

    try {
        // 如果地图未构建，尝试从缓存加载
        if (visualMapVec.empty()) {
            if (cfg.load_from_cache && fs::exists(cfg.map_cache_binary)) {
                std::cout << "地图未加载，从缓存加载..." << std::endl;
                visualMapVec = VisualMapData::loadMapFromFile(cfg.map_cache_binary);
                std::cout << "从缓存加载地图成功，共 " << visualMapVec.size() << " 帧" << std::endl;
            } else {
                std::cerr << "错误: 视觉地图为空，请先执行步骤1或启用缓存加载" << std::endl;
                return false;
            }
        }

        // ✅ 加载查询数据（直接传递范围参数）
        std::cout << "\n加载查询数据..." << std::endl;
        queryDataVec = VisualQueryData::buildQueryFromCSV(
            cfg.query_csv_path,
            cfg.query_images_folder,
            cfg.enable_query_range,        // 是否启用范围过滤
            cfg.query_start_image_id,      // 起始frame_id
            cfg.query_end_image_id         // 结束frame_id
        );

        std::cout << "\n最终用于匹配的查询帧数: " << queryDataVec.size() << std::endl;

        // 清空回溯文件
        std::ofstream(cfg.backtrack_file, std::ios::trunc).close();

        // 计算目标帧数（基于查询范围）
        int target_frames = cfg.enable_query_range
            ? (cfg.query_end_image_id - cfg.query_start_image_id + 1)
            : queryDataVec.size();

        // 准备PnP配置
        PnPConfig pnp_config;
        if (cfg.enable_pnp_refinement) {
            pnp_config.enable = true;
            pnp_config.min_inliers = cfg.pnp_min_inliers;
            pnp_config.ransac_reproj_error = cfg.pnp_ransac_reproj_error;

            // 构建相机内参矩阵
            pnp_config.camera_matrix = (cv::Mat_<double>(3, 3) <<
                cfg.camera_fx, 0, cfg.camera_cx,
                0, cfg.camera_fy, cfg.camera_cy,
                0, 0, 1
            );

            // 畸变系数（假设无畸变）
            pnp_config.dist_coeffs = cv::Mat::zeros(5, 1, CV_64F);
        }

        // 执行Viterbi匹配
        ViterbiMatch(
            cfg.result_csv_path,
            cfg.backtrack_file,
            visualMapVec,
            queryDataVec,
            target_frames,
            cfg.use_motion_prediction,  // 传递查询模式参数
            cfg.enable_pnp_refinement ? &pnp_config : nullptr  // 传递PnP配置
        );

        return true;
    } catch (const std::exception& e) {
        std::cerr << "\n匹配执行失败: " << e.what() << std::endl;
        return false;
    }
}

int main() {
    // 创建配置实例
    Config cfg;
    
    printConfig(cfg);
    
    if (!validatePaths(cfg)) {
        return 1;
    }
    
    bool success = true;
    
    if (cfg.step1_build_map) {
        if (!buildVisualMap(cfg)) {
            success = false;
        }
    }
    
    if (success && cfg.step2_run_matching) {
        if (!runMatching(cfg)) {
            success = false;
        }
    }
    
    printSeparator("执行完成");
    if (success) {
        std::cout << "所有步骤执行成功!" << std::endl;
        
        if (cfg.step2_run_matching) {
            std::cout << "\n输出文件:" << std::endl;
            std::cout << "  匹配结果: " << cfg.result_csv_path << std::endl;
            std::cout << "  回溯文件: " << cfg.backtrack_file << std::endl;
        }
        
        if (cfg.save_map_cache) {
            std::cout << "\n缓存文件:" << std::endl;
            std::cout << "  二进制:   " << cfg.map_cache_binary << std::endl;
            std::cout << "  文本信息: " << cfg.map_cache_text << std::endl;
        }
    } else {
        std::cout << "执行过程中出现错误" << std::endl;
    }
    
    return success ? 0 : 1;
}