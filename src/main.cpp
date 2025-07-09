#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <filesystem>
#include <nlohmann/json.hpp>

#include <numeric>
#include <cmath>
#include <unordered_map>
#include <unordered_set>
#include <limits>
#include <functional>

#include <opencv2/opencv.hpp>
#include <iomanip>

#include <queue>
#include <set>

#include "tracking_demo.h"
#include "input_output.h"

namespace fs = std::filesystem;
using json = nlohmann::json;


int main(int argc, char* argv[]) {
    std::string input_path = "C:/data/input_data_med.json";
    std::string output_path = "C:/data/tracking_output.json";
    std::string vis_dir = "C:/data/vis";

    if (!parse_args(argc, argv, input_path, output_path, vis_dir)) {
        std::cerr << "Usage: tracking-solution --input <path> --output <path> --vis-dir <dir>" << std::endl;
        return 1;
    }

    fs::create_directories(vis_dir);

    std::ifstream input_file(input_path);
    if (!input_file) {
        std::cerr << "Failed to open input file: " << input_path << std::endl;
        return 1;
    }

    json input_json, output_json;
    input_file >> input_json;

    std::map<int, std::string> frame_timestamps;
    std::vector<std::vector<Detection>> all_frames = parse_input_json(input_json, frame_timestamps);

    try {
        std::cout << "run_astar_tracking" << std::endl;
        TrackingState solution = run_astar_tracking(all_frames);
        output_json = build_output_json(solution, frame_timestamps);
    }
    catch (const std::exception& e) {
        std::cerr << "[FATAL] Tracking failed: " << e.what() << std::endl;
        return 1;
    }

    generate_frame_visualizations(output_json, vis_dir);
    generate_track_visualizations(output_json, vis_dir);
    generate_summary_image(output_json, vis_dir);

    std::ofstream output_file(output_path);
    if (!output_file) {
        std::cerr << "Failed to open output file: " << output_path << std::endl;
        return 1;
    }

    output_file << output_json.dump(2);
    std::cout << "Wrote output to: " << output_path << std::endl;

    return 0;
}
