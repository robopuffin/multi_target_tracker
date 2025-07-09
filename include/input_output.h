#pragma once
#include <string>
#include <map>
#include <vector>
#include <nlohmann/json.hpp>
#include "tracking_demo.h"

using json = nlohmann::json;

bool parse_args(int argc, char* argv[], std::string& input_path, std::string& output_path, std::string& vis_dir);

std::vector<std::vector<Detection>> parse_input_json(
    const json& input_json,
    std::map<int, std::string>& frame_timestamps
);

json build_output_json(
    const TrackingState& solution,
    const std::map<int, std::string>& frame_timestamps
);

void generate_frame_visualizations(const json& output_json, const std::string& vis_dir);
void generate_track_visualizations(const json& output_json, const std::string& vis_dir);
void generate_summary_image(const json& output_json, const std::string& vis_dir);
