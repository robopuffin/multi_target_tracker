#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <set>
#include <utility>      // for std::pair
#include <sstream>
#include <iomanip>      // for std::setw, std::setfill

#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>

#include "input_output.h"

using json = nlohmann::json;





// Parse command-line arguments
bool parse_args(int argc, char* argv[], std::string& input_path, std::string& output_path, std::string& vis_dir) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--input" && i + 1 < argc) input_path = argv[++i];
        else if (arg == "--output" && i + 1 < argc) output_path = argv[++i];
        else if (arg == "--vis-dir" && i + 1 < argc) vis_dir = argv[++i];
        else {
            std::cerr << "Unknown or incomplete argument: " << arg << std::endl;
            return false;
        }
    }

    std::cerr << "[DEBUG] input_path = " << input_path << "\n";
    std::cerr << "[DEBUG] output_path = " << output_path << "\n";
    std::cerr << "[DEBUG] vis_dir = " << vis_dir << "\n";

    return true;
}

std::vector<std::vector<Detection>> parse_input_json(
    const json& input_json,
    std::map<int, std::string>& frame_timestamps
) {
    std::vector<std::vector<Detection>> all_frames;
    int detection_index = 0;

    for (const auto& frame : input_json) {
        std::vector<Detection> frame_detections;
        int frame_id = frame["frame_id"];
        frame_timestamps[frame_id] = frame["timestamp"];
        for (const auto& det : frame["detections"]) {
            Detection d;
            d.frame_index = frame_id;
            d.detection_index = detection_index++;
            d.x = det["x"];
            d.y = det["y"];
            d.width = det["width"];
            d.height = det["height"];
            frame_detections.push_back(d);
        }

        all_frames.push_back(std::move(frame_detections));
    }

    return all_frames;
}



json build_output_json(
    const TrackingState& solution,
    const std::map<int, std::string>& frame_timestamps
) {
    // Build mapping: frame_id -> tracked objects
    std::map<int, std::vector<json>> frame_map;

    for (const auto& [track_id, history] : solution.track_histories) {
        for (const auto& det : history.detections) {
            json obj = {
                { "x", det.x },
                { "y", det.y },
                { "width", det.width },
                { "height", det.height },
                { "id", track_id }
            };
            frame_map[det.frame_index].emplace_back(std::move(obj));
        }
    }

    // Assemble output frames
    json output_json = json::array();
    for (const auto& [frame_id, objects] : frame_map) {
        json frame;
        frame["frame_id"] = frame_id;
        frame["timestamp"] = frame_timestamps.count(frame_id) ? frame_timestamps.at(frame_id) : "unknown";
        frame["tracked_objects"] = objects;
        output_json.emplace_back(std::move(frame));
    }

    // Sort frames by frame_id
    std::sort(output_json.begin(), output_json.end(),
        [](const json& a, const json& b) {
            return a["frame_id"].get<int>() < b["frame_id"].get<int>();
        });

    return output_json;
}


void generate_frame_visualizations(const json& output_json, const std::string& vis_dir) {
    const int img_size = 800;

    for (const auto& frame : output_json) {
        int frame_id = frame["frame_id"];
        const auto& tracked_objects = frame["tracked_objects"];

        cv::Mat vis = cv::Mat::zeros(img_size, img_size, CV_8UC3);

        for (const auto& obj : tracked_objects) {
            double x = obj["x"];
            double y = obj["y"];
            double w = obj["width"];
            double h = obj["height"];
            int id = obj["id"];

            int cx = static_cast<int>(x * img_size);
            int cy = static_cast<int>(y * img_size);
            int box_w = static_cast<int>(w * img_size);
            int box_h = static_cast<int>(h * img_size);

            cv::Rect rect(cx - box_w / 2, cy - box_h / 2, box_w, box_h);
            cv::rectangle(vis, rect, cv::Scalar(0, 255, 255), 1);
            cv::putText(vis, std::to_string(id), { cx + 12, cy },
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        }

        std::ostringstream fname;
        fname << vis_dir << "/frame_" << std::setw(4) << std::setfill('0') << frame_id << ".png";
        cv::imwrite(fname.str(), vis);
    }
}



void generate_track_visualizations(const json& output_json, const std::string& vis_dir) {
    const int img_size = 800;

    // Build track ID → list of (frame_id, detection)
    std::unordered_map<int, std::vector<std::pair<int, json>>> tracks;

    for (const auto& frame : output_json) {
        int frame_id = frame["frame_id"];
        for (const auto& obj : frame["tracked_objects"]) {
            int id = obj["id"];
            tracks[id].emplace_back(frame_id, obj);
        }
    }

    for (const auto& [id, detections] : tracks) {
        cv::Mat img = cv::Mat::zeros(img_size, img_size, CV_8UC3);

        for (const auto& [frame_id, det] : detections) {
            double x = det["x"];
            double y = det["y"];
            double w = det["width"];
            double h = det["height"];

            int cx = static_cast<int>(x * img_size);
            int cy = static_cast<int>(y * img_size);
            int box_w = static_cast<int>(w * img_size);
            int box_h = static_cast<int>(h * img_size);

            cv::Rect rect(cx - box_w / 2, cy - box_h / 2, box_w, box_h);
            cv::rectangle(img, rect, cv::Scalar(0, 255, 255), 1);

            // Label with frame ID
            cv::putText(img, std::to_string(frame_id), { rect.x, rect.y - 5 },
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        }

        std::ostringstream filename;
        filename << vis_dir << "/object_" << std::setw(4) << std::setfill('0') << id << ".png";
        cv::imwrite(filename.str(), img);
    }
}

void generate_summary_image(const json& output_json, const std::string& vis_dir) {
    const int H = 20;  // height per row
    const int W = 8;   // width per frame
    const int buffer = 50;
    const int img_size = 800;

    std::map<int, std::vector<bool>> presence;
    std::map<int, int> lifespans;
    std::map<int, std::set<int>> frame_to_ids;

    int match_count = 0;
    int continued_count = 0;
    int total_possible_matches = 0;

    int m_frames = static_cast<int>(output_json.size());

    // Build presence matrix and stats
    for (int f = 0; f < m_frames; ++f) {
        const auto& frame = output_json[f];
        std::set<int> this_frame_ids;

        for (const auto& obj : frame["tracked_objects"]) {
            int id = obj["id"];
            presence[id].resize(m_frames, false);
            presence[id][f] = true;
            lifespans[id]++;
            this_frame_ids.insert(id);
            match_count++;
        }

        frame_to_ids[f] = this_frame_ids;
        if (f > 0) {
            for (int id : this_frame_ids) {
                if (frame_to_ids[f - 1].count(id)) {
                    continued_count++;
                }
            }
        }

        total_possible_matches += std::min(frame_to_ids[f].size(), frame_to_ids[f - 1].size());
    }

    int n_tracks = static_cast<int>(presence.size());
    int canvas_width = std::max(m_frames * W + buffer * 2, 600);
    int canvas_height = n_tracks * H + 150;
    cv::Mat summary_img = cv::Mat::zeros(canvas_height, canvas_width, CV_8UC3);

    // Draw timeline grid
    int row = 0;
    for (const auto& [id, track] : presence) {
        for (int f = 0; f < m_frames; ++f) {
            if (track[f]) {
                cv::Rect box(buffer + f * W, row * H, W - 1, H - 2);
                cv::rectangle(summary_img, box, cv::Scalar(0, 255, 0), cv::FILLED);
            }
        }
        cv::putText(summary_img, std::to_string(id), { 5, row * H + 15 },
            cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(200, 200, 255), 1);
        row++;
    }

    // Compute stats
    double mean_length = 0.0;
    for (auto& [id, len] : lifespans) mean_length += len;
    if (n_tracks > 0) mean_length /= n_tracks;

    double success_rate = match_count / static_cast<double>(n_tracks * m_frames);
    double continued_rate = (total_possible_matches > 0)
        ? continued_count / static_cast<double>(total_possible_matches)
        : 0.0;

    // Render stats
    int y = n_tracks * H + 30;
    cv::putText(summary_img, "Track Summary", { buffer, y },
        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2); y += 30;
    cv::putText(summary_img, "Total Tracks: " + std::to_string(n_tracks), { buffer, y },
        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 255), 1); y += 20;
    cv::putText(summary_img, "Mean Track Length: " + std::to_string(mean_length), { buffer, y },
        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 255), 1); y += 20;
    cv::putText(summary_img, "Success Rate: " + std::to_string(success_rate), { buffer, y },
        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 255), 1); y += 20;
    cv::putText(summary_img, "Continued Tracks: " + std::to_string(continued_count) + " / " + std::to_string(total_possible_matches),
        { buffer, y }, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 255), 1);

    // Save
    cv::imwrite(vis_dir + "/track_summary.png", summary_img);
}
