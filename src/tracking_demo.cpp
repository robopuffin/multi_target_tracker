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

struct PairHash {
    std::size_t operator()(const std::pair<int, int>& p) const {
        return std::hash<int>()(p.first) ^ (std::hash<int>()(p.second) << 1);
    }
};

double euclidean_distance(double x1, double y1, double x2, double y2) {
    return std::sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
}

double Pg(int gap) {
    const double p1 = 0.8;
    if (gap < 1 || gap > 3) return 0.0;
    return std::pow(p1, gap);
}

double Pd(double distance, double h1, double w1) {
    const double Pd1 = 0.3;
    double norm_d = distance / std::max(h1, w1);
    return std::pow(Pd1, norm_d);
}

double Ps(double w1, double h1, double w2, double h2) {
    const double Ps1 = 0.5;
    double s1 = w1 * h1;
    double s2 = w2 * h2;
    if (s1 <= 0.0 || s2 <= 0.0) return 0.0;
    double ratio = s2 / s1;
    double log2_ratio = std::log2(ratio);
    return std::pow(Ps1, std::abs(log2_ratio));
}

double match_cost(const Detection& prev, const Detection& current) {
    int gap = current.frame_index - prev.frame_index;
    if (gap < 1 || gap > 3) return std::numeric_limits<double>::infinity();

    double d = euclidean_distance(prev.x, prev.y, current.x, current.y);
    double pd = Pd(d, prev.height, prev.width);
    double ps = Ps(prev.width, prev.height, current.width, current.height);
    double pg = Pg(gap);

    double p_total = pg * pd * ps;
    if (p_total <= 1e-9) return std::numeric_limits<double>::infinity();

    return -std::log(p_total);
}

std::vector<AssignmentOption> get_track_options(const Detection& det, const TrackingState& state) {
    struct ScoredTrack {
        int track_id;  // -1 means NEW_TRACK
        double cost;
    };

    std::vector<ScoredTrack> candidates;

    // Evaluate cost for all valid EXISTING_TRACK options
    for (const auto& [track_id, history] : state.track_histories) {
        const Detection& last = history.detections.back();
        int gap = det.frame_index - last.frame_index;
        if (gap < 1 || gap > 3) continue;

        double max_extent = std::max(last.width, last.height);
        double move_thresh = std::clamp(2 * max_extent, 0.01, 0.15);
        double dist = euclidean_distance(last.x, last.y, det.x, det.y);
        if (dist > move_thresh) continue;

        double s1 = last.width * last.height;
        double s2 = det.width * det.height;
        if (s1 <= 0.0 || s2 <= 0.0) continue;

        double ratio = s2 / s1;
        if (ratio < 0.833 || ratio > 1.20) continue;

        double cost = match_cost(last, det);
        if (cost < std::numeric_limits<double>::infinity())
            candidates.push_back({ track_id, cost });
    }

    // Add NEW_TRACK as a scored alternative
    const double NEW_TRACK_COST = 5.0;  // Tuneable cost (higher discourages new tracks)
    candidates.push_back({ -1, NEW_TRACK_COST });

    // Sort all candidates by increasing cost
    std::sort(candidates.begin(), candidates.end(),
        [](const ScoredTrack& a, const ScoredTrack& b) {
            return a.cost < b.cost;
        });

    // Select top options within tie threshold
    std::vector<AssignmentOption> options;
    const double TIE_THRESHOLD = 0.167;
    const size_t MAX_BRANCHING = 5;

    double best_cost = candidates.front().cost;
    size_t count = 0;

    for (const auto& c : candidates) {
        if (c.cost - best_cost > TIE_THRESHOLD) break;
        if (count >= MAX_BRANCHING) break;

        AssignmentOption opt;
        opt.type = (c.track_id == -1) ? AssignmentOption::NEW_TRACK : AssignmentOption::EXISTING_TRACK;
        opt.track_id = c.track_id;
        options.push_back(opt);
        count++;
    }

    bool has_new_track = false;
    for (const auto& opt : options) {
        if (opt.type == AssignmentOption::NEW_TRACK) {
            has_new_track = true;
            break;
        }
    }

    if (!has_new_track) {
        // Force NEW_TRACK as a backup if not already included
        options.push_back({ AssignmentOption::NEW_TRACK, -1 });
    }

    for (const auto& c : candidates) {
        std::cerr << "  candidate track_id=" << c.track_id
            << " cost=" << c.cost
            << " delta=" << c.cost - best_cost
            << (c.track_id == -1 ? " (NEW)" : "")
            << "\n";
    }

    return options;
}


std::vector<std::vector<AssignmentOption>> compute_assignment_options_global(
    const std::vector<Detection>& detections,
    const TrackingState& state
) {
    constexpr size_t MAX_CANDIDATES = 5;
    constexpr double TIE_THRESHOLD = 0.2;
    constexpr double NEW_TRACK_COST = 5.0;

    struct ScoredMatch {
        int track_id;
        double cost;
    };

    // Detection → top scoring matches
    std::unordered_map<int, std::vector<ScoredMatch>> det_to_tracks;

    // Track → best detection it's linked to
    std::unordered_map<int, std::pair<int, double>> track_to_best_det;

    // --- Pass 1: Evaluate all track-detection costs
    for (size_t det_idx = 0; det_idx < detections.size(); ++det_idx) {
        const Detection& det = detections[det_idx];
        std::vector<ScoredMatch> candidates;

        for (const auto& [track_id, history] : state.track_histories) {
            const Detection& last = history.detections.back();
            int gap = det.frame_index - last.frame_index;
            if (gap < 1 || gap > 3) continue;

            double max_extent = std::max(last.width, last.height);
            double move_thresh = std::clamp(2.0 * max_extent, 0.01, 0.15);
            double dist = euclidean_distance(last.x, last.y, det.x, det.y);
            if (dist > move_thresh) continue;

            double s1 = last.width * last.height;
            double s2 = det.width * det.height;
            if (s1 <= 0.0 || s2 <= 0.0) continue;

            double ratio = s2 / s1;
            if (ratio < 0.833 || ratio > 1.2) continue;

            double cost = match_cost(last, det);
            if (cost < std::numeric_limits<double>::infinity()) {
                candidates.push_back({ track_id, cost });

                // Update best detection for this track
                if (!track_to_best_det.count(track_id) || cost < track_to_best_det[track_id].second) {
                    track_to_best_det[track_id] = { static_cast<int>(det_idx), cost };
                }
            }
        }

        // Always add NEW_TRACK
        candidates.push_back({ -1, NEW_TRACK_COST });

        std::sort(candidates.begin(), candidates.end(), [](auto& a, auto& b) {
            return a.cost < b.cost;
            });

        if (candidates.size() > MAX_CANDIDATES)
            candidates.resize(MAX_CANDIDATES);

        det_to_tracks[det_idx] = std::move(candidates);
    }

    // --- Pass 2: Prune based on ownership and margin
    std::vector<std::vector<AssignmentOption>> final_options(detections.size());

    for (size_t det_idx = 0; det_idx < detections.size(); ++det_idx) {
        const auto& det_candidates = det_to_tracks[det_idx];
        double best_cost = det_candidates.front().cost;

        for (const auto& match : det_candidates) {
            if (match.cost - best_cost > TIE_THRESHOLD) break;

            if (match.track_id == -1) {
                final_options[det_idx].push_back({ AssignmentOption::NEW_TRACK, -1 });
                continue;
            }

            const auto& [best_det_for_track, track_best_cost] = track_to_best_det.at(match.track_id);
            if (best_det_for_track == static_cast<int>(det_idx)) {
                final_options[det_idx].push_back({ AssignmentOption::EXISTING_TRACK, match.track_id });
            }
        }

        if (final_options[det_idx].empty()) {
            final_options[det_idx].push_back({ AssignmentOption::NEW_TRACK, -1 });
        }
    }

    return final_options;
}



void enumerate_assignments(
    const TrackingState& state,
    const std::vector<Detection>& detections,
    const std::vector<std::vector<AssignmentOption>>& options,
    int index,
    std::vector<AssignmentOption>& current,
    std::unordered_set<int>& used_tracks,
    std::vector<std::vector<AssignmentOption>>& results
) {
    if (index == options.size()) {
        results.push_back(current);
        return;
    }

    for (const auto& opt : options[index]) {
        if (opt.type == AssignmentOption::EXISTING_TRACK &&
            used_tracks.count(opt.track_id)) {
            continue; // can't reuse a track in this assignment
        }

        current[index] = opt;
        if (opt.type == AssignmentOption::EXISTING_TRACK)
            used_tracks.insert(opt.track_id);

        enumerate_assignments(state, detections, options, index + 1, current, used_tracks, results);

        if (opt.type == AssignmentOption::EXISTING_TRACK)
            used_tracks.erase(opt.track_id);
    }
}

std::vector<TrackingState> generate_successors(
    const TrackingState& state,
    const std::vector<Detection>& detections,
    int& global_track_id_counter
) {
    std::vector<TrackingState> successors;
    std::vector<std::vector<AssignmentOption>> all_options;

    // Step 1: Gather assignment options for each detection
    all_options = compute_assignment_options_global(detections, state);

    // Step 2: Enumerate all valid joint assignments
    std::vector<AssignmentOption> current(detections.size());
    std::unordered_set<int> used;
    std::vector<std::vector<AssignmentOption>> assignments;

    enumerate_assignments(state, detections, all_options, 0, current, used, assignments);

    // Sort assignments by heuristic (lower cost preferred)
    std::unordered_map<std::pair<int, int>, double, PairHash> cost_cache;

    for (const auto& assignment : assignments) {
        for (size_t i = 0; i < assignment.size(); ++i) {
            const auto& opt = assignment[i];
            if (opt.type == AssignmentOption::EXISTING_TRACK) {
                auto key = std::make_pair(opt.track_id, detections[i].detection_index);
                if (!cost_cache.count(key)) {
                    const Detection& prev = state.track_histories.at(opt.track_id).detections.back();
                    cost_cache[key] = match_cost(prev, detections[i]);
                }
            }
        }
    }


    // --- Sort assignments by total cost (cached) ---
    auto cost_comparator = [&](const std::vector<AssignmentOption>& a1, const std::vector<AssignmentOption>& a2) {
        double cost1 = 0.0, cost2 = 0.0;
        for (size_t i = 0; i < a1.size(); ++i) {
            if (a1[i].type == AssignmentOption::EXISTING_TRACK) {
                auto key = std::make_pair(a1[i].track_id, detections[i].detection_index);
                cost1 += cost_cache[key];
            }
            else {
                cost1 += 5.0;
            }

            if (a2[i].type == AssignmentOption::EXISTING_TRACK) {
                auto key = std::make_pair(a2[i].track_id, detections[i].detection_index);
                cost2 += cost_cache[key];
            }
            else {
                cost2 += 5.0;
            }
        }
        return cost1 < cost2;
        };

    // --- Partial sort to retain top K ---
    const size_t MAX_ASSIGNMENTS = 100;
    if (assignments.size() > MAX_ASSIGNMENTS) {
        std::partial_sort(assignments.begin(), assignments.begin() + MAX_ASSIGNMENTS, assignments.end(), cost_comparator);
        assignments.resize(MAX_ASSIGNMENTS);
    }
    else {
        std::sort(assignments.begin(), assignments.end(), cost_comparator);
    }

    if (detections.empty()) {
        std::cerr << "[ERROR] No detections for frame " << state.current_frame << "\n";
    }


    for (const auto& opts : all_options) {
        if (opts.empty()) {
            std::cerr << "[ERROR] A detection has no assignment options\n";
            return {}; // bail early
        }
    }




    // Step 3: Build successors
    for (const auto& assign : assignments) {
        TrackingState next = state;
        next.current_frame = state.current_frame + 1;

        for (size_t i = 0; i < detections.size(); ++i) {
            const auto& det = detections[i];
            const auto& opt = assign[i];

            int tid;
            if (opt.type == AssignmentOption::NEW_TRACK) {
                tid = global_track_id_counter++;
                next.cost_so_far += 5.0;
            }
            else {

                tid = opt.track_id;
                const Detection& prev = state.track_histories.at(tid).detections.back();
                double cost = match_cost(prev, det);
                next.cost_so_far += cost;
            }

            next.assigned_ids[det.detection_index] = tid;
            next.track_histories[tid].detections.push_back(det);
            next.track_histories[tid].last_frame_seen = det.frame_index;
        }

        next.est_total_cost = next.cost_so_far;
        successors.push_back(std::move(next));
    }

    return successors;
}


// Priority queue element wrapper
struct CompareState {
    bool operator()(const TrackingState& a, const TrackingState& b) const {
        return a.est_total_cost > b.est_total_cost;
    }
};



TrackingState run_astar_tracking(
    const std::vector<std::vector<Detection>>& all_frames
) {
    constexpr size_t BEAM_WIDTH = 1000;
    constexpr double MAX_COST_DELTA = 5.0;

    int global_track_id = 0;

    std::priority_queue<TrackingState, std::vector<TrackingState>, CompareState> open;
    TrackingState start;
    start.current_frame = 1; // move on from frame 0

    std::cout << "Grabbing Detections" << std::endl;
    for (const auto& detection : all_frames[0]) {
        int new_id = global_track_id++;
        start.assigned_ids[detection.detection_index] = new_id;
        start.track_histories[new_id].detections.push_back(detection);
        start.track_histories[new_id].last_frame_seen = detection.frame_index;
    }
    open.push(start);

    size_t last_pruned_frame = start.current_frame;

    while (!open.empty()) {
        TrackingState current = open.top();
        open.pop();

        // Skip bad states that are too far behind
        if (!open.empty() && current.est_total_cost > open.top().est_total_cost + MAX_COST_DELTA) {
            std::cerr << "[DEBUG] Pruning node with cost " << current.est_total_cost
                << " (best is " << open.top().est_total_cost << ")\n";
            continue;
        }

        // Goal check
        if (current.current_frame >= all_frames.size()) {
            bool all_assigned = true;
            for (const auto& frame : all_frames) {
                for (const auto& det : frame) {
                    if (current.assigned_ids.find(det.detection_index) == current.assigned_ids.end()) {
                        all_assigned = false;
                        break;
                    }
                }
                if (!all_assigned) break;
            }

            if (all_assigned) return current;
            else continue;
        }

        // Generate successors
        const auto& detections = all_frames[current.current_frame];
        auto successors = generate_successors(current, detections, global_track_id);

        for (auto& next : successors) {
            int remaining_frames = static_cast<int>(all_frames.size()) - next.current_frame;
            double min_cost = -std::log(0.8 * 1.0 * 0.5); // conservative best-case cost
            next.est_total_cost = next.cost_so_far + remaining_frames * min_cost;
            open.push(std::move(next));
        }

        // === BEAM PRUNING: Only once per frame ===
        if (!open.empty() && open.top().current_frame > last_pruned_frame) {
            std::vector<TrackingState> best;
            best.reserve(BEAM_WIDTH);

            // Pull best BEAM_WIDTH states
            while (!open.empty() && best.size() < BEAM_WIDTH) {
                best.push_back(std::move(open.top()));
                open.pop();
            }

            // Discard the rest
            while (!open.empty()) open.pop();

            // Reinsert only top
            for (auto& s : best) {
                open.push(std::move(s));
            }

            std::cerr << "[DEBUG] Beam pruning at frame " << last_pruned_frame
                << ": kept " << best.size() << " best states\n";

            last_pruned_frame++;
        }
    }

    throw std::runtime_error("A* search failed to find a valid tracking configuration.");
}


