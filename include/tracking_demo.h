#pragma once

#include <cmath>
#include <vector>
#include <map>
#include <unordered_map>
#include <limits>
#include <algorithm>
#include <string>

struct AssignmentOption {
    enum Type { EXISTING_TRACK, NEW_TRACK };
    Type type;
    int track_id;  // only valid if EXISTING_TRACK
};


struct Detection {
    int frame_index;
    int detection_index;
    double x, y, width, height;
};


struct TrackHistory {
    std::vector<Detection> detections;
    int last_frame_seen = -1;
};

struct TrackingState {
    int current_frame = 0;
    std::map<int, int> assigned_ids; // detection_index -> track_id
    std::unordered_map<int, TrackHistory> track_histories;
    double cost_so_far = 0.0;
    double est_total_cost = 0.0;

    bool operator>(const TrackingState& other) const {
        return est_total_cost > other.est_total_cost;
    }
};

double euclidean_distance(double x1, double y1, double x2, double y2);

double Pg(int gap);

double Pd(double distance, double h1, double w1);

double Ps(double w1, double h1, double w2, double h2);

double match_cost(const Detection& prev, const Detection& current);

TrackingState run_astar_tracking(
    const std::vector<std::vector<Detection>>& all_frames
);
