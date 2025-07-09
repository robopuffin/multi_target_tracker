# Tracking Demo (C++20 + OpenCV)

This is an object tracking system built for a technical challenge. It uses A*-like assignment search and generates frame-by-frame tracking and visualizations from JSON input.

## 🔧 Requirements

- Docker
- Input JSON in the expected format

## 🚀 Run It (exact command)

From a folder containing `input_data.json`:

```bash
docker run -v $(pwd):/data tracking-solution \
  --input /data/input_data.json \
  --output /data/tracking_output.json \
  --vis-dir /data/visualization
