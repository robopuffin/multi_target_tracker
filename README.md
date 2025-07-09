# Tracking Demo (C++20 + OpenCV)

This is an object tracking system built for a technical challenge. It uses A*-like assignment search and generates frame-by-frame tracking and visualizations from JSON input.

---

## 🚀 Run It Instantly (No Build Required)

You can run this project using Docker without compiling anything:

```bash
docker pull ghcr.io/robopuffin/tracking-solution:latest

docker run -v $(pwd):/data ghcr.io/robopuffin/tracking-solution \
  --input /data/input_data.json \
  --output /data/tracking_output.json \
  --vis-dir /data/visualization
