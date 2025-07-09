FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential cmake libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN cmake -Bbuild -DCMAKE_BUILD_TYPE=Release && cmake --build build


ENTRYPOINT ["./build/tracking_demo"]
