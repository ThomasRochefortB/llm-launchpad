# GLM-5.3-Flash: pinned implementation of ggml-org/llama.cpp PR #27754.
# Keep the source revision and checksum in sync with llamacpp_glm5next_support.json.
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04 AS build
ARG CUDA_ARCHITECTURES=75;80;86;89;90;100;120
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl cmake g++ make libssl-dev && rm -rf /var/lib/apt/lists/*
WORKDIR /src
RUN curl -fL --retry 3 https://codeload.github.com/unslothai/llama.cpp/tar.gz/629b50552801912b3e2078f9799e4d77213197d7 -o source.tar.gz \
    && echo 'ce83b6acece1789d585a9af111f7cc34f96582a1baad31f7b6ed3d506764faf7  source.tar.gz' | sha256sum -c - \
    && tar -xzf source.tar.gz --strip-components=1 && rm source.tar.gz
RUN cmake -S . -B build -DGGML_CUDA=ON -DGGML_NATIVE=OFF \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" -DLLAMA_BUILD_TESTS=OFF \
    && cmake --build build --target llama-server llama-fit-params -j 4

FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libgomp1 libssl3 && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/build/bin/ /app/
ENV LD_LIBRARY_PATH=/app
# Required by the pinned GLM implementation for accurate fp32 GEMMs.
ENV NVIDIA_TF32_OVERRIDE=0
ENTRYPOINT ["/app/llama-server"]
