# The llama.cpp server image Vast rents, with an SSH server already in it.
#
# NOT PUBLISHED AND NOT USED. Kept as the record of why a rental waits before
# answering SSH, and as the recipe if that trade is ever worth making. Vast
# rents a digest-pinned upstream image today; see docs/vast.md before changing
# that.
#
# Vast's `runtype: ssh` provisioning has to produce an sshd before a rental can
# be reached, and the upstream image carries none: it is Ubuntu 24.04 with
# `/app/llama-server` as its entrypoint and no openssh package at all. The host
# therefore runs apt against archive.ubuntu.com at launch, which is the work
# visible in an instance's status_msg and the reason sshd binds minutes after
# the container reports `running`.
#
# Only the packages are added. The entrypoint, CUDA libraries and curl the
# projector staging relies on are inherited unchanged, so the runtime this
# serves is byte-for-byte the upstream one plus openssh.
#
# Keep the base digest in sync with llamacpp_runtime_support.json, and the
# published digest in sync with the llamacpp entry of vast_runtime.json.
FROM ghcr.io/ggml-org/llama.cpp@sha256:e52c610406cd18714902d1ca3bffadebca4a2a8370faaba8a5be5cc5d5203921
RUN apt-get update \
    && apt-get install -y --no-install-recommends openssh-server \
    && mkdir -p /run/sshd \
    && rm -rf /var/lib/apt/lists/*
