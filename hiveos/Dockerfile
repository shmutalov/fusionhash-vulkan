# Builds the Linux x86_64 vulkminer binary. No Vulkan SDK / OpenSSL needed:
# build.rs falls back to the committed SPIR-V and the pool client uses rustls.
FROM rust:1-bookworm AS build
WORKDIR /src
COPY . .
RUN cargo build --release
# artifact: /src/target/release/vulkminer
