# Shopping Sessions: native Rust pipeline.
FROM rust:1-bookworm AS build
ARG TARGETARCH
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN if [ "$TARGETARCH" = "amd64" ]; then export RUSTFLAGS="-C target-cpu=x86-64-v3"; fi; \
    cargo build --release --locked --features diag && cp target/release/solution /solution

FROM debian:bookworm-slim
COPY --from=build /solution /app/solution
ENV DATA_DIR=/data OUTPUT_DIR=/output SCRATCH_DIR=/scratch MIMALLOC_ALLOW_LARGE_OS_PAGES=1
CMD ["/app/solution"]
