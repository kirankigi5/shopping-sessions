# Shopping Sessions: native Rust pipeline (DIAG build: benchmarks then exits 1).
FROM rust:1-bookworm AS build
ARG TARGETARCH
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY src ./src
COPY gen ./gen
RUN if [ "$TARGETARCH" = "amd64" ]; then export RUSTFLAGS="-C target-cpu=x86-64-v3"; fi; \
    cargo build --release --locked --features diag && cp target/release/solution /solution && \
    cd gen && cargo build --release && cp target/release/generate /generate

FROM debian:bookworm-slim
COPY --from=build /solution /app/solution
COPY --from=build /generate /app/generate
ENV DATA_DIR=/data OUTPUT_DIR=/output SCRATCH_DIR=/scratch
CMD ["/app/solution"]
