# Shopping Sessions DIAG build: builds plain / PGO / AVX-512 variants, benchmarks them, exits 1.
FROM rust:1-bookworm AS build
WORKDIR /src
RUN rustup component add llvm-tools-preview
COPY gen ./gen
RUN cd gen && cargo build --release && cp target/release/generate /generate
COPY Cargo.toml Cargo.lock ./
COPY src ./src
# plain AVX2 build (what is scored today) + the diag driver, sharing one target dir
RUN RUSTFLAGS="-C target-cpu=x86-64-v3" cargo build --release --locked --target-dir /t3 && cp /t3/release/solution /sol-v3 && \
    RUSTFLAGS="-C target-cpu=x86-64-v3" cargo build --release --locked --features diag --target-dir /t3 && cp /t3/release/solution /diag
# PGO: instrumented build, profile on a generated 2M-event set, then optimized build
RUN RUSTFLAGS="-C target-cpu=x86-64-v3 -Cprofile-generate=/pgo" cargo build --release --locked --target-dir /tg && \
    /generate --seed 3 --events 2000000 --clicks 500000 --output /pgodata >/dev/null && \
    mkdir -p /pgoout && DATA_DIR=/pgodata OUTPUT_DIR=/pgoout /tg/release/solution && \
    $(find / -name llvm-profdata -path '*rustlib*' | head -1) merge -o /pgo.profdata /pgo && \
    RUSTFLAGS="-C target-cpu=x86-64-v3 -Cprofile-use=/pgo.profdata" cargo build --release --locked --target-dir /tp && cp /tp/release/solution /sol-pgo
# AVX-512 builds
# (explicit --target: the CPU flags then apply to the binary only, not to build scripts)
RUN RUSTFLAGS="-C target-cpu=x86-64-v4" cargo build --release --locked --target x86_64-unknown-linux-gnu --target-dir /t4 && cp /t4/x86_64-unknown-linux-gnu/release/solution /sol-v4 && \
    RUSTFLAGS="-C target-cpu=x86-64-v4 -Cprofile-use=/pgo.profdata" cargo build --release --locked --target x86_64-unknown-linux-gnu --target-dir /t4p && cp /t4p/x86_64-unknown-linux-gnu/release/solution /sol-pgo-v4
RUN (rustc --version; grep -m1 flags /proc/cpuinfo | grep -o 'avx512bw' | head -1) | tr '\n' ' ' > /build-info.txt

FROM debian:bookworm-slim
COPY --from=build /diag /app/solution
COPY --from=build /sol-v3 /sol-pgo /sol-v4 /sol-pgo-v4 /generate /build-info.txt /app/
ENV DATA_DIR=/data OUTPUT_DIR=/output SCRATCH_DIR=/scratch
CMD ["/app/solution"]
