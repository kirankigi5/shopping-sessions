# Shopping Sessions: native Rust pipeline, built with profile-guided optimization (PGO).
# The profile is collected during the build by running an instrumented binary on a generated
# 2M-event set. If any PGO step fails, the plain build is used instead.
FROM rust:1-bookworm AS build
ARG TARGETARCH
WORKDIR /src
COPY gen ./gen
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN if [ "$TARGETARCH" = "amd64" ]; then RF="-C target-cpu=x86-64-v3"; else RF=""; fi; \
    RUSTFLAGS="$RF" cargo build --release --locked && cp target/release/solution /solution-plain
RUN if [ "$TARGETARCH" = "amd64" ]; then RF="-C target-cpu=x86-64-v3"; else RF=""; fi; \
    ( rustup component add llvm-tools-preview && \
      (cd gen && cargo build --release) && \
      RUSTFLAGS="$RF -Cprofile-generate=/pgo" cargo build --release --locked --target-dir /tg && \
      gen/target/release/generate --seed 3 --events 2000000 --clicks 500000 --output /pgodata >/dev/null && \
      mkdir -p /pgoout && DATA_DIR=/pgodata OUTPUT_DIR=/pgoout /tg/release/solution && \
      $(find / -name llvm-profdata -path '*rustlib*' | head -1) merge -o /pgo.profdata /pgo && \
      RUSTFLAGS="$RF -Cprofile-use=/pgo.profdata" cargo build --release --locked --target-dir /tp && \
      cp /tp/release/solution /solution && echo "PGO build OK" ) \
    || (echo "PGO failed: using plain build" && cp /solution-plain /solution)

FROM debian:bookworm-slim
COPY --from=build /solution /app/solution
ENV DATA_DIR=/data OUTPUT_DIR=/output SCRATCH_DIR=/scratch MIMALLOC_ALLOW_LARGE_OS_PAGES=1
CMD ["/app/solution"]
