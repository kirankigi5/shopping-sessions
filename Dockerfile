# Shopping Sessions: native Rust pipeline, built with profile-guided optimization (PGO).
# The profile is collected during the build by running an instrumented binary on a generated
# 2M-event set. If any PGO step fails, the plain build is used instead.
FROM rust:1-bookworm AS build
ARG TARGETARCH
WORKDIR /src
COPY gen ./gen
COPY Cargo.toml Cargo.lock ./
COPY src ./src
# crt-static and relocation-model apply only to the solution binary. RUSTFLAGS
# would also hit proc-macro crates, which cannot be built that way.
RUN if [ "$TARGETARCH" = "amd64" ]; then RF="-C target-cpu=x86-64-v3"; ST="-C target-feature=+crt-static -C relocation-model=static"; else RF=""; ST=""; fi; \
    RUSTFLAGS="$RF" cargo rustc --release --locked --bin solution -- $ST && cp target/release/solution /solution-plain
RUN if [ "$TARGETARCH" = "amd64" ]; then RF="-C target-cpu=x86-64-v3"; ST="-C target-feature=+crt-static -C relocation-model=static"; else RF=""; ST=""; fi; \
    ( rustup component add llvm-tools-preview && \
      (cd gen && cargo build --release) && \
      RUSTFLAGS="$RF -Cprofile-generate=/pgo" cargo rustc --release --locked --bin solution --target-dir /tg -- $ST && \
      gen/target/release/generate --seed 3 --events 2000000 --clicks 500000 --output /pgodata >/dev/null && \
      mkdir -p /pgoout && DATA_DIR=/pgodata OUTPUT_DIR=/pgoout /tg/release/solution && \
      $(find / -name llvm-profdata -path '*rustlib*' | head -1) merge -o /pgo.profdata /pgo && \
      RUSTFLAGS="$RF -Cprofile-use=/pgo.profdata" cargo rustc --release --locked --bin solution --target-dir /tp -- $ST && \
      cp /tp/release/solution /solution && echo "PGO build OK" ) \
    || (echo "PGO failed: using plain build" && cp /solution-plain /solution)

FROM scratch
COPY --from=build /solution /app/solution
ENV DATA_DIR=/data OUTPUT_DIR=/output SCRATCH_DIR=/scratch MIMALLOC_ALLOW_LARGE_OS_PAGES=1
USER 1000:1000
CMD ["/app/solution"]
