# Shopping Sessions: native Rust pipeline, built with profile-guided optimization (PGO) as a static,
# non-PIE binary on an empty image (no dynamic loader or relocation work at container start).
# The PGO profile is collected during the build by running an instrumented binary on a generated
# 2M-event set. If any PGO step fails, the plain (still static) build is used instead.
FROM rust:1-bookworm AS build
ARG TARGETARCH
WORKDIR /src
COPY gen ./gen
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN case "$TARGETARCH" in \
      amd64) echo x86_64-unknown-linux-gnu > /triple; echo "-C target-cpu=x86-64-v3" > /cpuflags ;; \
      *)     echo aarch64-unknown-linux-gnu > /triple; echo "" > /cpuflags ;; \
    esac
RUN T=$(cat /triple); RF="$(cat /cpuflags) -C target-feature=+crt-static -C relocation-model=static"; \
    RUSTFLAGS="$RF" cargo build --release --locked --target $T && cp target/$T/release/solution /solution-plain
RUN T=$(cat /triple); RF="$(cat /cpuflags) -C target-feature=+crt-static -C relocation-model=static"; \
    ( rustup component add llvm-tools-preview && \
      (cd gen && cargo build --release) && \
      RUSTFLAGS="$RF -Cprofile-generate=/pgo" cargo build --release --locked --target $T --target-dir /tg && \
      gen/target/release/generate --seed 3 --events 2000000 --clicks 500000 --output /pgodata >/dev/null && \
      mkdir -p /pgoout && DATA_DIR=/pgodata OUTPUT_DIR=/pgoout /tg/$T/release/solution && \
      $(find / -name llvm-profdata -path '*rustlib*' | head -1) merge -o /pgo.profdata /pgo && \
      RUSTFLAGS="$RF -Cprofile-use=/pgo.profdata" cargo build --release --locked --target $T --target-dir /tp && \
      cp /tp/$T/release/solution /solution && echo "PGO build OK" ) \
    || (echo "PGO failed: using plain build" && cp /solution-plain /solution)
# the final image has no libraries: refuse to ship a dynamically linked binary
RUN file /solution && ! (file /solution | grep -q "dynamically linked")

FROM scratch
COPY --from=build /solution /app/solution
ENV DATA_DIR=/data OUTPUT_DIR=/output SCRATCH_DIR=/scratch
CMD ["/app/solution"]
