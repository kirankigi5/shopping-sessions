# Shopping Sessions: handoff

Read this first. It covers the goal, the current state, the evidence collected so far, and what to try next.

## 1. Goal

- Competition: Builders Buildathon, track **"Databricks - Shopping Sessions"**,
  https://www.joinbuilders.org/buildathon/tracks/shopping-sessions
- **Cutoff: Sun Nov 22 2026, 11:00 AM EST (16:00 UTC).**
- Score = container run time in seconds on hidden practice data (~20M events). Lower is better. A single wrong
  output row scores nothing.
- Final: ~60M fresh events, timed 3 times, each capped at 3 minutes; median counts. The top 10 are retimed 9 times
  on an idle machine. Practice times vary up to ~20% between scoring machines.
- **Current #1: 0.5084 s. Our best: 1.1390 s** (commit `1354d69`, scored Sun 12:44 AM, entered in the final).
  We need about 2.24x more speed (cut ~630 ms).
- Submissions: 2 per hour (rolling). The user submits on the web page. Rejected submissions do not change the best
  score. The final uses the best commit unless the user picks another.

## 2. The problem (full spec: `rules.md` in this repo; always trust rules.md)

The input is mounted read-only at `/data` (tmpfs on the scorer). It holds `events/*.csv`
(`event_id,user_id,ts,ingest_ts,event_type,product_id,price_cents`), `clicks/*.csv`
(`click_id,user_id,campaign_id,ts`), `links/*.csv` (`user_id_a,user_id_b`), `products.csv`, `campaigns.csv` and
`manifest.json` (window_start/window_end). Write `/output/user_daily/*.parquet` and
`/output/campaign_daily/*.parquet`.

Key rules:
- **CSV dialect.** Only `\n` ends a line. `\r` is data. A quote may only wrap a whole field. Wrong field count or
  an empty line means the line is malformed. Spaces are significant. Digits are ASCII only.
- **Timestamps** come in three forms. F1 `YYYY-MM-DDTHH:MM:SSZ`. F2 has an offset `±HH:MM` (hours 00-23,
  minutes 00-59). F3 is epoch ms as digits only, leading zeros allowed. Event window `[ws, we)`. Click window
  `[ws-7d, we)`.
- **Links.** Union-find over link lines. A person is named by the byte-wise smallest of their IDs.
- **Events, in this order:** parse, then dedup on event_id (keep the smallest ingest_ts, *before* validation),
  then validate, then drop late events (ingest-ts > 24h), then sessionize per person (gap > 30 min starts a new
  session; equal timestamps share a session). The session's day is the UTC date of its start.
- **Clicks:** credit each session to the person's click with the latest ts in `[start-7d, start]`; ties go to the
  smallest click_id. With no such click the session is `organic`. Every campaign with a click in that range
  assists the session once.
- **Outputs.** user_daily(user_id, day DATE, sessions, events, revenue_cents, top_category nullable).
  campaign_daily(campaign_id, channel, day, sessions, purchasing_sessions, revenue_cents, assisted_sessions).
  Any integer width is accepted.

**IMPORTANT: the generator was updated mid-event.** One in eight distinct user/event/click IDs now carries a
32-hex-character suffix. IDs must be treated as opaque, variable-width strings. The current generator, rules and
dev answer are in `../generator`, `../rules.md` and `../dev-answer`. The old copies are `*-old`.

## 3. Container and scorer facts (measured on the scorer via crash diagnostics)

- linux/amd64. Xeon @ 2.80GHz with AVX2 and AVX-512. **8 vCPUs = 4 physical cores x 2 hyperthreads**
  (cpuset 0-3,16-19). 24 GB memory limit. `/data` is tmpfs (in RAM). `/scratch` and `/output` are ext4.
- Per thread, the scorer is about 2x slower than an Apple M4 core. Going from 1 to 8 threads gives only about 5x,
  because hyperthreads share cores. **Total CPU work must drop. More threads do not help** (12 threads was slower).
- Kernel `sys` time is small on the scorer, so page faults are cheap there. `read()` and mmap inputs perform
  about the same; mmap is the default.

## 4. Repository and local setup

- Repo: https://github.com/kirankigi5/shopping-sessions (public), local `~/Desktop/shoppingsessions/solution`.
  Branches: `main` (scored), `diag` (diagnostic build: ships the generator, benchmarks 20M in /scratch, prints
  timings, **exits 1 on purpose** so the scorer returns the logs).
- Rust: `~/.cargo/bin` (put it on PATH). Use `export CARGO_TARGET_DIR=$HOME/ss-target`, which keeps build output
  out of the iCloud-synced Desktop.
- Python 3.12 venv with polars, pyarrow and duckdb: `~/Desktop/shoppingsessions/.venv`.
- Bench data (outside iCloud): `~/ss-bench/big20` (seed 7, 20M), `~/ss-bench/big60` (seed 11, 60M).
  Reference-checked seeds: `/tmp/seeds/s{2,3,4,5,15,21}` with `-ref` outputs (regenerate if /tmp was cleared, see
  below). Docker volumes `ss20n` (20M) and `ssdev` (dev) exist for Linux-VM tests.
- Files:
  - `src/main.rs`: the whole pipeline.
  - `src/pq.rs`: hand-written Parquet writer.
  - `src/diag.rs`: machine diagnostics (diag feature only).
  - `tools/ref.py`: slow, literal Python reference implementation of rules.md.
  - `tools/prof.py`: summarizes samply profiles.
  - `scripts/`: the starter's fixture runner, compare script and docker_run.sh.
  - `fixtures/`: the 36 edge-case fixtures.

### Verification checklist (run every one before every push)

```sh
cd ~/Desktop/shoppingsessions
export PATH=$HOME/.cargo/bin:$PATH CARGO_TARGET_DIR=$HOME/ss-target
(cd solution && cargo test --release -q && cargo build --release)
B=$HOME/ss-target/release/solution
.venv/bin/python solution/scripts/run_fixtures.py solution/fixtures -- $B          # 36/36
rm -rf /tmp/o && mkdir /tmp/o && DATA_DIR=dev OUTPUT_DIR=/tmp/o $B && .venv/bin/python solution/scripts/compare.py /tmp/o dev-answer   # MATCH
for s in 2 3 4 5 15 21; do d=/tmp/seeds/s$s; rm -rf $d-out && mkdir $d-out && DATA_DIR=$d OUTPUT_DIR=$d-out $B 2>/dev/null; .venv/bin/python solution/scripts/compare.py $d-out $d-ref | tail -1; done
# regenerate a seed if missing: generator/target/release/generate --seed S --events 500000 --clicks 125000 --output /tmp/seeds/sS && .venv/bin/python tools/ref.py /tmp/seeds/sS /tmp/seeds/sS-ref
# amd64 image exactly as the scorer runs it:
cd solution && docker buildx build --platform linux/amd64 --load -t shopping-sessions:latest . && CPUS=8 MEM=6g ../.venv/bin/python scripts/run_fixtures.py fixtures -- scripts/docker_run.sh && rm -rf /tmp/dout && CPUS=8 MEM=6g scripts/docker_run.sh ../dev /tmp/dout && ../.venv/bin/python scripts/compare.py /tmp/dout ../dev-answer
# timing (warm, 8 threads); the log shows phase end times
DATA_DIR=$HOME/ss-bench/big20 OUTPUT_DIR=/tmp/o2 NCPU=8 $B
```

For the final, also check 60M: `/usr/bin/time -l` peak memory (currently about 5.2 GB, the limit is 24 GB) and the
time cap.

## 5. Score history (practice, 20M)

| Commit | Score | Change |
|---|---|---|
| c2d75d2 | 7.21 s | first correct version (memchr tokenizer, u128 keys) |
| 0b1b569 | 6.73 s | SIMD tokenizer (the real bottleneck was not here) |
| 521701b | 1.78 s | **long IDs went through a global mutex.** Fixed with hashed event IDs and sharded interning |
| f9c8e5b | 1.57 s | lock-free per-thread long-ID maps, mmap input |
| 88f268f | 1.32 s | separator/quote masks, date cache, prefetched person map, click IDs stored as file locations |
| 77e6594 | 1.23 s | sessions: local person IDs + counting sort |
| 18dc333 | **1.19 s** | skip freeing memory and exit at once, allocation-free key compare, windowed key encoding |
| 34cfe6c | 1.20 s | people-sized session table, borrowed click-ID compare, shared output bytes, largest buckets first |
| 1354d69 | **1.139 s (best scored)** | custom Parquet writer, dedup loser flags + sequential emit, radix sort for whales, uninitialized grouping buffers |
| 181ae4b | (unscored) | SIMD exact ID encoding (SSE4.1/NEON) + branch-free SWAR time-of-day parse; windowed keys for links. Bigger Xeon win expected (scalar code is the Xeon's weak point). |
| 4938302 | (unscored, main HEAD) | + overlap user_daily writes with session tail; `SS_DENSE`/`SS_P`/`SS_Q` env knobs; `parallel_fin` helper. |

`1354d69` is the scored best / final entry. `181ae4b` and `4938302` are newer and **not yet scored** — submit one to
get a number. The `diag` branch (commit `4972485`) is based on `4938302` plus a P/Q/dense sweep; submit it as a
diagnostic (it exits 1 on purpose to return logs).

## 6. Current architecture (src/main.rs `pipeline()`)

1. Setup: parse products and campaigns. Mmap all CSVs. Split them into ~2 MB newline-aligned chunks, largest first.
2. Links run on a background thread (`build_person_map`: single-threaded union-find, hashbrown IDs) **concurrently
   with** event parsing on `NCPU` threads (`parse_events`). Each line is tokenized (`scan_lines`, one pass:
   separator mask + quote mask; a line is valid iff its quote count == 2 x the number of quote-wrapped fields).
   Each copy is validated at parse time. A 32-byte `ERec {eid,user,ingest,ts(INVALID if bad),pc=price<<8|cat}` is
   pushed into one of P=1024 per-thread buckets by eid hash.
3. Dedup (per eid bucket): `dedup_insert` (open addressing, marks losers in `dead`), then `dedup_emit`
   (sequential walk, batch-prefetched `PersonMap` lookup, push a 16-byte `SRec {person,ts,pc}` into Q=1024 person
   buckets). The same task pool parses clicks into a 24-byte `CRec {person,cid,ts,camp}` by person bucket.
4. Sessions (per person bucket, largest first): local person IDs (growable table), counting sort into person
   groups, per-person sort (radix sort at 2048 or more), then `process_person` (sessions, user-day rows, sliding
   7-day click window for credit and assists, dense `[campaign x day]` counters).
5. Output: `pq.rs` writes one user_daily file per thread plus one campaign_daily file. Then `process::exit(0)`
   without freeing anything.

Key encoding: IDs of 1 to 10 chars from `[0-9_a-z]` become exact 6-bit packed u64s (numeric order == byte
order). Other IDs become `SLOW | hash63`, with their bytes kept in per-thread `LongMap`s (looked up via `Longs`)
for ordering and output. Long click IDs are encoded as `SLOW|file|offset|len`. `key_cmp` and `cid_cmp` compare
real bytes when either key is long.

## 7. Scorer measurements (diag `07937b3`, 20M, 8 threads, ms)

**These are pre-SIMD-keys / pre-SWAR-time numbers (commit `1354d69`).** `181ae4b`/`4938302` should cut parse
(keys were ~90 instr each, now ~38 via one 16-byte vector load; time-of-day is now branch-free SWAR), but this is
**unverified on the Xeon** — a scored submission or a diag run is needed to confirm.

- Phase walls: setup 13 | event parse ~560-600 (links ~360-560 alongside) | dedup+clicks **370-575** |
  sessions 165-230 | write **80-180**. Total ~1.1-1.4 s inside the process.
- Summed thread time: parse 4.3-5.0 s, dedup insert 0.4-0.6, **dedup emit 1.2-1.7** (scales worst),
  clicks 0.8-1.0, sessions 1.3-1.8 (largest bucket 29-81 ms), links 0.35-0.55, write 0.5-1.3.
- Parse components on the scorer, single-thread ns/line (old benchmark; the PB lines in the next diag use the
  production code paths): tokenize 34 | +timestamps 69 | +keys 53 | ts+keys 80 | full 103-117. At 8 threads every
  number is about 2x higher (hyperthreading).
- 1 thread total: parse 2.6 s, dedup+clicks 1.45, sessions 0.83, write 0.21, links 0.23. **Single-thread work is
  about 5.3 s. To reach 0.5 s at ~5x parallelism it must fall to about 2.5 s.**

## 8. What to do next (priority order)

1. **Submit a scored run of `main` (4938302)** to see if SIMD keys + SWAR time + write-overlap helped on the Xeon.
   This is the single most valuable next data point — the parse change is the biggest Xeon-facing win but is
   unverified there. If it did not help, submit the `diag` branch (4972485) to read the new phase breakdown.
2. **Cut intermediate memory traffic (the biggest remaining lever).** Today each event is written as a 32 B ERec,
   read in dedup, written as a 16 B SRec, then copied again during grouping. The cleanest version:
   - Build links first, in parallel (parse link lines on all threads, then union-find ~0.2 s single thread).
     Then parsing can map user→person directly and scatter **by person**. Dedup then happens inside each person
     bucket. **Caveat:** about 1 in 64 duplicated events have copies with different users (so different persons),
     and invalid copies have no person at all. Resolve the rare cross-bucket/invalid cases via a compact
     (eid, ingest, bucket) 16 B triple scattered by eid, or a bloom/bitmap, marking losers per bucket. This
     removes the full eid→person emit shuffle (the phase that "scales worst", 1.2-1.7 s summed). **This is a
     sizable, correctness-risky refactor** (cross-person dups, invalid copies, links-on-critical-path) — do it
     only with time to test against all 6 seeds + dev + 36 fixtures + amd64.
   - Shrink ERec: `repr(packed)` gives 28 B but saves only ~5-10 ms (not bandwidth-bound); a person u32 index
     needs a global dict that does not exist at parse time. Low value.
3. **Parsing** (~50% of CPU) — the high-value parts are **done**: SIMD exact key encoding (SSE4.1/NEON) replaces
   the ~90-instr scalar key loop; SWAR branch-free time-of-day parse for F1/F2. Remaining: F3 (epoch-ms digits,
   ~32% of ts) could use a Lemire-style 8-digit SWAR, but it is only ~1-2% overall with non-trivial overflow
   edge cases at the 18-digit boundary — judged not worth the risk. Tokenize (34 ns/line, now the largest single
   component) is hard to cut without restructuring the tokenizer↔callback interface.
4. **Links thread**: parallelize link parsing. **Likely not worth it** — links (~0.23 s single-thread) finishes
   well before parse (~0.6 s wall) so it is not on the critical path; total CPU work is fixed regardless of
   oversubscription.
5. **Output**: 80-180 ms on the scorer. **Done** — user_daily writes now overlap the session tail via
   `parallel_fin`; only campaign_daily remains post-sessions.
6. **P/Q sizes and dedup density**: `SS_P`, `SS_Q`, `SS_DENSE` env knobs added; the `diag` branch (4972485) sweeps
   P/Q ∈ {1024,4096} × dense to find the Xeon L2-fit optimum (M4's huge L2 can't show this). Tune from a diag run.
7. Evaluated-and-rejected micro-opts: campaign-counter layout flip `[day][campaign]` (per-person access is
   already ~5 KB, fits L1 regardless); caching the `PersonMap` slot across prefetch→lookup (mix is ~3 cycles,
   saves ~3 ms wall); slot index for clicks. Not worth the churn.

## 9. Gotchas learned the hard way

- Always check the track page's downloads against your local copies (generator, rules, dev answer). They changed.
- Never take a global lock per record. Never call `env::var` inside hot loops.
- Mac timings are noisy (iCloud sync, other jobs). Compare A/B back to back, or use the single-thread
  `BENCH_PARSE=1 BENCH_THREADS=N` benchmark. The real truth is the scorer: use the diag crash trick
  (branch `diag`; its Dockerfile builds `--features diag` and ships the generator). Keep its stderr output short,
  because the scorer shows only the tail.
- Test the amd64 image before every push. The scorer builds the Dockerfile at the repo root. Rejected builds do
  not hurt the best score.
- Keep rare-path correctness: long IDs (>10 chars or chars outside `[0-9_a-z]`), uppercase IDs, `\r`, quoted
  fields, lines with no trailing newline, and windows that contain Feb 29 2028.
