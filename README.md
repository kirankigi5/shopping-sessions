# Shopping Sessions: native Rust pipeline

A single multithreaded Rust binary implementing `rules.md`. No engine to start: the
container runs `/app/solution`, which reads `/data` and writes both Parquet tables.

## How it works

1. **Lookups.** `products.csv` and `campaigns.csv` are parsed with the exact CSV dialect.
2. **Links** are parsed and joined with union-find on a background thread while events
   are parsed. Each person is named by the byte-wise smallest ID.
3. **Events** are memory-mapped, split into ~2 MB newline-aligned chunks, and tokenized
   with `memchr3` (`,` `"` `\n`). Each copy is validated (timestamps F1/F2/F3, window,
   type, product, price, late cutoff) at parse time, then scattered by `event_id` hash.
4. **Dedup** per `event_id` bucket keeps the copy with the smallest `ingest_ts` (open
   addressing table), drops it if invalid, maps the user to its person, and scatters by
   person hash. **Clicks** are parsed and scattered by person in the same phase.
5. **Sessions** per person bucket: sort events by `(person, ts)` and clicks by
   `(person, ts, click_id)`; split sessions on gaps over 30 min; a sliding 7-day click
   window gives the crediting click and the set of assisting campaigns.
6. **Output**: one uncompressed Parquet file per thread for `user_daily`, one for
   `campaign_daily`.

IDs of 1 to 12 characters from `[0-9_a-z]` are encoded exactly as base-38 `u64`s, so
numeric order equals byte-wise order. Any other string is interned and compared by bytes.

## Test

```sh
cargo build --release
python3 scripts/run_fixtures.py fixtures -- target/release/solution   # 36 edge cases
DATA_DIR=dev OUTPUT_DIR=out target/release/solution && python3 scripts/compare.py out dev-answer
python3 tools/ref.py DATA OUT_REF   # slow, literal reference implementation for other seeds
```
