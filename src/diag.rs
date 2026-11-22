// Diagnostics for the scorer machine. Prints to stderr only.
use std::io::Read;
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn cat(path: &str) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|e| format!("<{}>", e))
}

fn files(data: &Path) -> Vec<PathBuf> {
    let mut v = Vec::new();
    for d in ["events", "clicks", "links"] {
        if let Ok(rd) = std::fs::read_dir(data.join(d)) {
            for e in rd.flatten() {
                v.push(e.path());
            }
        }
    }
    v.sort();
    v
}

fn evict(paths: &[PathBuf]) {
    for p in paths {
        if let Ok(f) = std::fs::File::open(p) {
            #[cfg(target_os = "linux")]
            unsafe {
                libc::posix_fadvise(f.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED);
            }
            let _ = f.as_raw_fd();
        }
    }
}

fn read_all_parallel(paths: &[PathBuf], threads: usize, bufsize: usize) -> (f64, u64) {
    let next = std::sync::atomic::AtomicUsize::new(0);
    let t = Instant::now();
    let total: u64 = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..threads)
            .map(|_| {
                sc.spawn(|| {
                    let mut buf = vec![0u8; bufsize];
                    let mut sum = 0u64;
                    loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if i >= paths.len() {
                            break;
                        }
                        let mut f = std::fs::File::open(&paths[i]).unwrap();
                        loop {
                            let n = f.read(&mut buf).unwrap();
                            if n == 0 {
                                break;
                            }
                            sum += n as u64;
                        }
                    }
                    sum
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).sum()
    });
    (t.elapsed().as_secs_f64(), total)
}

fn mmap_all_parallel(paths: &[PathBuf], threads: usize) -> (f64, u64) {
    let next = std::sync::atomic::AtomicUsize::new(0);
    let t = Instant::now();
    let total: u64 = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..threads)
            .map(|_| {
                sc.spawn(|| {
                    let mut sum = 0u64;
                    loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if i >= paths.len() {
                            break;
                        }
                        let f = std::fs::File::open(&paths[i]).unwrap();
                        if f.metadata().unwrap().len() == 0 {
                            continue;
                        }
                        let m = unsafe { memmap2::Mmap::map(&f).unwrap() };
                        let mut x = 0u64;
                        let mut o = 0;
                        while o < m.len() {
                            x = x.wrapping_add(unsafe { std::ptr::read_volatile(m.as_ptr().add(o)) } as u64);
                            o += 4096;
                        }
                        std::hint::black_box(x);
                        sum += m.len() as u64;
                    }
                    sum
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).sum()
    });
    (t.elapsed().as_secs_f64(), total)
}

/// Tokenize-only and scatter microbenchmarks on in-memory copies of the event files.
pub fn bench_parse(data: &Path) {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(data.join("events")).unwrap().flatten().map(|e| e.path()).collect();
    paths.sort();
    let bufs: Vec<Vec<u8>> = paths.iter().map(|p| std::fs::read(p).unwrap()).collect();
    let lines: usize = bufs.iter().map(|b| memchr::memchr_iter(b'\n', b).count()).sum();
    eprintln!("bench: {} files, {} lines, avx2 path={}", bufs.len(), lines, cfg!(target_feature = "avx2"));
    for th in [1usize, 8] {
        for variant in 0..4 {
            let t = Instant::now();
            let next = std::sync::atomic::AtomicUsize::new(0);
            std::thread::scope(|sc| {
                for _ in 0..th {
                    sc.spawn(|| {
                        let fan = [1usize, 1, 64, 1024][variant];
                        let mut out: Vec<Vec<[u64; 4]>> = (0..fan).map(|_| Vec::with_capacity(lines / th / fan + 64)).collect();
                        let mut sink = 0u64;
                        for _rep in 0..3 {
                            loop {
                                let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                if i >= bufs.len() * 3 {
                                    break;
                                }
                                let b = &bufs[i % bufs.len()];
                                crate::scan_lines::<7>(b, |f| {
                                    if variant == 0 {
                                        sink += f[0].len() as u64;
                                    } else {
                                        let k = crate::eq_key(f[0]);
                                        let u = crate::eq_key(f[1]);
                                        let ts = crate::parse_ts(f[2]).unwrap_or(0) as u64;
                                        let ing = crate::parse_ts(f[3]).unwrap_or(0) as u64;
                                        if variant == 1 {
                                            sink = sink.wrapping_add(k ^ u ^ ts ^ ing);
                                        } else {
                                            out[crate::part_of(k, fan - 1)].push([k, u, ts, ing]);
                                        }
                                    }
                                });
                            }
                        }
                        std::hint::black_box(sink);
                        std::hint::black_box(&out);
                    });
                }
            });
            let el = t.elapsed().as_secs_f64();
            eprintln!(
                "bench threads={} variant={} ({}): {:.1} ns/line/thread",
                th,
                variant,
                ["tokenize", "parse", "parse+64buckets", "parse+1024buckets"][variant],
                el * 1e9 * th as f64 / (lines * 3) as f64
            );
        }
    }
}

pub fn run(data: &Path) {
    let t0 = Instant::now();
    eprintln!("=== DIAG ===");
    eprintln!("uname: {}", cat("/proc/sys/kernel/osrelease").trim());
    let cpu = cat("/proc/cpuinfo");
    let model = cpu.lines().find(|l| l.starts_with("model name")).unwrap_or("");
    let ncpu_info = cpu.lines().filter(|l| l.starts_with("processor")).count();
    let flags = cpu.lines().find(|l| l.starts_with("flags")).unwrap_or("");
    eprintln!("cpu: {} x{} avx2={} avx512bw={}", model, ncpu_info, flags.contains(" avx2"), flags.contains(" avx512bw"));
    eprintln!("available_parallelism: {:?}", std::thread::available_parallelism());
    eprintln!("cgroup cpu.max: {}", cat("/sys/fs/cgroup/cpu.max").trim());
    eprintln!("cgroup cpuset: {}", cat("/sys/fs/cgroup/cpuset.cpus.effective").trim());
    eprintln!("cgroup memory.max: {}", cat("/sys/fs/cgroup/memory.max").trim());
    eprintln!("THP: {}", cat("/sys/kernel/mm/transparent_hugepage/enabled").trim());
    let status = cat("/proc/self/status");
    for l in status.lines().filter(|l| l.starts_with("Cpus_allowed_list")) {
        eprintln!("{}", l);
    }
    let mi = cat("/proc/meminfo");
    for l in mi.lines().take(5) {
        eprintln!("meminfo {}", l);
    }
    for l in cat("/proc/mounts").lines() {
        if l.contains(" /data") || l.contains(" /output") || l.contains(" /scratch") || l.contains(" / ") {
            eprintln!("mount: {}", l);
        }
    }
    for k in ["NCPU", "MEM_GB", "DATA_DIR", "OUTPUT_DIR"] {
        eprintln!("env {}={:?}", k, std::env::var(k).ok());
    }

    let paths = files(data);
    let threads = 8;
    // cold read (as delivered), then warm, then evict + mmap cold, then warm mmap
    let (a, n) = read_all_parallel(&paths, threads, 1 << 20);
    eprintln!("read() as-delivered: {:.3}s {} MB -> {:.0} MB/s", a, n >> 20, n as f64 / a / 1e6);
    let (a, n) = read_all_parallel(&paths, threads, 1 << 20);
    eprintln!("read() warm: {:.3}s -> {:.0} MB/s", a, n as f64 / a / 1e6);
    let (a, n) = mmap_all_parallel(&paths, threads);
    eprintln!("mmap warm: {:.3}s -> {:.0} MB/s", a, n as f64 / a / 1e6);
    evict(&paths);
    let (a, n) = read_all_parallel(&paths, threads, 1 << 20);
    eprintln!("read() after evict: {:.3}s -> {:.0} MB/s", a, n as f64 / a / 1e6);
    evict(&paths);
    let (a, n) = mmap_all_parallel(&paths, threads);
    eprintln!("mmap after evict: {:.3}s -> {:.0} MB/s", a, n as f64 / a / 1e6);
    let (a, n) = read_all_parallel(&paths, 1, 1 << 20);
    eprintln!("read() warm 1 thread: {:.3}s -> {:.0} MB/s", a, n as f64 / a / 1e6);

    // page fault / zeroing cost
    for th in [1usize, 8] {
        let t = Instant::now();
        std::thread::scope(|sc| {
            for _ in 0..th {
                sc.spawn(|| {
                    let mut v: Vec<u8> = Vec::with_capacity(256 << 20);
                    unsafe { v.set_len(256 << 20) };
                    let mut o = 0;
                    while o < v.len() {
                        v[o] = 1;
                        o += 4096;
                    }
                    std::hint::black_box(&v);
                });
            }
        });
        let el = t.elapsed().as_secs_f64();
        eprintln!("touch 256MB fresh x{}: {:.3}s -> {:.2} GB/s", th, el, (th * 256) as f64 / 1024.0 / el);
    }
    // memory bandwidth (memcpy) and compute
    {
        let src = vec![1u8; 256 << 20];
        let mut dst = vec![0u8; 256 << 20];
        let t = Instant::now();
        for _ in 0..4 {
            dst.copy_from_slice(&src);
            std::hint::black_box(&dst);
        }
        eprintln!("memcpy 1 thread: {:.2} GB/s", 1.0 / t.elapsed().as_secs_f64());
    }
    for th in [1usize, 4, 8] {
        let t = Instant::now();
        std::thread::scope(|sc| {
            for k in 0..th {
                sc.spawn(move || {
                    let mut x = k as u64 + 1;
                    for _ in 0..200_000_000u64 {
                        x ^= x << 13;
                        x ^= x >> 7;
                        x ^= x << 17;
                    }
                    std::hint::black_box(x);
                });
            }
        });
        eprintln!("xorshift 200M x{}: {:.3}s", th, t.elapsed().as_secs_f64());
    }
    eprintln!("=== DIAG took {:.3}s ===", t0.elapsed().as_secs_f64());
}
