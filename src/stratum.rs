//! Pool client. Two implementations:
//!   * `MockPool`   — a synthetic job for benchmarking, no network.
//!   * `StratumPool`— WebSocket JSON-RPC 2.0, CryptoNote-style login/job/submit
//!                    (the usual shape for a cn/gpu coin). Field mapping lives
//!                    in `parse_job`/`submit`; adjust there if a pool differs.

use anyhow::Result;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tungstenite::Message;

pub struct Job {
    pub job_id: String,
    pub input: Vec<u8>,
    pub target: u64,
    pub extra_nonce: u64,
    pub received: Instant,
    nonce_cursor: AtomicU64,
}

impl Job {
    pub fn new(job_id: String, input: Vec<u8>, target: u64, extra_nonce: u64) -> Self {
        Self {
            job_id,
            input,
            target,
            extra_nonce,
            received: Instant::now(),
            nonce_cursor: AtomicU64::new(0),
        }
    }

    /// 128-byte input blob with the trailing 0x01 pad applied (matches the
    /// upstream `setJob`).
    pub fn input_128(&self) -> [u8; 128] {
        let mut b = [0u8; 128];
        let n = self.input.len().min(127);
        b[..n].copy_from_slice(&self.input[..n]);
        b[n] = 0x01;
        b
    }

    /// Reserve a contiguous nonce range of `count` values; returns the base.
    pub fn reserve(&self, count: u64) -> u64 {
        self.nonce_cursor.fetch_add(count, Ordering::Relaxed)
    }
}

pub trait Pool: Send + Sync {
    fn current_job(&self) -> Option<Arc<Job>>;
    fn submit(&self, job: &Arc<Job>, nonce: u64);
    fn is_mock(&self) -> bool {
        false
    }
    fn url(&self) -> &str;
}

// ---------------------------------------------------------------------------
// Mock pool (benchmark)
// ---------------------------------------------------------------------------

pub struct MockPool {
    job: Arc<Job>,
}

impl MockPool {
    pub fn new() -> Self {
        let input: Vec<u8> = (0..76u16).map(|i| (i * 7 + 3) as u8).collect();
        Self {
            // target 0 -> effectively no shares, so benchmarking measures pure
            // pipeline throughput.
            job: Arc::new(Job::new("mock".into(), input, 0, 0)),
        }
    }
}

impl Pool for MockPool {
    fn current_job(&self) -> Option<Arc<Job>> {
        Some(self.job.clone())
    }
    fn submit(&self, _job: &Arc<Job>, _nonce: u64) {}
    fn is_mock(&self) -> bool {
        true
    }
    fn url(&self) -> &str {
        "mock://benchmark"
    }
}

// ---------------------------------------------------------------------------
// Real stratum pool
// ---------------------------------------------------------------------------

struct Shared {
    current: Mutex<Option<Arc<Job>>>,
    session_id: Mutex<Option<String>>,
    req_id: AtomicU64,
}

pub struct StratumPool {
    url: String,
    shared: Arc<Shared>,
    tx: Sender<String>,
}

impl StratumPool {
    pub fn connect(url: &str, user: String, pass: String) -> Result<Self> {
        let shared = Arc::new(Shared {
            current: Mutex::new(None),
            session_id: Mutex::new(None),
            req_id: AtomicU64::new(1),
        });
        let (tx, rx) = mpsc::channel::<String>();

        let io_url = url.to_string();
        let io_shared = shared.clone();
        std::thread::spawn(move || io_loop(io_url, user, pass, io_shared, rx));

        Ok(Self {
            url: url.to_string(),
            shared,
            tx,
        })
    }
}

impl Pool for StratumPool {
    fn current_job(&self) -> Option<Arc<Job>> {
        self.shared.current.lock().unwrap().clone()
    }

    fn submit(&self, job: &Arc<Job>, nonce: u64) {
        let id = self.shared.req_id.fetch_add(1, Ordering::Relaxed);
        let sid = self
            .shared
            .session_id
            .lock()
            .unwrap()
            .clone()
            .unwrap_or_default();
        let msg = serde_json::json!({
            "id": id,
            "jsonrpc": "2.0",
            "method": "submit",
            "params": {
                "id": sid,
                "job_id": job.job_id,
                "nonce": format!("{:016x}", nonce),
            }
        })
        .to_string();
        let _ = self.tx.send(msg);
    }

    fn url(&self) -> &str {
        &self.url
    }
}

fn io_loop(url: String, user: String, pass: String, shared: Arc<Shared>, rx: Receiver<String>) {
    loop {
        match run_connection(&url, &user, &pass, &shared, &rx) {
            Ok(()) => log::warn!("pool connection closed, reconnecting in 5s"),
            Err(e) => log::error!("pool connection error: {e:#}; reconnecting in 5s"),
        }
        *shared.current.lock().unwrap() = None;
        std::thread::sleep(Duration::from_secs(5));
    }
}

fn set_read_timeout(ws: &tungstenite::WebSocket<tungstenite::stream::MaybeTlsStream<std::net::TcpStream>>) {
    use tungstenite::stream::MaybeTlsStream;
    let dur = Some(Duration::from_millis(500));
    match ws.get_ref() {
        MaybeTlsStream::Plain(s) => {
            let _ = s.set_read_timeout(dur);
        }
        MaybeTlsStream::NativeTls(s) => {
            let _ = s.get_ref().set_read_timeout(dur);
        }
        _ => {}
    }
}

fn run_connection(
    url: &str,
    user: &str,
    pass: &str,
    shared: &Arc<Shared>,
    rx: &Receiver<String>,
) -> Result<()> {
    log::info!("connecting to {url}");
    let (mut ws, _resp) = tungstenite::connect(url)?;
    set_read_timeout(&ws);

    // login
    let id = shared.req_id.fetch_add(1, Ordering::Relaxed);
    let login = serde_json::json!({
        "id": id,
        "jsonrpc": "2.0",
        "method": "login",
        "params": { "login": user, "pass": pass, "agent": "fusionhash-vulkan/0.1" }
    })
    .to_string();
    ws.send(Message::Text(login.into()))?;

    loop {
        // drain outgoing submits
        while let Ok(msg) = rx.try_recv() {
            ws.send(Message::Text(msg.into()))?;
        }

        match ws.read() {
            Ok(Message::Text(t)) => handle_message(&t, shared),
            Ok(Message::Binary(b)) => {
                if let Ok(t) = String::from_utf8(b.to_vec()) {
                    handle_message(&t, shared);
                }
            }
            Ok(Message::Ping(p)) => {
                ws.send(Message::Pong(p))?;
            }
            Ok(Message::Close(_)) => return Ok(()),
            Ok(_) => {}
            Err(tungstenite::Error::Io(e))
                if e.kind() == std::io::ErrorKind::WouldBlock
                    || e.kind() == std::io::ErrorKind::TimedOut =>
            {
                // idle tick; loop back to drain outgoing
            }
            Err(e) => return Err(e.into()),
        }
    }
}

fn handle_message(text: &str, shared: &Arc<Shared>) {
    let v: serde_json::Value = match serde_json::from_str(text) {
        Ok(v) => v,
        Err(_) => {
            log::debug!("non-JSON frame: {text}");
            return;
        }
    };

    // session id from a login result
    if let Some(sid) = v.get("result").and_then(|r| r.get("id")).and_then(|x| x.as_str()) {
        *shared.session_id.lock().unwrap() = Some(sid.to_string());
    }

    // job either nested in a login result, or as a `job` notification
    let job_val = v
        .get("result")
        .and_then(|r| r.get("job"))
        .or_else(|| {
            if v.get("method").and_then(|m| m.as_str()) == Some("job") {
                v.get("params")
            } else {
                None
            }
        });

    if let Some(jv) = job_val {
        match parse_job(jv) {
            Some(job) => {
                log::info!(
                    "new job {} target=0x{:016x} extraNonce=0x{:x} ({} input bytes)",
                    job.job_id,
                    job.target,
                    job.extra_nonce,
                    job.input.len()
                );
                *shared.current.lock().unwrap() = Some(Arc::new(job));
            }
            None => log::warn!("could not parse job: {jv}"),
        }
        return;
    }

    // submit acknowledgements / errors
    if let Some(err) = v.get("error") {
        if !err.is_null() {
            log::warn!("share rejected: {err}");
            return;
        }
    }
    if v.get("result").is_some() && v.get("method").is_none() {
        log::debug!("rpc result: {text}");
    }
}

fn parse_num(v: &serde_json::Value) -> Option<u64> {
    if let Some(n) = v.as_u64() {
        return Some(n);
    }
    if let Some(s) = v.as_str() {
        let s = s.trim_start_matches("0x");
        if let Ok(n) = u64::from_str_radix(s, 16) {
            return Some(n);
        }
    }
    None
}

fn parse_job(jv: &serde_json::Value) -> Option<Job> {
    let job_id = jv
        .get("job_id")
        .or_else(|| jv.get("jobId"))
        .and_then(|x| x.as_str())?
        .to_string();

    let blob_hex = jv
        .get("blob")
        .or_else(|| jv.get("input"))
        .or_else(|| jv.get("header"))
        .and_then(|x| x.as_str())?;
    let input = hex::decode(blob_hex.trim_start_matches("0x")).ok()?;

    let target = jv
        .get("target")
        .and_then(parse_num)
        .unwrap_or(u64::MAX);

    let extra_nonce = jv
        .get("extraNonce")
        .or_else(|| jv.get("extra_nonce"))
        .and_then(parse_num)
        .unwrap_or(0);

    Some(Job::new(job_id, input, target, extra_nonce))
}
