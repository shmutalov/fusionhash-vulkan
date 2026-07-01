//! Pool client. Two implementations:
//!   * `MockPool`   — a synthetic job for benchmarking, no network.
//!   * `StratumPool`— the FusionLayer protocol: go-ethereum JSON-RPC 2.0 over
//!                    WebSocket. Subscribe with `eth_subscribe("newWork", user,
//!                    pass, agent)`; jobs arrive as `eth_subscription`
//!                    notifications carrying a 6-string array
//!                    `[jobId, powHash, seedHash, target, blockNumber,
//!                    extraNonce]`; shares are sent with `eth_submitWork`.

use anyhow::Result;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tungstenite::Message;

const AGENT: &str = "warpminer";

pub struct Job {
    pub job_id: String,
    pub pow_hash: Vec<u8>, // 32 bytes
    pub input: Vec<u8>,    // powHash(32) || seedHash(32) || 00*8  (72 bytes)
    pub target: u64,
    pub extra_nonce: u64,
    pub received: Instant,
    nonce_cursor: AtomicU64,
}

impl Job {
    /// 128-byte input blob with the trailing 0x01 pad applied (matches the
    /// upstream `setJob`: the pad lands at byte 72).
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
        let mut input = vec![0u8; 72];
        for (i, b) in input.iter_mut().take(64).enumerate() {
            *b = (i * 7 + 3) as u8;
        }
        Self {
            // target 0 -> effectively no shares, so benchmarking measures pure
            // pipeline throughput.
            job: Arc::new(Job {
                job_id: "mock".into(),
                pow_hash: input[..32].to_vec(),
                input,
                target: 0,
                extra_nonce: 0,
                received: Instant::now(),
                nonce_cursor: AtomicU64::new(0),
            }),
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
// Real pool (FusionLayer / go-ethereum RPC)
// ---------------------------------------------------------------------------

struct Shared {
    current: Mutex<Option<Arc<Job>>>,
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
        let nonce_hex = format!("0x{:016x}", nonce);
        let pow_hex = format!("0x{}", hex::encode(&job.pow_hash));
        let msg = serde_json::json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": "eth_submitWork",
            "params": [job.job_id, nonce_hex, pow_hex],
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

fn set_read_timeout(
    ws: &tungstenite::WebSocket<tungstenite::stream::MaybeTlsStream<std::net::TcpStream>>,
) {
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

    // eth_subscribe("newWork", user, pass, agent)
    let id = shared.req_id.fetch_add(1, Ordering::Relaxed);
    let sub = serde_json::json!({
        "jsonrpc": "2.0",
        "id": id,
        "method": "eth_subscribe",
        "params": ["newWork", user, pass, AGENT],
    })
    .to_string();
    ws.send(Message::Text(sub.into()))?;
    log::info!("subscribed to newWork as {user}");

    loop {
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
                // idle tick; loop back to drain outgoing submits
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

    // Job notification: {"method":"eth_subscription","params":{"result":[..6..]}}
    if v.get("method").and_then(|m| m.as_str()) == Some("eth_subscription") {
        if let Some(arr) = v
            .get("params")
            .and_then(|p| p.get("result"))
            .and_then(|r| r.as_array())
        {
            match parse_job(arr) {
                Some(job) => {
                    let diff = if job.target == 0 { 0 } else { u64::MAX / job.target + 1 };
                    log::info!(
                        "new job {} block={} diff={} target=0x{:016x} extraNonce=0x{:x}",
                        job.job_id,
                        job_block(arr),
                        diff,
                        job.target,
                        job.extra_nonce,
                    );
                    *shared.current.lock().unwrap() = Some(Arc::new(job));
                }
                None => log::warn!("could not parse job: {text}"),
            }
        }
        return;
    }

    // eth_submitWork ack / eth_subscribe confirmation / errors.
    if let Some(err) = v.get("error") {
        if !err.is_null() {
            log::warn!("rpc error: {err}");
            return;
        }
    }
    match v.get("result") {
        Some(serde_json::Value::Bool(true)) => log::info!("share accepted"),
        Some(serde_json::Value::Bool(false)) => log::warn!("share rejected"),
        Some(serde_json::Value::String(s)) => log::debug!("subscription id {s}"),
        _ => {}
    }
}

fn job_block(arr: &[serde_json::Value]) -> u64 {
    arr.get(4)
        .and_then(|x| x.as_str())
        .and_then(|s| u64::from_str_radix(s.trim_start_matches("0x"), 16).ok())
        .unwrap_or(0)
}

fn parse_job(arr: &[serde_json::Value]) -> Option<Job> {
    if arr.len() < 6 {
        return None;
    }
    let s = |i: usize| arr[i].as_str();

    let job_id = s(0)?.to_string();
    let pow_hash = hex::decode(s(1)?.trim_start_matches("0x")).ok()?;
    let seed_hash = hex::decode(s(2)?.trim_start_matches("0x")).ok()?;
    if pow_hash.len() != 32 || seed_hash.len() != 32 {
        return None;
    }

    // target = top 64 bits of the 256-bit target hex.
    let thex = s(3)?.trim_start_matches("0x");
    let top16: String = thex.chars().take(16).collect();
    let target = u64::from_str_radix(&top16, 16).ok()?;

    // extraNonce = hex, left-justified into 64 bits.
    let enh = s(5)?;
    let extra_nonce = if enh.is_empty() {
        0
    } else {
        u64::from_str_radix(&format!("{:0<16}", enh), 16).ok()?
    };

    // input = powHash || seedHash || 8 zero bytes (72 bytes)
    let mut input = Vec::with_capacity(72);
    input.extend_from_slice(&pow_hash);
    input.extend_from_slice(&seed_hash);
    input.extend_from_slice(&[0u8; 8]);

    Some(Job {
        job_id,
        pow_hash,
        input,
        target,
        extra_nonce,
        received: Instant::now(),
        nonce_cursor: AtomicU64::new(0),
    })
}
