//! A minimal in-process HTTP/1.1 server for driving the Vertex wire paths end
//! to end. Point a client at it with `VertexConfig::with_base_url`.

#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

pub const TOKEN_PATH: &str = "/computeMetadata/";
pub const TOKEN_BODY: &str =
    r#"{"access_token":"stub-token","expires_in":3600,"token_type":"Bearer"}"#;

#[derive(Debug, Clone)]
pub struct CapturedRequest {
    pub method: String,
    pub path: String,
    pub headers: HashMap<String, String>,
    pub body: String,
}

impl CapturedRequest {
    pub fn json(&self) -> serde_json::Value {
        serde_json::from_str(&self.body).expect("stub server captured a non-JSON body")
    }

    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .get(&name.to_ascii_lowercase())
            .map(String::as_str)
    }

    pub fn is_token_fetch(&self) -> bool {
        self.path.contains(TOKEN_PATH)
    }
}

pub struct Reply {
    status: u16,
    body: String,
    delay: Duration,
    headers: Vec<(String, String)>,
    content_type: String,
}

impl Reply {
    pub fn ok(body: impl Into<String>) -> Self {
        Self {
            status: 200,
            body: body.into(),
            delay: Duration::ZERO,
            headers: Vec::new(),
            content_type: "application/json".to_string(),
        }
    }

    pub fn error(status: u16, body: impl Into<String>) -> Self {
        Self {
            status,
            body: body.into(),
            delay: Duration::ZERO,
            headers: Vec::new(),
            content_type: "application/json".to_string(),
        }
    }

    /// Hold the response back, so completion order can differ from request order.
    pub fn after(mut self, delay: Duration) -> Self {
        self.delay = delay;
        self
    }

    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    pub fn with_content_type(mut self, content_type: impl Into<String>) -> Self {
        self.content_type = content_type.into();
        self
    }
}

pub struct StubServer {
    pub base_url: String,
    requests: Arc<Mutex<Vec<CapturedRequest>>>,
}

impl StubServer {
    pub async fn start<F>(handler: F) -> Self
    where
        F: Fn(&CapturedRequest) -> Reply + Send + Sync + 'static,
    {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind stub server");
        let addr = listener.local_addr().expect("stub server address");
        let requests = Arc::new(Mutex::new(Vec::new()));

        let handler = Arc::new(handler);
        let log = requests.clone();
        tokio::spawn(async move {
            while let Ok((stream, _)) = listener.accept().await {
                // One task per connection, so concurrent callers overlap.
                let handler = handler.clone();
                let log = log.clone();
                tokio::spawn(async move { serve(stream, handler, log).await });
            }
        });

        Self {
            base_url: format!("http://{addr}"),
            requests,
        }
    }

    pub fn requests(&self) -> Vec<CapturedRequest> {
        self.requests.lock().expect("stub request log").clone()
    }

    pub fn request_count(&self) -> usize {
        self.requests.lock().expect("stub request log").len()
    }

    pub fn api_requests(&self) -> Vec<CapturedRequest> {
        self.requests()
            .into_iter()
            .filter(|r| !r.is_token_fetch())
            .collect()
    }
}

async fn serve<F>(mut stream: TcpStream, handler: Arc<F>, log: Arc<Mutex<Vec<CapturedRequest>>>)
where
    F: Fn(&CapturedRequest) -> Reply + Send + Sync + 'static,
{
    let Some(request) = read_request(&mut stream).await else {
        return;
    };
    // Logged on arrival, before any delay, so counts reflect what was sent.
    log.lock().expect("stub request log").push(request.clone());

    let reply = handler(&request);
    if !reply.delay.is_zero() {
        tokio::time::sleep(reply.delay).await;
    }

    let extra: String = reply
        .headers
        .iter()
        .map(|(name, value)| format!("{name}: {value}\r\n"))
        .collect();
    let response = format!(
        "HTTP/1.1 {} STUB\r\nContent-Type: {}\r\n{extra}Content-Length: {}\r\nConnection: close\r\n\r\n{}",
        reply.status,
        reply.content_type,
        reply.body.len(),
        reply.body
    );
    let _ = stream.write_all(response.as_bytes()).await;
    let _ = stream.shutdown().await;
}

async fn read_request(stream: &mut TcpStream) -> Option<CapturedRequest> {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 4096];

    let head_end = loop {
        if let Some(position) = find(&buf, b"\r\n\r\n") {
            break position + 4;
        }
        match stream.read(&mut chunk).await {
            Ok(0) | Err(_) => return None,
            Ok(n) => buf.extend_from_slice(&chunk[..n]),
        }
    };

    let head = String::from_utf8_lossy(&buf[..head_end]).into_owned();
    let mut lines = head.lines();
    let start_line = lines.next().unwrap_or_default();
    let mut parts = start_line.split_whitespace();
    let method = parts.next().unwrap_or_default().to_string();
    let path = parts.next().unwrap_or_default().to_string();

    let mut headers = HashMap::new();
    for line in lines {
        if let Some((name, value)) = line.split_once(':') {
            headers.insert(name.trim().to_ascii_lowercase(), value.trim().to_string());
        }
    }

    let content_length = headers
        .get("content-length")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(0);
    while buf.len() < head_end + content_length {
        match stream.read(&mut chunk).await {
            Ok(0) | Err(_) => break,
            Ok(n) => buf.extend_from_slice(&chunk[..n]),
        }
    }

    Some(CapturedRequest {
        method,
        path,
        headers,
        body: String::from_utf8_lossy(&buf[head_end..]).into_owned(),
    })
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}
