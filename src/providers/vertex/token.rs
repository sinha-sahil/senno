use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use jsonwebtoken::{Algorithm, EncodingKey, Header};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

use crate::error::Error;

const DEFAULT_TOKEN_URI: &str = "https://oauth2.googleapis.com/token";
const JWT_LIFETIME_SECS: u64 = 3600;
const EXPIRY_SAFETY_MARGIN: Duration = Duration::from_secs(60);
const MAX_TOKEN_LIFETIME: Duration = Duration::from_secs(JWT_LIFETIME_SECS);
const TOKEN_FETCH_TIMEOUT: Duration = Duration::from_secs(15);
const TOKEN_FETCH_MAX_ATTEMPTS: u32 = 2;
const TOKEN_FETCH_RETRY_DELAY: Duration = Duration::from_millis(200);

pub const DEFAULT_METADATA_BASE_URL: &str = "http://metadata.google.internal";
const METADATA_CACHE_KEY: &str = "metadata";

#[derive(Clone, Deserialize)]
pub struct ServiceAccountKey {
    client_email: String,
    private_key: String,
    #[serde(default)]
    private_key_id: Option<String>,
    #[serde(default)]
    token_uri: Option<String>,
}

impl std::fmt::Debug for ServiceAccountKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ServiceAccountKey")
            .field("client_email", &self.client_email)
            .field("private_key_id", &self.private_key_id)
            .field("token_uri", &self.token_uri)
            .field("private_key", &"<redacted>")
            .finish()
    }
}

impl ServiceAccountKey {
    pub fn new(client_email: impl Into<String>, private_key: impl Into<String>) -> Self {
        Self {
            client_email: client_email.into(),
            private_key: normalize_private_key(&private_key.into()),
            private_key_id: None,
            token_uri: None,
        }
    }

    pub async fn from_path(path: &str) -> Result<Self, Error> {
        let json = tokio::fs::read_to_string(path).await.map_err(|e| {
            Error::config(format!(
                "Failed to read service account key file at {path}: {e}"
            ))
        })?;
        Self::from_json(&json)
    }

    fn from_json(json: &str) -> Result<Self, Error> {
        let mut key: Self = serde_json::from_str(json)
            .map_err(|e| Error::config(format!("Failed to parse service account JSON: {e}")))?;
        key.private_key = normalize_private_key(&key.private_key);
        Ok(key)
    }

    fn token_uri(&self) -> &str {
        self.token_uri.as_deref().unwrap_or(DEFAULT_TOKEN_URI)
    }
}

// `.env` stores newlines as `\\n`; PEM parsers need real newlines.
fn normalize_private_key(raw: &str) -> String {
    raw.replace("\\n", "\n")
}

#[derive(Serialize)]
struct JwtClaims<'a> {
    iss: &'a str,
    scope: String,
    aud: &'a str,
    exp: u64,
    iat: u64,
}

#[derive(Deserialize)]
struct TokenResponse {
    access_token: String,
    expires_in: u64,
}

#[derive(Deserialize)]
struct TokenErrorResponse {
    error: String,
    #[serde(default)]
    error_description: Option<String>,
}

#[derive(Clone)]
struct CachedToken {
    value: String,
    expires_at: Instant,
}

type CacheKey = (String, Vec<String>);

enum AuthSource {
    ServiceAccount(ServiceAccountKey),
    Metadata { base_url: String },
}

pub struct TokenSource {
    http: reqwest::Client,
    auth: AuthSource,
    cache: Mutex<HashMap<CacheKey, CachedToken>>,
    locks: Mutex<HashMap<CacheKey, Arc<Mutex<()>>>>,
}

impl TokenSource {
    pub fn new(http: reqwest::Client, key: ServiceAccountKey) -> Self {
        Self {
            http,
            auth: AuthSource::ServiceAccount(key),
            cache: Mutex::new(HashMap::new()),
            locks: Mutex::new(HashMap::new()),
        }
    }

    /// GCE/Cloud Run metadata-server auth; `base_url` is
    /// [`DEFAULT_METADATA_BASE_URL`] outside of tests.
    pub fn metadata(http: reqwest::Client, base_url: impl Into<String>) -> Self {
        let base_url = base_url.into().trim_end_matches('/').to_string();
        Self {
            http,
            auth: AuthSource::Metadata { base_url },
            cache: Mutex::new(HashMap::new()),
            locks: Mutex::new(HashMap::new()),
        }
    }

    pub async fn access_token(&self, scopes: &[&str]) -> Result<String, Error> {
        // Sort so the same scope set in different orders shares a cache slot.
        let mut sorted_scopes: Vec<String> = scopes.iter().map(|s| s.to_string()).collect();
        sorted_scopes.sort();
        let cache_key_id = match &self.auth {
            AuthSource::ServiceAccount(k) => k.client_email.clone(),
            AuthSource::Metadata { .. } => METADATA_CACHE_KEY.to_string(),
        };
        let cache_key: CacheKey = (cache_key_id, sorted_scopes);

        if let Some(token) = self.lookup_cached(&cache_key).await {
            return Ok(token);
        }

        let lock = {
            let mut locks = self.locks.lock().await;
            locks
                .entry(cache_key.clone())
                .or_insert_with(|| Arc::new(Mutex::new(())))
                .clone()
        };
        let _guard = lock.lock().await;

        if let Some(token) = self.lookup_cached(&cache_key).await {
            return Ok(token);
        }

        let fetched = self.fetch_token(scopes).await?;
        let mut cache = self.cache.lock().await;
        cache.insert(cache_key, fetched.clone());
        Ok(fetched.value)
    }

    async fn lookup_cached(&self, key: &CacheKey) -> Option<String> {
        let cache = self.cache.lock().await;
        cache
            .get(key)
            .filter(|t| t.expires_at > Instant::now())
            .map(|t| t.value.clone())
    }

    async fn fetch_token(&self, scopes: &[&str]) -> Result<CachedToken, Error> {
        match &self.auth {
            AuthSource::ServiceAccount(key) => self.fetch_sa_token(key, scopes).await,
            AuthSource::Metadata { base_url } => self.fetch_metadata_token(base_url).await,
        }
    }

    async fn fetch_sa_token(
        &self,
        key: &ServiceAccountKey,
        scopes: &[&str],
    ) -> Result<CachedToken, Error> {
        let assertion = sign_assertion(key, scopes)?;

        let mut last_transient: Option<String> = None;
        for attempt in 1..=TOKEN_FETCH_MAX_ATTEMPTS {
            match self.try_fetch_sa_token(key, &assertion).await {
                Ok(token) => return Ok(token),
                Err(TokenFetchError::Permanent(e)) => return Err(e),
                Err(TokenFetchError::Transient(detail)) => {
                    if attempt < TOKEN_FETCH_MAX_ATTEMPTS {
                        tracing::warn!(
                            attempt,
                            error = %detail,
                            "Transient GCP token fetch error, retrying"
                        );
                        tokio::time::sleep(TOKEN_FETCH_RETRY_DELAY).await;
                    }
                    last_transient = Some(detail);
                }
            }
        }

        Err(Error::provider(
            "gcp-oauth2",
            format!(
                "failed to fetch access token after {TOKEN_FETCH_MAX_ATTEMPTS} attempts: {}",
                last_transient.unwrap_or_else(|| "unknown error".into())
            ),
        ))
    }

    async fn try_fetch_sa_token(
        &self,
        key: &ServiceAccountKey,
        assertion: &str,
    ) -> Result<CachedToken, TokenFetchError> {
        let send_fut = self
            .http
            .post(key.token_uri())
            .form(&[
                ("grant_type", "urn:ietf:params:oauth:grant-type:jwt-bearer"),
                ("assertion", assertion),
            ])
            .send();

        let resp = match tokio::time::timeout(TOKEN_FETCH_TIMEOUT, send_fut).await {
            Ok(Ok(r)) => r,
            Ok(Err(e)) => return Err(TokenFetchError::Transient(format!("send error: {e}"))),
            Err(_) => {
                return Err(TokenFetchError::Transient(format!(
                    "timed out after {}s",
                    TOKEN_FETCH_TIMEOUT.as_secs()
                )));
            }
        };

        let status = resp.status();
        let body = resp
            .bytes()
            .await
            .map_err(|e| TokenFetchError::Transient(format!("read body error: {e}")))?;

        if status.is_server_error() {
            return Err(TokenFetchError::Transient(format!(
                "HTTP {status}: {}",
                parse_token_error(&body)
            )));
        }

        if !status.is_success() {
            return Err(TokenFetchError::Permanent(Error::provider_permanent(
                "gcp-oauth2",
                format!(
                    "token endpoint returned {status}: {}",
                    parse_token_error(&body)
                ),
            )));
        }

        let token: TokenResponse = serde_json::from_slice(&body).map_err(|e| {
            TokenFetchError::Permanent(Error::provider_permanent(
                "gcp-oauth2",
                format!("malformed token response: {e}"),
            ))
        })?;

        Ok(cached_with_safety_margin(token))
    }

    async fn fetch_metadata_token(&self, base_url: &str) -> Result<CachedToken, Error> {
        let url = format!("{base_url}/computeMetadata/v1/instance/service-accounts/default/token");

        let mut last_transient: Option<String> = None;
        for attempt in 1..=TOKEN_FETCH_MAX_ATTEMPTS {
            match self.try_fetch_metadata_token(&url).await {
                Ok(token) => return Ok(token),
                Err(TokenFetchError::Permanent(e)) => return Err(e),
                Err(TokenFetchError::Transient(detail)) => {
                    if attempt < TOKEN_FETCH_MAX_ATTEMPTS {
                        tracing::warn!(
                            attempt,
                            error = %detail,
                            "Transient GCP metadata-server token fetch error, retrying"
                        );
                        tokio::time::sleep(TOKEN_FETCH_RETRY_DELAY).await;
                    }
                    last_transient = Some(detail);
                }
            }
        }

        Err(Error::provider(
            "gcp-metadata",
            format!(
                "failed to fetch metadata-server token after {TOKEN_FETCH_MAX_ATTEMPTS} attempts: {}",
                last_transient.unwrap_or_else(|| "unknown error".into())
            ),
        ))
    }

    async fn try_fetch_metadata_token(&self, url: &str) -> Result<CachedToken, TokenFetchError> {
        let send_fut = self
            .http
            .get(url)
            .header("Metadata-Flavor", "Google")
            .send();

        let resp = match tokio::time::timeout(TOKEN_FETCH_TIMEOUT, send_fut).await {
            Ok(Ok(r)) => r,
            Ok(Err(e)) => return Err(TokenFetchError::Transient(format!("send error: {e}"))),
            Err(_) => {
                return Err(TokenFetchError::Transient(format!(
                    "timed out after {}s",
                    TOKEN_FETCH_TIMEOUT.as_secs()
                )));
            }
        };

        let status = resp.status();
        let body = resp
            .bytes()
            .await
            .map_err(|e| TokenFetchError::Transient(format!("read body error: {e}")))?;

        if status.is_server_error() {
            return Err(TokenFetchError::Transient(format!(
                "metadata server returned HTTP {status}: {}",
                String::from_utf8_lossy(&body)
            )));
        }

        if !status.is_success() {
            return Err(TokenFetchError::Permanent(Error::provider_permanent(
                "gcp-metadata",
                format!(
                    "metadata server returned HTTP {status}: {}",
                    String::from_utf8_lossy(&body)
                ),
            )));
        }

        let token: TokenResponse = serde_json::from_slice(&body).map_err(|e| {
            TokenFetchError::Permanent(Error::provider_permanent(
                "gcp-metadata",
                format!("Malformed metadata-server response: {e}"),
            ))
        })?;

        Ok(cached_with_safety_margin(token))
    }
}

fn sign_assertion(key: &ServiceAccountKey, scopes: &[&str]) -> Result<String, Error> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| Error::internal(format!("Clock error: {e}")))?
        .as_secs();

    let claims = JwtClaims {
        iss: &key.client_email,
        scope: scopes.join(" "),
        aud: key.token_uri(),
        iat: now,
        exp: now + JWT_LIFETIME_SECS,
    };

    let mut header = Header::new(Algorithm::RS256);
    header.kid = key.private_key_id.clone();

    let encoding_key = EncodingKey::from_rsa_pem(key.private_key.as_bytes())
        .map_err(|e| Error::config(format!("Invalid GCP service account private key: {e}")))?;

    jsonwebtoken::encode(&header, &claims, &encoding_key)
        .map_err(|e| Error::internal(format!("Failed to sign JWT for GCP token: {e}")))
}

fn cached_with_safety_margin(token: TokenResponse) -> CachedToken {
    let lifetime = Duration::from_secs(token.expires_in)
        .min(MAX_TOKEN_LIFETIME)
        .checked_sub(EXPIRY_SAFETY_MARGIN)
        .unwrap_or(Duration::ZERO);
    CachedToken {
        value: token.access_token,
        expires_at: Instant::now()
            .checked_add(lifetime)
            .unwrap_or_else(Instant::now),
    }
}

enum TokenFetchError {
    Transient(String),
    Permanent(Error),
}

fn parse_token_error(body: &[u8]) -> String {
    serde_json::from_slice::<TokenErrorResponse>(body)
        .map(|e| match e.error_description {
            Some(desc) => format!("{}: {desc}", e.error),
            None => e.error,
        })
        .unwrap_or_else(|_| String::from_utf8_lossy(body).into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;
    async fn spawn_stub(responses: Vec<&'static str>) -> (String, std::sync::Arc<AtomicUsize>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let count = std::sync::Arc::new(AtomicUsize::new(0));
        let count_clone = count.clone();

        tokio::spawn(async move {
            let mut idx = 0;
            while let Ok((mut stream, _)) = listener.accept().await {
                let resp = responses
                    .get(idx)
                    .copied()
                    .unwrap_or("HTTP/1.1 500 Internal Server Error\r\nContent-Length: 0\r\n\r\n");
                idx += 1;
                count_clone.fetch_add(1, Ordering::SeqCst);

                let mut buf = [0u8; 1024];
                let mut total = Vec::new();
                loop {
                    let n = match stream.read(&mut buf).await {
                        Ok(0) | Err(_) => break,
                        Ok(n) => n,
                    };
                    total.extend_from_slice(&buf[..n]);
                    if total.windows(4).any(|w| w == b"\r\n\r\n") {
                        break;
                    }
                }
                let _ = stream.write_all(resp.as_bytes()).await;
                let _ = stream.shutdown().await;
            }
        });

        (format!("http://{addr}"), count)
    }

    fn ok_body(body: &str) -> String {
        format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        )
    }

    #[test]
    fn an_absurd_expires_in_cannot_overflow_the_deadline() {
        let cached = cached_with_safety_margin(TokenResponse {
            access_token: "t".into(),
            expires_in: u64::MAX,
        });

        assert!(cached.expires_at <= Instant::now() + MAX_TOKEN_LIFETIME);
    }

    #[tokio::test]
    async fn metadata_token_happy_path_and_cache_hit() {
        let token_body = r#"{"access_token":"ya29.fake","expires_in":3600,"token_type":"Bearer"}"#;
        let resp = ok_body(token_body);
        let leaked: &'static str = Box::leak(resp.into_boxed_str());
        // Only one response staged — second call must hit the cache.
        let (base_url, count) = spawn_stub(vec![leaked]).await;

        let http = reqwest::Client::new();
        let ts = TokenSource::metadata(http, base_url);

        let token = ts.access_token(&["scope-a"]).await.unwrap();
        assert_eq!(token, "ya29.fake");

        let token2 = ts.access_token(&["scope-a"]).await.unwrap();
        assert_eq!(token2, "ya29.fake");

        assert_eq!(
            count.load(Ordering::SeqCst),
            1,
            "second access_token should hit the cache, not the stub"
        );
    }

    #[tokio::test]
    async fn metadata_token_transient_5xx_then_success() {
        let ok =
            ok_body(r#"{"access_token":"ya29.retry","expires_in":3600,"token_type":"Bearer"}"#);
        let ok_leaked: &'static str = Box::leak(ok.into_boxed_str());
        let (base_url, count) = spawn_stub(vec![
            "HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\n\r\n",
            ok_leaked,
        ])
        .await;

        let http = reqwest::Client::new();
        let ts = TokenSource::metadata(http, base_url);

        let token = ts.access_token(&["scope-a"]).await.unwrap();
        assert_eq!(token, "ya29.retry");
        assert_eq!(
            count.load(Ordering::SeqCst),
            2,
            "should have retried once after the 503"
        );
    }

    #[tokio::test]
    async fn metadata_token_permanent_4xx_no_retry() {
        let (base_url, count) =
            spawn_stub(vec!["HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n"]).await;

        let http = reqwest::Client::new();
        let ts = TokenSource::metadata(http, base_url);

        let err = ts.access_token(&["scope-a"]).await.unwrap_err();
        assert!(
            format!("{err:?}").contains("403"),
            "error should mention the 403 status, got: {err:?}"
        );
        assert_eq!(
            count.load(Ordering::SeqCst),
            1,
            "4xx is permanent — must not retry"
        );
    }

    #[tokio::test]
    async fn metadata_token_malformed_body_is_permanent() {
        let resp = ok_body(r#"{"unexpected":"shape"}"#);
        let leaked: &'static str = Box::leak(resp.into_boxed_str());
        let (base_url, count) = spawn_stub(vec![leaked]).await;

        let http = reqwest::Client::new();
        let ts = TokenSource::metadata(http, base_url);

        let err = ts.access_token(&["scope-a"]).await.unwrap_err();
        assert!(
            format!("{err:?}").to_lowercase().contains("malformed"),
            "error should mention malformed, got: {err:?}"
        );
        assert_eq!(
            count.load(Ordering::SeqCst),
            1,
            "malformed body is permanent — must not retry"
        );
    }
}
