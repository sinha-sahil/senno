# Phase 3 — Helpers

## Objective

Port the support machinery the Vertex provider needs: env resolution and GCP
service-account token fetching.

## Tasks

- [x] `src/providers/vertex/env.rs`: copy `resolve_optional` + `resolve_with_default`
  from arche `src/config/resolve.rs` (both infallible; `resolve_required*` not needed)
- [x] `src/providers/vertex/token.rs`: port arche `src/gcp/token.rs` with:
  - `ServiceAccountKey` (new/from_path/from_json, token_uri override, redacted
    Debug) — **drop** `sign_rs256_sha256`, `parsed_key` cache, `parse_rsa_pem`
    (GCS signed-URL concerns; removes `rsa` + `sha2` deps)
  - `TokenSource` with scope-sorted cache, per-key locks, transient/permanent
    fetch split, retry + timeout; **drop** `signer_email`/`sign_blob`
  - Keep metadata-server auth and make `TokenSource` + constructors public API
    so nothing is dead code and the tested retry/cache logic keeps its coverage
- [ ] Error mapping: `gcp-oauth2` fetch failures → `Error::Provider` (retryable
  by transience); metadata-server failures → `Error::Provider("gcp-metadata")`;
  key parsing → `Error::Config`

## Outputs

`src/providers/vertex/env.rs`, `src/providers/vertex/token.rs`

## Validation

token.rs stub-server tests (cache hit, 5xx retry, 4xx permanent, malformed
permanent) pass under `--features vertex`.
