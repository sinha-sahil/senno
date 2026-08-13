# Implementation Checklist

## Phase 1: Types

- [x] `src/error.rs` — senno::Error with provider/bad-response/config/tool variants
- [x] `src/types.rs` — LLM types ported, `parse()` → `Error::BadResponse`
- [x] `src/provider.rs` + `src/one_shot.rs`
- [x] `src/agent/types.rs` + `src/agent/config.rs`

## Phase 3: Helpers

- [x] `src/providers/vertex/env.rs` — resolve_optional / resolve_with_default
- [x] `src/providers/vertex/token.rs` — ServiceAccountKey + TokenSource
      (rsa/sha2 dropped, metadata auth public)

## Phase 7: Remote (Vertex)

- [x] `src/providers/vertex/client.rs` + `config.rs` + `mod.rs`
      (public path: `senno::providers::vertex`)
- [x] `src/providers/vertex/gemini.rs`
- [x] `src/providers/vertex/anthropic.rs`

## Phase 9: Integration

- [x] `src/agent/engine.rs` + `compactor.rs` + `mod.rs`; `stream.rs` behind `axum`
- [x] `src/lib.rs` surface + `Cargo.toml` features
- [x] `tests/llm.rs` ported
- [x] rust-toolchain.toml / .gitignore / cargo-husky / release.yml / README

## Verification

- [x] `cargo check --all-features`
- [x] `cargo clippy --all-targets --all-features -- -D warnings`
- [x] `cargo test --all-features` — 52 unit + 8 integration, 0 failures
- [x] `cargo fmt --check`
- [x] `cargo check` (default features only — dep tree is just serde/serde_json/
      thiserror/tracing/futures/async-stream/indexmap/time)
- [x] `cargo publish --dry-run`

## Post-extraction (user)

- [ ] Initial commit on `release` (everything is staged)
- [ ] `gh repo create sinha-sahil/senno` + push `release` branch (set
      CARGO_REGISTRY_TOKEN secret + `publish` environment for the workflow)
- [ ] `cargo publish` 0.1.0 (claims the name; needs cargo login) — or let the
      release workflow do it on first push
- [ ] Register senno.rs
- [ ] arche 5.0.0: delete llm/agent/vertex per Phase 9 follow-through
