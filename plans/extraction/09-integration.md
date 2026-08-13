# Phase 9 — Integration

## Objective

Assemble the crate surface, feature flags, engine port, tests, repo scaffolding —
and define the arche-side follow-through.

## Tasks

- [ ] `src/agent/{engine,compactor,mod}.rs`: port; engine aliases
  `use crate::types as llm` to keep the diff minimal; tool errors stringify via
  `Display` (drop `detailed_error_message`)
- [ ] `src/agent/stream.rs` behind `axum` feature
- [x] `src/lib.rs`: flatten `llm` module to crate root re-exports; `pub mod
  agent`; `pub mod providers` (vertex feature-gated inside `providers/mod.rs`)
- [ ] `Cargo.toml`: core deps (serde, serde_json, thiserror, tracing, futures,
  async-stream, indexmap, time/formatting); `vertex` feature → reqwest,
  jsonwebtoken, nanoid, tokio(sync,time,fs); `axum` feature → axum;
  docs.rs all-features
- [ ] `tests/llm.rs`: port with `senno::` paths and `Error` assertions
- [ ] Scaffolding: rust-toolchain.toml (1.97.0), .gitignore, cargo-husky
  pre-commit, `.github/workflows/release.yml` (arche pipeline, senno URLs),
  README with quickstarts
- [ ] Verify: check / clippy -D warnings / test / fmt, each across default and
  all-features

## arche follow-through (rides the 5.0.0 breaking release)

- [ ] Delete `src/llm`, `src/agent`, `src/gcp/vertex`, `tests/llm.rs`
- [ ] Remove `pub mod llm` / `pub mod agent` from lib.rs, `pub mod vertex` from
  gcp/mod.rs; drop now-unused deps (indexmap, async-stream, nanoid if unused
  elsewhere)
- [ ] Remove README sections (llm, agent, Vertex AI); changelog points to senno
- [ ] Apps migrate: `arche::llm::…` → `senno::…`, add
  `impl From<senno::Error> for AppError`

## Outputs

Compiling, tested, publishable senno 0.1.0.

## Validation

Verification quartet green (see checklist); `cargo publish --dry-run` clean.
