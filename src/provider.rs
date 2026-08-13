use crate::error::Error;
use crate::types::{GenerateRequest, GenerateResponse, StreamChunk};
use futures::Stream;
use std::future::Future;
use std::pin::Pin;

pub type LlmStream = Pin<Box<dyn Stream<Item = Result<StreamChunk, Error>> + Send>>;

type StreamGenerateFuture<'a> = Pin<Box<dyn Future<Output = Result<LlmStream, Error>> + Send + 'a>>;

type GenerateFuture<'a> =
    Pin<Box<dyn Future<Output = Result<GenerateResponse, Error>> + Send + 'a>>;

pub trait LlmProvider: Send + Sync {
    fn generate<'a>(&'a self, request: &'a GenerateRequest) -> GenerateFuture<'a>;

    fn stream_generate<'a>(&'a self, request: &'a GenerateRequest) -> StreamGenerateFuture<'a>;
}
