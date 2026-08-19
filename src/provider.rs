use crate::error::Error;
use crate::types::{
    BatchJob, EmbedRequest, EmbedResponse, Embedding, GenerateRequest, GenerateResponse,
    StreamChunk,
};
use futures::Stream;
use std::future::Future;
use std::pin::Pin;

pub type LlmStream = Pin<Box<dyn Stream<Item = Result<StreamChunk, Error>> + Send>>;

type StreamGenerateFuture<'a> = Pin<Box<dyn Future<Output = Result<LlmStream, Error>> + Send + 'a>>;

type GenerateFuture<'a> =
    Pin<Box<dyn Future<Output = Result<GenerateResponse, Error>> + Send + 'a>>;

type EmbedFuture<'a> = Pin<Box<dyn Future<Output = Result<EmbedResponse, Error>> + Send + 'a>>;

type BatchJobFuture<'a, T> = Pin<Box<dyn Future<Output = Result<BatchJob<T>, Error>> + Send + 'a>>;

type CancelFuture<'a> = Pin<Box<dyn Future<Output = Result<(), Error>> + Send + 'a>>;

pub trait LlmProvider: Send + Sync {
    fn generate<'a>(&'a self, request: &'a GenerateRequest) -> GenerateFuture<'a>;

    fn stream_generate<'a>(&'a self, request: &'a GenerateRequest) -> StreamGenerateFuture<'a>;
}

pub trait EmbeddingProvider: Send + Sync {
    fn embed<'a>(&'a self, request: &'a EmbedRequest) -> EmbedFuture<'a>;
}

pub trait BatchGenerationProvider: Send + Sync {
    fn create_batch<'a>(
        &'a self,
        requests: &'a [GenerateRequest],
    ) -> BatchJobFuture<'a, GenerateResponse>;

    fn get_batch<'a>(&'a self, name: &'a str) -> BatchJobFuture<'a, GenerateResponse>;

    fn cancel_batch<'a>(&'a self, name: &'a str) -> CancelFuture<'a>;
}

pub trait BatchEmbeddingProvider: Send + Sync {
    fn create_embedding_batch<'a>(
        &'a self,
        request: &'a EmbedRequest,
    ) -> BatchJobFuture<'a, Embedding>;

    fn get_embedding_batch<'a>(&'a self, name: &'a str) -> BatchJobFuture<'a, Embedding>;

    fn cancel_embedding_batch<'a>(&'a self, name: &'a str) -> CancelFuture<'a>;
}
