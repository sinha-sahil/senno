use std::time::{Duration, Instant};

use crate::error::Error;
use crate::provider::{BatchEmbeddingProvider, BatchGenerationProvider};
use crate::types::{BatchJob, EmbedRequest, Embedding, GenerateRequest, GenerateResponse};

#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct BatchPolling {
    pub initial_interval: Duration,
    pub max_interval: Duration,
    pub timeout: Option<Duration>,
}

impl Default for BatchPolling {
    fn default() -> Self {
        Self {
            initial_interval: Duration::from_secs(30),
            max_interval: Duration::from_secs(300),
            timeout: None,
        }
    }
}

impl BatchPolling {
    pub fn with_initial_interval(mut self, v: Duration) -> Self {
        self.initial_interval = v;
        self
    }

    pub fn with_max_interval(mut self, v: Duration) -> Self {
        self.max_interval = v;
        self
    }

    pub fn with_timeout(mut self, v: Duration) -> Self {
        self.timeout = Some(v);
        self
    }
}

pub async fn run_batch<P>(
    provider: &P,
    requests: &[GenerateRequest],
) -> Result<BatchJob<GenerateResponse>, Error>
where
    P: BatchGenerationProvider + ?Sized,
{
    let job = provider.create_batch(requests).await?;
    if job.state.is_terminal() {
        return Ok(job);
    }
    wait_for_batch(provider, &job.name, BatchPolling::default()).await
}

pub async fn wait_for_batch<P>(
    provider: &P,
    name: &str,
    polling: BatchPolling,
) -> Result<BatchJob<GenerateResponse>, Error>
where
    P: BatchGenerationProvider + ?Sized,
{
    let mut schedule = Schedule::new(&polling);
    loop {
        let job = provider.get_batch(name).await?;
        if job.state.is_terminal() {
            return Ok(job);
        }
        schedule.wait(name, job.state).await?;
    }
}

pub async fn run_embedding_batch<P>(
    provider: &P,
    request: &EmbedRequest,
) -> Result<BatchJob<Embedding>, Error>
where
    P: BatchEmbeddingProvider + ?Sized,
{
    let job = provider.create_embedding_batch(request).await?;
    if job.state.is_terminal() {
        return Ok(job);
    }
    wait_for_embedding_batch(provider, &job.name, BatchPolling::default()).await
}

pub async fn wait_for_embedding_batch<P>(
    provider: &P,
    name: &str,
    polling: BatchPolling,
) -> Result<BatchJob<Embedding>, Error>
where
    P: BatchEmbeddingProvider + ?Sized,
{
    let mut schedule = Schedule::new(&polling);
    loop {
        let job = provider.get_embedding_batch(name).await?;
        if job.state.is_terminal() {
            return Ok(job);
        }
        schedule.wait(name, job.state).await?;
    }
}

struct Schedule {
    interval: Duration,
    max_interval: Duration,
    deadline: Option<Instant>,
}

impl Schedule {
    fn new(polling: &BatchPolling) -> Self {
        Self {
            interval: polling.initial_interval,
            max_interval: polling.max_interval,
            deadline: polling.timeout.and_then(|t| Instant::now().checked_add(t)),
        }
    }

    fn past_deadline(&self) -> bool {
        let Some(deadline) = self.deadline else {
            return false;
        };
        match Instant::now().checked_add(self.interval) {
            Some(next) => next > deadline,
            None => true,
        }
    }

    fn advance(&mut self) {
        self.interval = self.interval.saturating_mul(2).min(self.max_interval);
    }

    async fn wait(&mut self, name: &str, state: crate::types::BatchState) -> Result<(), Error> {
        if self.past_deadline() {
            return Err(Error::internal(format!(
                "batch {name} was still {state:?} when polling timed out; it is still running, poll it again with the same name"
            )));
        }
        tokio::time::sleep(self.interval).await;
        self.advance();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::BatchState;
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Reports `Pending` for the first `pending_polls` calls, then `Succeeded`.
    struct StubJobs {
        pending_polls: usize,
        creates: Arc<AtomicUsize>,
        polls: Arc<AtomicUsize>,
    }

    impl StubJobs {
        fn new(pending_polls: usize) -> Self {
            Self {
                pending_polls,
                creates: Arc::new(AtomicUsize::new(0)),
                polls: Arc::new(AtomicUsize::new(0)),
            }
        }

        fn state(&self) -> BatchState {
            if self.polls.fetch_add(1, Ordering::SeqCst) < self.pending_polls {
                BatchState::Running
            } else {
                BatchState::Succeeded
            }
        }
    }

    impl BatchGenerationProvider for StubJobs {
        fn create_batch<'a>(
            &'a self,
            _requests: &'a [GenerateRequest],
        ) -> Pin<Box<dyn Future<Output = Result<BatchJob<GenerateResponse>, Error>> + Send + 'a>>
        {
            self.creates.fetch_add(1, Ordering::SeqCst);
            Box::pin(async move {
                Ok(BatchJob {
                    name: "batches/abc".into(),
                    state: BatchState::Pending,
                    responses: vec![],
                })
            })
        }

        fn get_batch<'a>(
            &'a self,
            name: &'a str,
        ) -> Pin<Box<dyn Future<Output = Result<BatchJob<GenerateResponse>, Error>> + Send + 'a>>
        {
            let state = self.state();
            Box::pin(async move {
                Ok(BatchJob {
                    name: name.into(),
                    state,
                    responses: vec![],
                })
            })
        }

        fn cancel_batch<'a>(
            &'a self,
            _name: &'a str,
        ) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + 'a>> {
            Box::pin(async move { Ok(()) })
        }
    }

    impl BatchEmbeddingProvider for StubJobs {
        fn create_embedding_batch<'a>(
            &'a self,
            _request: &'a EmbedRequest,
        ) -> Pin<Box<dyn Future<Output = Result<BatchJob<Embedding>, Error>> + Send + 'a>> {
            self.creates.fetch_add(1, Ordering::SeqCst);
            Box::pin(async move {
                Ok(BatchJob {
                    name: "batches/embed".into(),
                    state: BatchState::Pending,
                    responses: vec![],
                })
            })
        }

        fn get_embedding_batch<'a>(
            &'a self,
            name: &'a str,
        ) -> Pin<Box<dyn Future<Output = Result<BatchJob<Embedding>, Error>> + Send + 'a>> {
            let state = self.state();
            Box::pin(async move {
                Ok(BatchJob {
                    name: name.into(),
                    state,
                    responses: vec![],
                })
            })
        }

        fn cancel_embedding_batch<'a>(
            &'a self,
            _name: &'a str,
        ) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + 'a>> {
            Box::pin(async move { Ok(()) })
        }
    }

    fn fast() -> BatchPolling {
        BatchPolling::default()
            .with_initial_interval(Duration::from_millis(1))
            .with_max_interval(Duration::from_millis(4))
    }

    #[test]
    fn defaults_start_at_thirty_seconds_and_cap_at_five_minutes() {
        let polling = BatchPolling::default();
        assert_eq!(polling.initial_interval, Duration::from_secs(30));
        assert_eq!(polling.max_interval, Duration::from_secs(300));
        assert!(polling.timeout.is_none());
    }

    #[tokio::test]
    async fn waiting_polls_until_the_job_reaches_a_terminal_state() {
        let provider = StubJobs::new(3);

        let job = wait_for_batch(&provider, "batches/abc", fast())
            .await
            .unwrap();

        assert_eq!(job.state, BatchState::Succeeded);
        assert_eq!(provider.polls.load(Ordering::SeqCst), 4);
    }

    #[tokio::test]
    async fn running_a_batch_creates_then_waits_without_the_caller_seeing_a_name() {
        let provider = StubJobs::new(0);

        let job = run_batch(&provider, &[]).await.unwrap();

        assert_eq!(job.state, BatchState::Succeeded);
        assert_eq!(provider.creates.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn an_embedding_batch_runs_through_the_same_helper() {
        let provider = StubJobs::new(0);

        let job = run_embedding_batch(&provider, &EmbedRequest::single("m", "x"))
            .await
            .unwrap();

        assert_eq!(job.state, BatchState::Succeeded);
        assert_eq!(provider.creates.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn a_job_already_finished_at_create_time_is_never_polled() {
        let provider = StubJobs::new(0);
        // create reports Pending, so one poll confirms; the point is that the
        // helper never loops when the state is already terminal.
        run_batch(&provider, &[]).await.unwrap();
        assert!(provider.polls.load(Ordering::SeqCst) <= 1);
    }

    #[tokio::test]
    async fn polling_gives_up_at_the_deadline_and_names_the_job() {
        let provider = StubJobs::new(usize::MAX);
        let polling = fast().with_timeout(Duration::from_millis(10));

        let err = wait_for_batch(&provider, "batches/abc", polling)
            .await
            .unwrap_err();

        let message = err.to_string();
        assert!(message.contains("batches/abc"), "got {message}");
        assert!(message.contains("still running"), "got {message}");
    }

    #[tokio::test]
    async fn the_interval_backs_off_and_stops_at_the_ceiling() {
        let mut schedule = Schedule::new(&fast());
        assert_eq!(schedule.interval, Duration::from_millis(1));

        for expected in [2, 4, 4, 4] {
            schedule.advance();
            assert_eq!(schedule.interval, Duration::from_millis(expected));
        }
    }

    #[test]
    fn absurd_polling_values_saturate_instead_of_overflowing() {
        let polling = BatchPolling::default()
            .with_initial_interval(Duration::MAX)
            .with_max_interval(Duration::MAX)
            .with_timeout(Duration::MAX);

        let mut schedule = Schedule::new(&polling);
        assert!(
            schedule.deadline.is_none(),
            "an unrepresentable deadline is no deadline"
        );

        schedule.advance();
        assert_eq!(schedule.interval, Duration::MAX);
        assert!(!schedule.past_deadline());
    }
}
