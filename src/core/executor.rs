use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Instant;

use rayon::{ThreadPool, ThreadPoolBuilder};

use crate::TantivyError;

static NEXT_THREAD_POOL_EXECUTOR_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct MapExecutionTelemetry {
    pub args_collect_us: u64,
    pub spawn_to_start_total_us: u64,
    pub spawn_to_start_p50_us: u64,
    pub spawn_to_start_p95_us: u64,
    pub spawn_to_start_p99_us: u64,
    pub spawn_to_start_max_us: u64,
    pub spawn_loop_total_us: u64,
    pub scope_total_us: u64,
    pub scope_wait_after_spawn_loop_us: u64,
    pub scope_wait_after_last_task_finish_us: u64,
    pub result_drain_total_us: u64,
    pub first_task_start_since_scope_start_us: u64,
    pub last_task_start_since_scope_start_us: u64,
    pub first_task_finish_since_scope_start_us: u64,
    pub last_task_finish_since_scope_start_us: u64,
    pub tasks_started_before_spawn_loop_end: usize,
    pub tasks_started_after_spawn_loop_end: usize,
    pub worker_thread_count: usize,
}

#[derive(Clone, Copy, Debug, Default)]
struct DurationSummary {
    total_us: u64,
    p50_us: u64,
    p95_us: u64,
    p99_us: u64,
    max_us: u64,
}

#[derive(Default)]
struct MapExecutionTimingState {
    spawn_to_start_times_us: Vec<u64>,
    worker_threads: HashSet<String>,
    first_task_start_since_scope_start_us: Option<u64>,
    last_task_start_since_scope_start_us: u64,
    first_task_finish_since_scope_start_us: Option<u64>,
    last_task_finish_since_scope_start_us: u64,
    tasks_started_before_spawn_loop_end: usize,
    tasks_started_after_spawn_loop_end: usize,
}

#[derive(Default)]
struct MapExecutionTimings {
    state: Mutex<MapExecutionTimingState>,
}

impl MapExecutionTimings {
    fn record(
        &self,
        spawn_to_start_time_us: u64,
        worker_thread: String,
        started_before_spawn_loop_end: bool,
        started_since_scope_start_us: u64,
        finished_since_scope_start_us: u64,
    ) {
        let mut state = self
            .state
            .lock()
            .expect("map execution timing lock should not be poisoned");
        state.spawn_to_start_times_us.push(spawn_to_start_time_us);
        state.worker_threads.insert(worker_thread);
        match state.first_task_start_since_scope_start_us {
            Some(current_first) => {
                if started_since_scope_start_us < current_first {
                    state.first_task_start_since_scope_start_us = Some(started_since_scope_start_us);
                }
            }
            None => {
                state.first_task_start_since_scope_start_us = Some(started_since_scope_start_us);
            }
        }
        if started_since_scope_start_us > state.last_task_start_since_scope_start_us {
            state.last_task_start_since_scope_start_us = started_since_scope_start_us;
        }
        match state.first_task_finish_since_scope_start_us {
            Some(current_first) => {
                if finished_since_scope_start_us < current_first {
                    state.first_task_finish_since_scope_start_us =
                        Some(finished_since_scope_start_us);
                }
            }
            None => {
                state.first_task_finish_since_scope_start_us = Some(finished_since_scope_start_us);
            }
        }
        if finished_since_scope_start_us > state.last_task_finish_since_scope_start_us {
            state.last_task_finish_since_scope_start_us = finished_since_scope_start_us;
        }
        if started_before_spawn_loop_end {
            state.tasks_started_before_spawn_loop_end += 1;
        } else {
            state.tasks_started_after_spawn_loop_end += 1;
        }
    }

    fn summary(
        &self,
        args_collect_us: u64,
        spawn_loop_total_us: u64,
        scope_total_us: u64,
        result_drain_total_us: u64,
    ) -> MapExecutionTelemetry {
        let state = self
            .state
            .lock()
            .expect("map execution timing lock should not be poisoned");
        let spawn_to_start_summary = duration_summary(&state.spawn_to_start_times_us);
        MapExecutionTelemetry {
            args_collect_us,
            spawn_to_start_total_us: spawn_to_start_summary.total_us,
            spawn_to_start_p50_us: spawn_to_start_summary.p50_us,
            spawn_to_start_p95_us: spawn_to_start_summary.p95_us,
            spawn_to_start_p99_us: spawn_to_start_summary.p99_us,
            spawn_to_start_max_us: spawn_to_start_summary.max_us,
            spawn_loop_total_us,
            scope_total_us,
            scope_wait_after_spawn_loop_us: scope_total_us.saturating_sub(spawn_loop_total_us),
            scope_wait_after_last_task_finish_us: scope_total_us.saturating_sub(
                state.last_task_finish_since_scope_start_us,
            ),
            result_drain_total_us,
            first_task_start_since_scope_start_us: state
                .first_task_start_since_scope_start_us
                .unwrap_or(0),
            last_task_start_since_scope_start_us: state.last_task_start_since_scope_start_us,
            first_task_finish_since_scope_start_us: state
                .first_task_finish_since_scope_start_us
                .unwrap_or(0),
            last_task_finish_since_scope_start_us: state.last_task_finish_since_scope_start_us,
            tasks_started_before_spawn_loop_end: state.tasks_started_before_spawn_loop_end,
            tasks_started_after_spawn_loop_end: state.tasks_started_after_spawn_loop_end,
            worker_thread_count: state.worker_threads.len(),
        }
    }
}

fn duration_summary(values_us: &[u64]) -> DurationSummary {
    if values_us.is_empty() {
        return DurationSummary::default();
    }

    let mut sorted = values_us.to_vec();
    sorted.sort_unstable();

    DurationSummary {
        total_us: sorted.iter().sum(),
        p50_us: percentile(&sorted, 50),
        p95_us: percentile(&sorted, 95),
        p99_us: percentile(&sorted, 99),
        max_us: *sorted.last().unwrap_or(&0),
    }
}

fn percentile(sorted_values_us: &[u64], percentile: usize) -> u64 {
    if sorted_values_us.is_empty() {
        return 0;
    }
    let last_idx = sorted_values_us.len() - 1;
    let idx = (last_idx * percentile) / 100;
    sorted_values_us[idx]
}

fn duration_to_us(duration: std::time::Duration) -> u64 {
    duration.as_micros().try_into().unwrap_or(u64::MAX)
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ExecutorTelemetrySnapshot {
    pub executor_id: u64,
    pub num_threads: usize,
    pub age_us: u64,
    pub map_calls: u64,
    pub started_threads: usize,
    pub first_thread_start_delay_us: u64,
    pub last_thread_start_delay_us: u64,
}

struct ThreadPoolTelemetry {
    executor_id: u64,
    num_threads: usize,
    created_at: Instant,
    map_calls: AtomicU64,
    started_threads: AtomicUsize,
    first_thread_start_delay_us: AtomicU64,
    last_thread_start_delay_us: AtomicU64,
}

impl ThreadPoolTelemetry {
    fn new(num_threads: usize) -> Self {
        Self {
            executor_id: NEXT_THREAD_POOL_EXECUTOR_ID.fetch_add(1, Ordering::Relaxed),
            num_threads,
            created_at: Instant::now(),
            map_calls: AtomicU64::new(0),
            started_threads: AtomicUsize::new(0),
            first_thread_start_delay_us: AtomicU64::new(u64::MAX),
            last_thread_start_delay_us: AtomicU64::new(0),
        }
    }

    fn record_thread_started(&self) {
        let start_delay_us = self
            .created_at
            .elapsed()
            .as_micros()
            .try_into()
            .unwrap_or(u64::MAX);
        self.started_threads.fetch_add(1, Ordering::Relaxed);

        let mut current_first = self.first_thread_start_delay_us.load(Ordering::Relaxed);
        while start_delay_us < current_first {
            match self.first_thread_start_delay_us.compare_exchange_weak(
                current_first,
                start_delay_us,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(previous) => current_first = previous,
            }
        }

        let mut current_last = self.last_thread_start_delay_us.load(Ordering::Relaxed);
        while start_delay_us > current_last {
            match self.last_thread_start_delay_us.compare_exchange_weak(
                current_last,
                start_delay_us,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(previous) => current_last = previous,
            }
        }
    }

    fn increment_map_calls(&self) {
        self.map_calls.fetch_add(1, Ordering::Relaxed);
    }

    fn snapshot(&self) -> ExecutorTelemetrySnapshot {
        let first_thread_start_delay_us = self.first_thread_start_delay_us.load(Ordering::Relaxed);
        ExecutorTelemetrySnapshot {
            executor_id: self.executor_id,
            num_threads: self.num_threads,
            age_us: self
                .created_at
                .elapsed()
                .as_micros()
                .try_into()
                .unwrap_or(u64::MAX),
            map_calls: self.map_calls.load(Ordering::Relaxed),
            started_threads: self.started_threads.load(Ordering::Relaxed),
            first_thread_start_delay_us: if first_thread_start_delay_us == u64::MAX {
                0
            } else {
                first_thread_start_delay_us
            },
            last_thread_start_delay_us: self.last_thread_start_delay_us.load(Ordering::Relaxed),
        }
    }
}

#[allow(missing_docs)]
#[doc(hidden)]
pub struct ThreadPoolExecutor {
    pool: ThreadPool,
    telemetry: Arc<ThreadPoolTelemetry>,
}

/// Search executor whether search request are single thread or multithread.
///
/// We don't expose Rayon thread pool directly here for several reasons.
///
/// First dependency hell. It is not a good idea to expose the
/// API of a dependency, knowing it might conflict with a different version
/// used by the client. Second, we may stop using rayon in the future.
pub enum Executor {
    /// Single thread variant of an Executor
    SingleThread,
    /// Thread pool variant of an Executor
    ThreadPool(ThreadPoolExecutor),
}

impl Executor {
    /// Creates an Executor that performs all task in the caller thread.
    pub fn single_thread() -> Executor {
        Executor::SingleThread
    }

    /// Creates an Executor that dispatches the tasks in a thread pool.
    pub fn multi_thread(num_threads: usize, prefix: &'static str) -> crate::Result<Executor> {
        let telemetry = Arc::new(ThreadPoolTelemetry::new(num_threads));
        let pool = ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .thread_name(move |num| format!("{prefix}{num}"))
            .start_handler({
                let telemetry = telemetry.clone();
                move |_| telemetry.record_thread_started()
            })
            .build()?;
        Ok(Executor::ThreadPool(ThreadPoolExecutor { pool, telemetry }))
    }

    /// Returns executor lifecycle telemetry for multithreaded executors.
    pub fn telemetry_snapshot(&self) -> Option<ExecutorTelemetrySnapshot> {
        match self {
            Executor::SingleThread => None,
            Executor::ThreadPool(thread_pool_executor) => {
                Some(thread_pool_executor.telemetry.snapshot())
            }
        }
    }

    /// Perform a map in the thread pool.
    ///
    /// Regardless of the executor (`SingleThread` or `ThreadPool`), panics in the task
    /// will propagate to the caller.
    pub fn map<
        A: Send,
        R: Send,
        AIterator: Iterator<Item = A>,
        F: Sized + Sync + Fn(A) -> crate::Result<R>,
    >(
        &self,
        f: F,
        args: AIterator,
    ) -> crate::Result<Vec<R>> {
        let (results, _telemetry) = self.map_with_telemetry(f, args)?;
        Ok(results)
    }

    pub(crate) fn map_with_telemetry<
        A: Send,
        R: Send,
        AIterator: Iterator<Item = A>,
        F: Sized + Sync + Fn(A) -> crate::Result<R>,
    >(
        &self,
        f: F,
        args: AIterator,
    ) -> crate::Result<(Vec<R>, MapExecutionTelemetry)> {
        match self {
            Executor::SingleThread => {
                let args_collect_start = Instant::now();
                let args: Vec<A> = args.collect();
                let args_collect_us = duration_to_us(args_collect_start.elapsed());
                let mut results = Vec::new();
                let scope_start = Instant::now();
                let mut first_task_start_since_scope_start_us = None;
                let mut first_task_finish_since_scope_start_us = None;
                let mut last_task_start_since_scope_start_us = 0;
                let mut last_task_finish_since_scope_start_us = 0;
                for arg in args {
                    let started_at = Instant::now();
                    let started_since_scope_start_us =
                        duration_to_us(started_at.duration_since(scope_start));
                    results.push(f(arg)?);
                    let finished_since_scope_start_us = duration_to_us(scope_start.elapsed());
                    match first_task_start_since_scope_start_us {
                        Some(current_first) => {
                            if started_since_scope_start_us < current_first {
                                first_task_start_since_scope_start_us =
                                    Some(started_since_scope_start_us);
                            }
                        }
                        None => {
                            first_task_start_since_scope_start_us = Some(started_since_scope_start_us);
                        }
                    }
                    if started_since_scope_start_us > last_task_start_since_scope_start_us {
                        last_task_start_since_scope_start_us = started_since_scope_start_us;
                    }
                    match first_task_finish_since_scope_start_us {
                        Some(current_first) => {
                            if finished_since_scope_start_us < current_first {
                                first_task_finish_since_scope_start_us =
                                    Some(finished_since_scope_start_us);
                            }
                        }
                        None => {
                            first_task_finish_since_scope_start_us =
                                Some(finished_since_scope_start_us);
                        }
                    }
                    if finished_since_scope_start_us > last_task_finish_since_scope_start_us {
                        last_task_finish_since_scope_start_us = finished_since_scope_start_us;
                    }
                }
                let scope_total_us = duration_to_us(scope_start.elapsed());
                let num_tasks = results.len();
                Ok((
                    results,
                    MapExecutionTelemetry {
                        args_collect_us,
                        scope_total_us,
                        scope_wait_after_spawn_loop_us: 0,
                        scope_wait_after_last_task_finish_us: 0,
                        first_task_start_since_scope_start_us: first_task_start_since_scope_start_us
                            .unwrap_or(0),
                        last_task_start_since_scope_start_us,
                        first_task_finish_since_scope_start_us:
                            first_task_finish_since_scope_start_us.unwrap_or(0),
                        last_task_finish_since_scope_start_us,
                        tasks_started_before_spawn_loop_end: num_tasks,
                        worker_thread_count: usize::from(num_tasks > 0),
                        ..MapExecutionTelemetry::default()
                    },
                ))
            }
            Executor::ThreadPool(thread_pool_executor) => {
                thread_pool_executor.telemetry.increment_map_calls();
                let args_collect_start = Instant::now();
                let args: Vec<A> = args.collect();
                let args_collect_us = duration_to_us(args_collect_start.elapsed());
                let num_fruits = args.len();
                let timings = Arc::new(MapExecutionTimings::default());
                let spawn_loop_finished = Arc::new(AtomicBool::new(false));
                let spawn_loop_total_us = Arc::new(AtomicU64::new(0));
                let scope_start = Instant::now();
                let fruit_receiver = {
                    let (fruit_sender, fruit_receiver) = crossbeam_channel::unbounded();
                    let parent_span = tracing::Span::current();
                    thread_pool_executor.pool.scope(|scope| {
                        let _parent_span_guard = parent_span.enter();
                        let spawn_loop_start = Instant::now();
                        for (idx, arg) in args.into_iter().enumerate() {
                            // We name references for f and fruit_sender_ref because we do not
                            // want these two to be moved into the closure.
                            let f_ref = &f;
                            let fruit_sender_ref = &fruit_sender;
                            let parent_span = tracing::Span::current();
                            let timings = timings.clone();
                            let spawn_loop_finished = spawn_loop_finished.clone();
                            let scope_start = scope_start;
                            let spawned_at = Instant::now();
                            scope.spawn(move |_| {
                                let _parent_span_guard = parent_span.enter();
                                let started_at = Instant::now();
                                let worker_thread = std::thread::current();
                                let worker_thread_name = worker_thread
                                    .name()
                                    .map(ToOwned::to_owned)
                                    .unwrap_or_else(|| format!("{:?}", worker_thread.id()));
                                let fruit = f_ref(arg);
                                let finished_since_scope_start_us =
                                    duration_to_us(scope_start.elapsed());
                                timings.record(
                                    duration_to_us(started_at.duration_since(spawned_at)),
                                    worker_thread_name,
                                    !spawn_loop_finished.load(Ordering::Relaxed),
                                    duration_to_us(started_at.duration_since(scope_start)),
                                    finished_since_scope_start_us,
                                );
                                if let Err(err) = fruit_sender_ref.send((idx, fruit)) {
                                    error!(
                                        "Failed to send search task. It probably means all search \
                                         threads have panicked. {:?}",
                                        err
                                    );
                                }
                            });
                        }
                        spawn_loop_total_us.store(
                            duration_to_us(spawn_loop_start.elapsed()),
                            Ordering::Relaxed,
                        );
                        spawn_loop_finished.store(true, Ordering::Relaxed);
                    });
                    fruit_receiver
                    // This ends the scope of fruit_sender.
                    // This is important as it makes it possible for the fruit_receiver iteration to
                    // terminate.
                };
                let scope_total_us = duration_to_us(scope_start.elapsed());
                let mut result_placeholders: Vec<Option<R>> =
                    std::iter::repeat_with(|| None).take(num_fruits).collect();
                let result_drain_start = Instant::now();
                for (pos, fruit_res) in fruit_receiver {
                    let fruit = fruit_res?;
                    result_placeholders[pos] = Some(fruit);
                }
                let result_drain_total_us = duration_to_us(result_drain_start.elapsed());
                let results: Vec<R> = result_placeholders.into_iter().flatten().collect();
                if results.len() != num_fruits {
                    return Err(TantivyError::InternalError(
                        "One of the mapped execution failed.".to_string(),
                    ));
                }
                Ok((
                    results,
                    timings.summary(
                        args_collect_us,
                        spawn_loop_total_us.load(Ordering::Relaxed),
                        scope_total_us,
                        result_drain_total_us,
                    ),
                ))
            }
        }
    }

    /// Spawn a task in the thread pool
    pub fn spawn<OP, R>(&self, op: OP) -> crate::Result<impl Fn() -> crate::Result<R>>
    where
        R: Send + 'static,
        OP: Fn() -> crate::Result<R> + Send + 'static,
    {
        let (fruit_sender, fruit_receiver) = crossbeam_channel::unbounded();
        match self {
            Executor::SingleThread => {
                let res = op();
                fruit_sender.send(res).map_err(|_| {
                    TantivyError::InternalError(
                        "Failed to send search task. It probably means all search \
                     threads have panicked."
                            .to_string(),
                    )
                })?;
            }
            Executor::ThreadPool(thread_pool_executor) => {
                let parent_span = tracing::Span::current();
                thread_pool_executor.pool.spawn(move || {
                    let _parent_span_guard = parent_span.enter();
                    let res = op();
                    match fruit_sender.send(res) {
                        Ok(_) => (),
                        Err(_) => error!(
                            "Failed to send search task. It probably means all search \
                     threads have panicked."
                        ),
                    }
                });
            }
        };

        Ok(move || {
            fruit_receiver.recv().map_err(|_| {
                TantivyError::InternalError(
                    "Failed to send search task. It probably means all search \
                     threads have panicked."
                        .to_string(),
                )
            })?
        })
    }
}

#[cfg(test)]
mod tests {

    use super::Executor;

    #[test]
    #[should_panic(expected = "panic should propagate")]
    fn test_panic_propagates_single_thread() {
        let _result: Vec<usize> = Executor::single_thread()
            .map(
                |_| {
                    panic!("panic should propagate");
                },
                vec![0].into_iter(),
            )
            .unwrap();
    }

    #[test]
    #[should_panic] //< unfortunately the panic message is not propagated
    fn test_panic_propagates_multi_thread() {
        let _result: Vec<usize> = Executor::multi_thread(1, "search-test")
            .unwrap()
            .map(
                |_| {
                    panic!("panic should propagate");
                },
                vec![0].into_iter(),
            )
            .unwrap();
    }

    #[test]
    fn test_map_singlethread() {
        let result: Vec<usize> = Executor::single_thread()
            .map(|i| Ok(i * 2), 0..1_000)
            .unwrap();
        assert_eq!(result.len(), 1_000);
        for i in 0..1_000 {
            assert_eq!(result[i], i * 2);
        }
    }

    #[test]
    fn test_map_multithread() {
        let result: Vec<usize> = Executor::multi_thread(3, "search-test")
            .unwrap()
            .map(|i| Ok(i * 2), 0..10)
            .unwrap();
        assert_eq!(result.len(), 10);
        for i in 0..10 {
            assert_eq!(result[i], i * 2);
        }
    }

    #[test]
    #[ignore] // This test demonstrates a deadlock - run with --ignored to see it hang
    fn test_nested_spawn_deadlock() {
        // This reproduces the deadlock in production where:
        // 1. executor.map() spawns jobs into a Rayon pool
        // 2. Each job calls executor.spawn() which queues more work
        // 3. Each job blocks waiting for its spawned work
        // 4. All workers get blocked, nobody left to execute the spawned work
        // Result: DEADLOCK

        use std::sync::Arc;
        use std::time::Duration;

        let executor = Arc::new(Executor::multi_thread(4, "deadlock-test-").unwrap());

        // Simulate 8 segments (more than pool size of 4)
        let segments: Vec<usize> = (0..8).collect();

        let executor_clone = executor.clone();
        let result = executor.map(
            |segment_id| {
                // Simulate open_with_custom_alive_set_parallel:
                // Spawn 3 file loads per segment
                let fut1 = executor_clone.spawn(move || {
                    std::thread::sleep(Duration::from_millis(10));
                    Ok(segment_id * 100 + 1)
                })?;

                let fut2 = executor_clone.spawn(move || {
                    std::thread::sleep(Duration::from_millis(10));
                    Ok(segment_id * 100 + 2)
                })?;

                let fut3 = executor_clone.spawn(move || {
                    std::thread::sleep(Duration::from_millis(10));
                    Ok(segment_id * 100 + 3)
                })?;

                // Block waiting for results (this is where deadlock happens)
                let f1 = fut1()?;
                let f2 = fut2()?;
                let f3 = fut3()?;

                Ok((f1, f2, f3))
            },
            segments.iter().copied(),
        );

        // If you run this test, it will hang forever
        // With 4 workers and 8 segments × 3 spawns = 24 queued jobs,
        // all 4 workers block on their first spawn, leaving nobody to execute the remaining 20 jobs
        result.unwrap();
    }

    #[test]
    fn test_sequential_inner_work_no_deadlock() {
        // This shows that non-nested parallelism works fine
        use std::time::Duration;

        let executor = Executor::multi_thread(4, "safe-test-").unwrap();

        let result = executor.map(
            |segment_id| {
                // Do work directly without spawning
                std::thread::sleep(Duration::from_millis(10));
                Ok(format!("segment-{}", segment_id))
            },
            0..8,
        );

        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 8);
    }

    #[test]
    fn test_scope_based_nesting_works() {
        // This demonstrates the FIX: use scope() instead of spawn() for nested parallelism
        // scope() uses work-stealing so workers remain productive while waiting
        use std::sync::Mutex;
        use std::time::Duration;

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .thread_name(|i| format!("fix-test-{}", i))
            .build()
            .unwrap();

        let segments: Vec<usize> = (0..8).collect();

        let results: Vec<_> = pool.install(|| {
            segments
                .iter()
                .map(|&segment_id| {
                    // Use scope for nested parallelism (doesn't block!)
                    let f1 = Mutex::new(None);
                    let f2 = Mutex::new(None);
                    let f3 = Mutex::new(None);

                    pool.scope(|s| {
                        s.spawn(|_| {
                            std::thread::sleep(Duration::from_millis(10));
                            *f1.lock().unwrap() = Some(segment_id * 100 + 1);
                        });

                        s.spawn(|_| {
                            std::thread::sleep(Duration::from_millis(10));
                            *f2.lock().unwrap() = Some(segment_id * 100 + 2);
                        });

                        s.spawn(|_| {
                            std::thread::sleep(Duration::from_millis(10));
                            *f3.lock().unwrap() = Some(segment_id * 100 + 3);
                        });

                        // scope() uses work-stealing, so workers process other jobs while waiting
                    });

                    // Extract values to avoid lifetime issues
                    let v1 = f1.lock().unwrap().unwrap();
                    let v2 = f2.lock().unwrap().unwrap();
                    let v3 = f3.lock().unwrap().unwrap();
                    (v1, v2, v3)
                })
                .collect()
        });

        assert_eq!(results.len(), 8);
        // Verify results are correct
        for (i, result) in results.iter().enumerate() {
            assert_eq!(result.0, i * 100 + 1);
            assert_eq!(result.1, i * 100 + 2);
            assert_eq!(result.2, i * 100 + 3);
        }
    }
}
