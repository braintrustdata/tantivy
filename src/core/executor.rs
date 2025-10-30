use std::sync::Arc;

#[cfg(feature = "quickwit")]
use futures_util::{future::Either, FutureExt};

use crate::TantivyError;

/// Executor makes it possible to run tasks in single thread or
/// in a thread pool.
#[derive(Clone)]
pub enum Executor {
    /// Single thread variant of an Executor
    SingleThread,
    /// Thread pool variant of an Executor
    ThreadPool(Arc<rayon::ThreadPool>),
}

#[cfg(feature = "quickwit")]
impl From<Arc<rayon::ThreadPool>> for Executor {
    fn from(thread_pool: Arc<rayon::ThreadPool>) -> Self {
        Executor::ThreadPool(thread_pool)
    }
}

impl Executor {
    /// Creates an Executor that performs all task in the caller thread.
    pub fn single_thread() -> Executor {
        Executor::SingleThread
    }

    /// Creates an Executor that dispatches the tasks in a thread pool.
    pub fn multi_thread(num_threads: usize, prefix: &'static str) -> crate::Result<Executor> {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .thread_name(move |num| format!("{prefix}{num}"))
            .build()?;
        Ok(Executor::ThreadPool(Arc::new(pool)))
    }

    /// Perform a map in the thread pool.
    ///
    /// Regardless of the executor (`SingleThread` or `ThreadPool`), panics in the task
    /// will propagate to the caller.
    pub fn map<A, R, F>(&self, f: F, args: impl Iterator<Item = A>) -> crate::Result<Vec<R>>
    where
        A: Send,
        R: Send,
        F: Sized + Sync + Fn(A) -> crate::Result<R>,
    {
        match self {
            Executor::SingleThread => args.map(f).collect::<crate::Result<_>>(),
            Executor::ThreadPool(pool) => {
                let args: Vec<A> = args.collect();
                let num_fruits = args.len();
                let fruit_receiver = {
                    let (fruit_sender, fruit_receiver) = crossbeam_channel::unbounded();
                    let parent_span = tracing::Span::current();
                    pool.scope(|scope| {
                        let _parent_span_guard = parent_span.enter();
                        for (idx, arg) in args.into_iter().enumerate() {
                            // We name references for f and fruit_sender_ref because we do not
                            // want these two to be moved into the closure.
                            let f_ref = &f;
                            let fruit_sender_ref = &fruit_sender;
                            let parent_span = tracing::Span::current();
                            scope.spawn(move |_| {
                                let _parent_span_guard = parent_span.enter();
                                let fruit = f_ref(arg);
                                if let Err(err) = fruit_sender_ref.send((idx, fruit)) {
                                    error!(
                                        "Failed to send search task. It probably means all search \
                                         threads have panicked. {err:?}"
                                    );
                                }
                            });
                        }
                    });
                    fruit_receiver
                    // This ends the scope of fruit_sender.
                    // This is important as it makes it possible for the fruit_receiver iteration to
                    // terminate.
                };
                let mut result_placeholders: Vec<Option<R>> =
                    std::iter::repeat_with(|| None).take(num_fruits).collect();
                for (pos, fruit_res) in fruit_receiver {
                    let fruit = fruit_res?;
                    result_placeholders[pos] = Some(fruit);
                }
                let results: Vec<R> = result_placeholders.into_iter().flatten().collect();
                if results.len() != num_fruits {
                    return Err(TantivyError::InternalError(
                        "One of the mapped execution failed.".to_string(),
                    ));
                }
                Ok(results)
            }
        }
    }

    /// Spawn a task on the pool, returning a future completing on task success.
    ///
    /// If the task panics, returns `Err(())`.
    #[cfg(feature = "quickwit")]
    pub fn spawn_blocking<T: Send + 'static>(
        &self,
        cpu_intensive_task: impl FnOnce() -> T + Send + 'static,
    ) -> impl std::future::Future<Output = Result<T, ()>> {
        match self {
            Executor::SingleThread => Either::Left(std::future::ready(Ok(cpu_intensive_task()))),
            Executor::ThreadPool(pool) => {
                let (sender, receiver) = oneshot::channel();
                pool.spawn(|| {
                    if sender.is_closed() {
                        return;
                    }
                    let task_result = cpu_intensive_task();
                    let _ = sender.send(task_result);
                });

                let res = receiver.map(|res| res.map_err(|_| ()));
                Either::Right(res)
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
            Executor::ThreadPool(pool) => {
                let parent_span = tracing::Span::current();
                pool.spawn(move || {
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

    #[cfg(feature = "quickwit")]
    #[test]
    fn test_cancel_cpu_intensive_tasks() {
        use std::sync::atomic::{AtomicU64, Ordering};
        use std::sync::Arc;

        let counter: Arc<AtomicU64> = Default::default();

        let other_counter: Arc<AtomicU64> = Default::default();

        let mut futures = Vec::new();
        let mut other_futures = Vec::new();

        let (tx, rx) = crossbeam_channel::bounded::<()>(0);
        let rx = Arc::new(rx);
        let executor = Executor::multi_thread(3, "search-test").unwrap();
        for _ in 0..1000 {
            let counter_clone: Arc<AtomicU64> = counter.clone();
            let other_counter_clone: Arc<AtomicU64> = other_counter.clone();

            let rx_clone = rx.clone();
            let rx_clone2 = rx.clone();
            let fut = executor.spawn_blocking(move || {
                counter_clone.fetch_add(1, Ordering::SeqCst);
                let _ = rx_clone.recv();
            });
            futures.push(fut);
            let other_fut = executor.spawn_blocking(move || {
                other_counter_clone.fetch_add(1, Ordering::SeqCst);
                let _ = rx_clone2.recv();
            });
            other_futures.push(other_fut);
        }

        // We execute 100 futures.
        for _ in 0..100 {
            tx.send(()).unwrap();
        }

        let counter_val = counter.load(Ordering::SeqCst);
        let other_counter_val = other_counter.load(Ordering::SeqCst);
        assert!(counter_val >= 30);
        assert!(other_counter_val >= 30);

        drop(other_futures);

        // We execute 100 futures.
        for _ in 0..100 {
            tx.send(()).unwrap();
        }

        let counter_val2 = counter.load(Ordering::SeqCst);
        assert!(counter_val2 >= counter_val + 100 - 6);

        let other_counter_val2 = other_counter.load(Ordering::SeqCst);
        assert!(other_counter_val2 <= other_counter_val + 6);
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
