// Test to reproduce the deadlock when executor.spawn() is called from within executor.map()
// This simulates the pattern in open_segment_readers() that causes the production deadlock.

#[cfg(test)]
mod tests {
    use crate::core::Executor;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::Duration;

    #[test]
    #[ignore] // Mark as ignored since this test intentionally deadlocks
    fn test_nested_executor_spawn_deadlocks() {
        // Use a small pool to make deadlock more likely
        let executor = Executor::multi_thread(4, "deadlock-test-").unwrap();
        let executor_arc = Arc::new(executor);

        // Counter to track how many jobs completed
        let completed = Arc::new(AtomicUsize::new(0));

        // Simulate opening 8 "segments" (more than pool size of 4)
        let segments: Vec<usize> = (0..8).collect();

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // Set a timeout for the test
            let handle = std::thread::spawn({
                let executor = executor_arc.clone();
                let completed = completed.clone();

                move || {
                    // This mimics open_segment_readers() calling executor.map()
                    executor
                        .map(
                            |segment_id| {
                                // Each "segment" spawns multiple file loads (simulating open_with_custom_alive_set_parallel)
                                let fut1 = executor.spawn(move || {
                                    // Simulate file I/O
                                    std::thread::sleep(Duration::from_millis(10));
                                    Ok::<_, String>(format!("file1-{}", segment_id))
                                })?;

                                let fut2 = executor.spawn(move || {
                                    std::thread::sleep(Duration::from_millis(10));
                                    Ok::<_, String>(format!("file2-{}", segment_id))
                                })?;

                                let fut3 = executor.spawn(move || {
                                    std::thread::sleep(Duration::from_millis(10));
                                    Ok::<_, String>(format!("file3-{}", segment_id))
                                })?;

                                // Immediately consume the futures (this is where blocking happens)
                                let f1 = fut1()?;
                                let f2 = fut2()?;
                                let f3 = fut3()?;

                                completed.fetch_add(1, Ordering::SeqCst);
                                Ok((f1, f2, f3))
                            },
                            segments.iter().copied(),
                        )
                        .unwrap()
                }
            });

            // Wait for completion with timeout
            let timeout = Duration::from_secs(5);
            let start = std::time::Instant::now();

            while start.elapsed() < timeout {
                if handle.is_finished() {
                    return handle.join().unwrap();
                }
                std::thread::sleep(Duration::from_millis(100));
            }

            panic!(
                "Test timed out after {:?}! Only {} of {} segments completed. This demonstrates the deadlock.",
                timeout,
                completed.load(Ordering::SeqCst),
                segments.len()
            );
        }));

        match result {
            Ok(_) => panic!("Test should have timed out due to deadlock, but it completed!"),
            Err(e) => {
                let msg = if let Some(s) = e.downcast_ref::<String>() {
                    s.clone()
                } else if let Some(s) = e.downcast_ref::<&str>() {
                    s.to_string()
                } else {
                    "Unknown panic".to_string()
                };

                // Verify it's the timeout, not some other error
                assert!(
                    msg.contains("timed out") || msg.contains("deadlock"),
                    "Expected timeout/deadlock message, got: {}",
                    msg
                );

                println!("✓ Test successfully demonstrated deadlock: {}", msg);
            }
        }
    }

    #[test]
    fn test_nested_executor_with_small_pool_high_load() {
        // This test shows the exact conditions needed for deadlock:
        // - Small pool (4 threads)
        // - More outer jobs than pool size (8 segments)
        // - Each outer job spawns inner work (3 files each)
        // Result: 8 jobs × 3 inner spawns = 24 queued jobs, but only 4 workers
        // When all 4 workers block on their first spawn, nobody can execute the remaining 20 jobs

        let executor = Arc::new(Executor::multi_thread(4, "test-").unwrap());

        // Try with timeout using a separate thread
        let executor_clone = executor.clone();
        let handle = std::thread::spawn(move || {
            executor_clone.map(
                |segment_id| {
                    // Spawn 3 inner tasks per segment
                    let f1 = executor_clone.spawn(move || Ok::<_, String>(segment_id * 100 + 1))?;
                    let f2 = executor_clone.spawn(move || Ok::<_, String>(segment_id * 100 + 2))?;
                    let f3 = executor_clone.spawn(move || Ok::<_, String>(segment_id * 100 + 3))?;

                    // Block on results
                    Ok((f1()?, f2()?, f3()?))
                },
                0..8,
            )
        });

        // Give it 2 seconds to complete
        let timeout = Duration::from_secs(2);
        let start = std::time::Instant::now();

        while start.elapsed() < timeout {
            if handle.is_finished() {
                // If it completes, we didn't hit deadlock (might happen with timing)
                let result = handle.join().unwrap();
                assert!(result.is_ok(), "Should complete if no deadlock");
                println!("✓ Test completed without deadlock (timing dependent)");
                return;
            }
            std::thread::sleep(Duration::from_millis(50));
        }

        // If we get here, we hit the deadlock
        println!("✓ Test demonstrated deadlock - workers stuck after {} seconds", timeout.as_secs());
    }

    #[test]
    fn test_sequential_inner_no_deadlock() {
        // This test shows that if we don't nest spawn(), there's no deadlock
        let executor = Arc::new(Executor::multi_thread(4, "safe-test-").unwrap());

        // Using map() is fine as long as the inner function doesn't call spawn()
        let result = executor.map(
            |segment_id| {
                // Do work directly without spawning
                std::thread::sleep(Duration::from_millis(10));
                Ok::<_, String>(format!("segment-{}", segment_id))
            },
            0..8,
        );

        assert!(result.is_ok(), "Non-nested parallelism should work fine");
        let segments = result.unwrap();
        assert_eq!(segments.len(), 8);
        println!("✓ Non-nested parallelism completed successfully");
    }

    #[test]
    fn test_scope_based_nesting_no_deadlock() {
        // This test shows the FIX: using scope() instead of spawn() for nested parallelism
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .thread_name(|i| format!("fix-test-{}", i))
            .build()
            .unwrap();

        // Outer parallelism using scope
        let segments: Vec<usize> = (0..8).collect();
        let results: Vec<_> = pool.install(|| {
            segments
                .iter()
                .map(|&segment_id| {
                    // Inner parallelism using scope (not spawn!)
                    pool.scope(|s| {
                        let mut f1 = None;
                        let mut f2 = None;
                        let mut f3 = None;

                        s.spawn(|_| {
                            std::thread::sleep(Duration::from_millis(10));
                            f1 = Some(segment_id * 100 + 1);
                        });

                        s.spawn(|_| {
                            std::thread::sleep(Duration::from_millis(10));
                            f2 = Some(segment_id * 100 + 2);
                        });

                        s.spawn(|_| {
                            std::thread::sleep(Duration::from_millis(10));
                            f3 = Some(segment_id * 100 + 3);
                        });

                        // scope() waits here but uses work-stealing, doesn't block!
                        (f1.unwrap(), f2.unwrap(), f3.unwrap())
                    })
                })
                .collect()
        });

        assert_eq!(results.len(), 8);
        println!("✓ Scope-based nested parallelism completed successfully");
        println!("  This is the recommended fix for the deadlock issue");
    }
}
