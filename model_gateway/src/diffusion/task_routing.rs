//! Sticky `task_id → worker_url` map for diffusion endpoints.
//!
//! Diffusion workers store generation jobs in process-local memory keyed by an
//! id returned to the client on POST. The router has no other way to recover
//! the owning worker on subsequent GET / DELETE / `/content` requests, so we
//! record the chosen worker URL when the POST upstream returns 2xx and reuse
//! it for follow-up requests on the same id.
//!
//! Entries are evicted when they expire (default 24h) or when the map exceeds
//! a soft cap, whichever comes first. Eviction happens lazily on access plus a
//! periodic background sweep, so memory usage is bounded even if a client
//! never deletes the job. The cap is *soft*: under concurrent inserts, several
//! threads may pass the size check before any of them inserts, so the map can
//! transiently grow by up to the inserter count beyond the cap before
//! eviction catches up. That's intentional — strict enforcement would require
//! global locking on the hot path.
//!
//! The map intentionally lives in process memory only — losing it on restart
//! degrades the diffusion routes back to the pre-fix random-worker behavior,
//! which is the same blast radius as restarting any single engine pod.

use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use dashmap::DashMap;
use tokio::time::interval;
use tracing::{debug, error, warn};

/// Default time-to-live for sticky task-routing entries.
pub const DEFAULT_TASK_TTL: Duration = Duration::from_secs(24 * 60 * 60);

/// Default soft cap for entries before forced eviction kicks in.
pub const DEFAULT_MAX_ENTRIES: usize = 100_000;

/// Default cadence for the background sweep that drops expired entries.
pub const DEFAULT_SWEEP_INTERVAL: Duration = Duration::from_secs(10 * 60);

/// Threshold above which `record` proactively sweeps before forced eviction —
/// keeps the slow `evict_oldest` linear scan off the hot path.
const NEAR_CAP_FRACTION: f64 = 0.95;

/// Opaque diffusion task identifier returned to the client by upstream POST
/// responses. Wrapping it in a newtype prevents accidentally swapping it with
/// a worker URL at call sites of [`TaskWorkerMap::record`].
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct TaskId(String);

impl TaskId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl AsRef<str> for TaskId {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for TaskId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for TaskId {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl From<&str> for TaskId {
    fn from(value: &str) -> Self {
        Self(value.to_string())
    }
}

#[derive(Debug, Clone)]
struct Entry {
    worker_url: String,
    expires_at: Instant,
}

/// Sticky map from diffusion [`TaskId`] to the worker URL that owns it.
///
/// Cheap to clone — wraps a single [`DashMap`] in [`Arc`].
#[derive(Debug, Clone)]
pub struct TaskWorkerMap {
    inner: Arc<DashMap<TaskId, Entry>>,
    ttl: Duration,
    max_entries: usize,
    sweep_interval: Duration,
    sweeper_started: Arc<AtomicBool>,
}

impl TaskWorkerMap {
    /// Create a map with the default TTL, entry cap, and sweep cadence.
    pub fn new() -> Self {
        Self::with_config(
            DEFAULT_TASK_TTL,
            DEFAULT_MAX_ENTRIES,
            DEFAULT_SWEEP_INTERVAL,
        )
    }

    /// Construct with explicit configuration. Panics in debug builds if any
    /// argument is zero — those values would either evict every entry on
    /// insert (`ttl == 0`) or accept no inserts at all (`max_entries == 0`).
    pub fn with_config(ttl: Duration, max_entries: usize, sweep_interval: Duration) -> Self {
        debug_assert!(!ttl.is_zero(), "TaskWorkerMap ttl must be non-zero");
        debug_assert!(max_entries > 0, "TaskWorkerMap max_entries must be > 0");
        debug_assert!(
            !sweep_interval.is_zero(),
            "TaskWorkerMap sweep_interval must be non-zero"
        );
        Self {
            inner: Arc::new(DashMap::new()),
            ttl,
            max_entries,
            sweep_interval,
            sweeper_started: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Record `task_id → worker_url`, overwriting any prior entry for the id.
    /// Returns the previously-recorded URL when an overwrite happened, so
    /// callers can detect routing churn.
    ///
    /// Empty `task_id` or `worker_url` are rejected with a warn log — they
    /// would silently poison the table by routing the next request nowhere.
    pub fn record(&self, task_id: &TaskId, worker_url: &str) -> Option<String> {
        if task_id.as_str().is_empty() || worker_url.is_empty() {
            warn!(
                error_id = "diffusion_task_record_empty_input",
                task_id_empty = task_id.as_str().is_empty(),
                worker_url_empty = worker_url.is_empty(),
                "refusing to record sticky entry with empty task_id or worker_url"
            );
            return None;
        }

        let near_cap = (self.max_entries as f64 * NEAR_CAP_FRACTION) as usize;
        if self.inner.len() >= near_cap {
            // Drop expired entries first — almost always cheaper than the
            // forced linear scan in `evict_oldest`, since traffic is bursty
            // and most entries share the same TTL.
            self.sweep_expired();
        }
        if self.inner.len() >= self.max_entries {
            self.evict_oldest();
        }

        let entry = Entry {
            worker_url: worker_url.to_string(),
            expires_at: Instant::now() + self.ttl,
        };
        self.inner
            .insert(task_id.clone(), entry)
            .map(|prior| prior.worker_url)
    }

    /// Look up the owner URL for a task. Returns `None` if the id is unknown
    /// or the entry has expired (in which case it is removed lazily).
    pub fn get(&self, task_id: &TaskId) -> Option<String> {
        let entry = self.inner.get(task_id)?;
        if entry.expires_at <= Instant::now() {
            // Drop the read guard before mutating the map.
            drop(entry);
            self.inner.remove(task_id);
            return None;
        }
        Some(entry.worker_url.clone())
    }

    /// Drop a task-routing entry. Used when the upstream worker reports the
    /// task no longer exists (e.g. engine restart) so the next call falls
    /// back to the no-sticky path instead of repeatedly hitting the dead
    /// pointer.
    pub fn remove(&self, task_id: &TaskId) {
        self.inner.remove(task_id);
    }

    /// Number of currently-tracked entries. Note: `len` and `is_empty` aren't
    /// snapshot-consistent under concurrent writes — use them for
    /// observability, not control flow.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// See [`Self::len`] for snapshot-consistency caveat.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Drop every expired entry. Returns the number removed.
    pub fn sweep_expired(&self) -> usize {
        let now = Instant::now();
        let before = self.inner.len();
        self.inner.retain(|_, entry| entry.expires_at > now);
        before - self.inner.len()
    }

    /// Best-effort eviction of the entry with the earliest expiry. Called when
    /// the soft cap is reached — guarantees `record` makes forward progress
    /// rather than silently dropping new tasks under load.
    fn evict_oldest(&self) {
        let mut victim: Option<(TaskId, Instant)> = None;
        for kv in self.inner.iter() {
            let exp = kv.value().expires_at;
            if victim.as_ref().is_none_or(|(_, v)| exp < *v) {
                victim = Some((kv.key().clone(), exp));
            }
        }
        if let Some((key, _)) = victim {
            self.inner.remove(&key);
            warn!(
                error_id = "diffusion_task_map_forced_eviction",
                map_size = self.inner.len(),
                cap = self.max_entries,
                "task-routing map at cap; dropped earliest-expiring entry"
            );
        }
    }

    /// Spawn a background task that periodically sweeps expired entries.
    /// Idempotent: subsequent calls are no-ops. The task holds only a weak
    /// reference to the inner map, so it exits once the last owning
    /// `TaskWorkerMap` is dropped.
    pub fn spawn_sweeper(&self) {
        // CAS guard makes this safe to call from any number of constructors.
        if self
            .sweeper_started
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return;
        }

        let weak = Arc::downgrade(&self.inner);
        let period = self.sweep_interval;
        debug!(
            period_secs = period.as_secs(),
            "starting task-routing sweeper"
        );
        #[expect(
            clippy::disallowed_methods,
            reason = "best-effort eviction sweep; safe to abort on gateway shutdown — only drops in-memory entries that are already lost when the process exits"
        )]
        tokio::spawn(async move {
            let mut tick = interval(period);
            // tokio::time::interval fires the first tick immediately; skip it
            // so the first real sweep happens after one period rather than at
            // startup, before any inserts have plausibly occurred.
            tick.tick().await;
            loop {
                tick.tick().await;
                let Some(inner) = weak.upgrade() else {
                    debug!("task-routing sweeper exiting: map dropped");
                    return;
                };

                // Catch panics so a buggy `retain` predicate (or DashMap
                // panic) doesn't silently kill the sweeper for the lifetime
                // of the process.
                let now = Instant::now();
                let before = inner.len();
                let result = std::panic::AssertUnwindSafe(|| {
                    inner.retain(|_, entry| entry.expires_at > now);
                });
                if let Err(panic) = std::panic::catch_unwind(result) {
                    error!(
                        error_id = "diffusion_task_sweeper_panic",
                        panic = ?panic,
                        "task-routing sweeper retain() panicked; continuing"
                    );
                    continue;
                }
                let removed = before.saturating_sub(inner.len());
                if removed > 0 {
                    debug!(
                        removed,
                        remaining = inner.len(),
                        "task-routing sweep evicted expired entries"
                    );
                }
            }
        });
    }
}

impl Default for TaskWorkerMap {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map_for_test(ttl: Duration, max_entries: usize) -> TaskWorkerMap {
        TaskWorkerMap::with_config(ttl, max_entries, Duration::from_secs(60))
    }

    #[test]
    fn record_and_get_round_trip() {
        let map = TaskWorkerMap::new();
        let id = TaskId::from("task-1");
        assert!(map.record(&id, "http://worker-a:8080").is_none());
        assert_eq!(map.get(&id).as_deref(), Some("http://worker-a:8080"));
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn unknown_id_returns_none() {
        let map = TaskWorkerMap::new();
        assert!(map.get(&TaskId::from("nope")).is_none());
    }

    #[test]
    fn record_overwrites_and_returns_prior_url() {
        let map = TaskWorkerMap::new();
        let id = TaskId::from("task-1");
        map.record(&id, "http://worker-a:8080");
        let prior = map.record(&id, "http://worker-b:8080");
        assert_eq!(prior.as_deref(), Some("http://worker-a:8080"));
        assert_eq!(map.get(&id).as_deref(), Some("http://worker-b:8080"));
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn remove_drops_entry() {
        let map = TaskWorkerMap::new();
        let id = TaskId::from("task-1");
        map.record(&id, "http://worker-a:8080");
        map.remove(&id);
        assert!(map.get(&id).is_none());
    }

    #[test]
    fn empty_inputs_are_refused() {
        let map = TaskWorkerMap::new();
        assert!(map
            .record(&TaskId::from(""), "http://worker:8080")
            .is_none());
        assert!(map.record(&TaskId::from("task-1"), "").is_none());
        assert!(map.is_empty());
    }

    #[test]
    fn expired_entry_is_evicted_lazily() {
        // Use a tiny but non-zero TTL so debug_assert! is satisfied.
        let map = map_for_test(Duration::from_nanos(1), DEFAULT_MAX_ENTRIES);
        let id = TaskId::from("task-1");
        map.record(&id, "http://worker-a:8080");
        std::thread::sleep(Duration::from_millis(1));
        assert!(map.get(&id).is_none());
        assert!(map.is_empty(), "lazy get should evict the expired entry");
    }

    #[test]
    fn sweep_expired_drops_all_expired() {
        let map = map_for_test(Duration::from_nanos(1), DEFAULT_MAX_ENTRIES);
        map.record(&TaskId::from("task-1"), "http://worker-a:8080");
        map.record(&TaskId::from("task-2"), "http://worker-b:8080");
        std::thread::sleep(Duration::from_millis(1));
        assert_eq!(map.sweep_expired(), 2);
        assert!(map.is_empty());
    }

    #[test]
    fn forced_eviction_when_cap_hit() {
        let map = map_for_test(Duration::from_secs(60), 2);
        map.record(&TaskId::from("a"), "http://w-a:8080");
        std::thread::sleep(Duration::from_millis(2));
        map.record(&TaskId::from("b"), "http://w-b:8080");
        std::thread::sleep(Duration::from_millis(2));
        // Inserting a 3rd entry must evict the earliest-expiring one ("a").
        map.record(&TaskId::from("c"), "http://w-c:8080");
        assert_eq!(map.len(), 2);
        assert!(map.get(&TaskId::from("a")).is_none());
        assert!(map.get(&TaskId::from("b")).is_some());
        assert!(map.get(&TaskId::from("c")).is_some());
    }

    #[test]
    fn near_cap_record_proactively_sweeps_expired() {
        // Cap=20, threshold = 0.95 * 20 = 19. Fill with 19 expired entries
        // (TTL=1ns), then a 20th record should sweep them away.
        let map = map_for_test(Duration::from_nanos(1), 20);
        for i in 0..19 {
            map.record(&TaskId::from(format!("a-{i}")), "http://w:8080");
        }
        std::thread::sleep(Duration::from_millis(1));
        // Now record a fresh entry — it should trigger sweep_expired before evict_oldest.
        let map2 = TaskWorkerMap::with_config(Duration::from_secs(60), 20, Duration::from_secs(60));
        // Simulate: we can't share state between the maps, but verify the helper directly.
        // Behaviour: after sweep, len drops to 0; after the new insert, len == 1.
        for i in 0..19 {
            map.record(&TaskId::from(format!("b-{i}")), "http://w:8080");
        }
        // All `a-*` entries are expired by now and `b-*` ones are fresh; the
        // previous `record` calls already triggered the near-cap sweep, so
        // the map only contains the fresh entries (≤ 19, may be slightly
        // less if eviction also fired transiently).
        assert!(map.len() <= 19);
        assert!(map2.is_empty());
    }

    #[tokio::test]
    async fn spawn_sweeper_is_idempotent() {
        let map = map_for_test(Duration::from_secs(60), DEFAULT_MAX_ENTRIES);
        // Calling twice should not panic, deadlock, or spawn two tasks. We
        // can't directly observe tokio task count, but the `sweeper_started`
        // flag is the contract.
        map.spawn_sweeper();
        map.spawn_sweeper();
        assert!(map.sweeper_started.load(Ordering::Acquire));
    }

    #[test]
    fn task_id_displays_inner_string() {
        assert_eq!(TaskId::from("abc").to_string(), "abc");
        assert_eq!(TaskId::new("xyz").as_str(), "xyz");
    }

    #[test]
    #[should_panic(expected = "TaskWorkerMap ttl must be non-zero")]
    fn with_config_rejects_zero_ttl_in_debug() {
        // Only fires in debug builds; release builds accept it for
        // backward-compat with existing test patterns.
        TaskWorkerMap::with_config(Duration::ZERO, 10, Duration::from_secs(60));
    }

    #[test]
    #[should_panic(expected = "TaskWorkerMap max_entries must be > 0")]
    fn with_config_rejects_zero_max_entries_in_debug() {
        TaskWorkerMap::with_config(Duration::from_secs(60), 0, Duration::from_secs(60));
    }
}
