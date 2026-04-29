//! Integration tests for the diffusion sticky-routing flow.
//!
//! These tests exercise the full axum stack — a `POST /v1/videos` returns a
//! task id minted by *some* mock worker, and subsequent
//! `GET /v1/videos/{id}` calls must keep hitting that same worker even though
//! the gateway's default policy is round-robin.
//!
//! The mock worker exposes the diffusion endpoints in `mock_worker.rs`; each
//! worker keeps a per-process `diffusion_jobs` set so a follow-up against the
//! "wrong" worker returns 404, mirroring the real engine behaviour. The tests
//! observe routing via the `x-worker-id` response header that every diffusion
//! handler emits.

use std::sync::atomic::Ordering;

use axum::{
    body::{to_bytes, Body},
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
};
use serde_json::{json, Value};
use tower::ServiceExt;

use crate::common::{AppTestContext, TestRouterConfig, TestWorkerConfig};

const MAX_RESP_BODY: usize = 1024 * 1024;

#[cfg(test)]
mod diffusion_sticky_routing_tests {
    use super::*;

    /// Helper: send a POST /v1/videos with the given JSON body and return
    /// (status, x-worker-id, parsed body).
    async fn post_video(app: &axum::Router, body: Value) -> (StatusCode, Option<String>, Value) {
        let req = Request::builder()
            .method("POST")
            .uri("/v1/videos")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&body).unwrap()))
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        let status = resp.status();
        let worker_id = resp
            .headers()
            .get("x-worker-id")
            .and_then(|v| v.to_str().ok())
            .map(str::to_string);
        let bytes = to_bytes(resp.into_body(), MAX_RESP_BODY).await.unwrap();
        let parsed: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
        (status, worker_id, parsed)
    }

    /// Helper: send a GET /v1/videos/{id} and return (status, x-worker-id).
    async fn get_video(app: &axum::Router, video_id: &str) -> (StatusCode, Option<String>) {
        let req = Request::builder()
            .method("GET")
            .uri(format!("/v1/videos/{video_id}"))
            .body(Body::empty())
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        let status = resp.status();
        let worker_id = resp
            .headers()
            .get("x-worker-id")
            .and_then(|v| v.to_str().ok())
            .map(str::to_string);
        (status, worker_id)
    }

    async fn delete_video(app: &axum::Router, video_id: &str) -> (StatusCode, Option<String>) {
        let req = Request::builder()
            .method("DELETE")
            .uri(format!("/v1/videos/{video_id}"))
            .body(Body::empty())
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        let status = resp.status();
        let worker_id = resp
            .headers()
            .get("x-worker-id")
            .and_then(|v| v.to_str().ok())
            .map(str::to_string);
        (status, worker_id)
    }

    /// Round-trip: POST creates a task on one worker; every subsequent GET
    /// for that task id must reach the same worker, even though the
    /// round-robin policy would otherwise spread the requests.
    #[tokio::test]
    async fn post_then_get_pins_to_same_worker() {
        let config = TestRouterConfig::round_robin(3851);
        let workers = TestWorkerConfig::healthy_workers(19951, 4);
        let ctx = AppTestContext::new_with_config(config, workers).await;
        let app = ctx.create_app();

        // POST a video — pick whatever worker round-robin lands on.
        let (status, post_worker, body) =
            post_video(&app, json!({ "model": "mock-model", "prompt": "hi" })).await;
        assert_eq!(status, StatusCode::OK, "POST should succeed: body={body:?}");
        let post_worker = post_worker.expect("POST response missing x-worker-id");
        let task_id = body
            .get("id")
            .and_then(Value::as_str)
            .expect("POST body missing id")
            .to_string();

        // 10 sequential GETs must all reach the same worker that handled POST.
        for attempt in 0..10 {
            let (status, get_worker) = get_video(&app, &task_id).await;
            assert_eq!(
                status,
                StatusCode::OK,
                "GET attempt #{attempt} should be 200, hit worker={get_worker:?}"
            );
            assert_eq!(
                get_worker.as_deref(),
                Some(post_worker.as_str()),
                "GET attempt #{attempt} drifted from owning worker"
            );
        }

        ctx.shutdown().await;
    }

    /// Engine restart: when the owning worker loses the task (we simulate by
    /// flipping its `diffusion_force_404` flag), the gateway should evict the
    /// sticky entry on the first 404 and the *next* GET should round-robin.
    /// At least one of the other workers will report 404 too — but the key
    /// invariant is that the gateway no longer keeps aiming at the dead
    /// worker after the eviction.
    #[tokio::test]
    async fn upstream_404_evicts_sticky_entry() {
        let config = TestRouterConfig::round_robin(3852);
        let workers = TestWorkerConfig::healthy_workers(19961, 3);
        let ctx = AppTestContext::new_with_config(config, workers).await;
        let app = ctx.create_app();

        let (status, post_worker, body) =
            post_video(&app, json!({ "model": "mock-model", "prompt": "hi" })).await;
        assert_eq!(status, StatusCode::OK);
        let post_worker = post_worker.unwrap();
        let task_id = body["id"].as_str().unwrap().to_string();

        // Confirm sticky pinning works first.
        let (s, w) = get_video(&app, &task_id).await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(w.as_deref(), Some(post_worker.as_str()));

        // Simulate engine restart: flip force_404 on the owning worker.
        for worker in &ctx.workers {
            let cfg = worker.config_snapshot().await;
            if format!("worker-{}", cfg.port) == post_worker {
                cfg.diffusion_force_404.store(true, Ordering::Relaxed);
            }
        }

        // First GET after restart returns 404 (owning worker has lost the task).
        let (s1, _) = get_video(&app, &task_id).await;
        assert_eq!(
            s1,
            StatusCode::NOT_FOUND,
            "first GET after engine restart should return 404"
        );

        // Now the sticky entry should be evicted. A subsequent GET goes via
        // round-robin; it may still 404 (no other worker owns the task) but
        // it must NOT be pinned to the dead worker — we verify by checking
        // that across multiple GETs we see a different x-worker-id at least
        // once. Without eviction, every GET would keep hitting `post_worker`.
        let mut seen_other_worker = false;
        for _ in 0..10 {
            let (_, w) = get_video(&app, &task_id).await;
            if let Some(w) = w {
                if w != post_worker {
                    seen_other_worker = true;
                    break;
                }
            }
        }
        assert!(
            seen_other_worker,
            "after eviction, GETs should fan out across workers; \
             still pinned to {post_worker}"
        );

        ctx.shutdown().await;
    }

    /// POST without `model` in body or query is rejected with 400 *before*
    /// any worker is picked — so no sticky entry is recorded and the user
    /// gets a clear "model required" error instead of silent random routing.
    #[tokio::test]
    async fn post_without_model_returns_400_and_does_not_pin() {
        let config = TestRouterConfig::round_robin(3853);
        let workers = TestWorkerConfig::healthy_workers(19971, 2);
        let ctx = AppTestContext::new_with_config(config, workers).await;
        let app = ctx.create_app();

        // No `model` field, no `?model=` query.
        let req = Request::builder()
            .method("POST")
            .uri("/v1/videos")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(r#"{"prompt":"hi"}"#))
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

        // Verify no sticky entry was created — every worker's diffusion_jobs
        // set should still be empty.
        let mut total = 0;
        for worker in &ctx.workers {
            let cfg = worker.config_snapshot().await;
            total += cfg.diffusion_jobs.read().await.len();
        }
        assert_eq!(total, 0, "no worker should have recorded a job");

        ctx.shutdown().await;
    }

    /// Successful DELETE clears the sticky entry — a follow-up GET on the
    /// same id must NOT be pinned to the worker that previously owned it.
    #[tokio::test]
    async fn successful_delete_clears_sticky_entry() {
        let config = TestRouterConfig::round_robin(3854);
        let workers = TestWorkerConfig::healthy_workers(19981, 3);
        let ctx = AppTestContext::new_with_config(config, workers).await;
        let app = ctx.create_app();

        let (_, post_worker, body) =
            post_video(&app, json!({ "model": "mock-model", "prompt": "hi" })).await;
        let post_worker = post_worker.unwrap();
        let task_id = body["id"].as_str().unwrap().to_string();

        // DELETE should reach the owning worker and succeed.
        let (status, delete_worker) = delete_video(&app, &task_id).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(delete_worker.as_deref(), Some(post_worker.as_str()));

        // After DELETE the sticky entry is gone. Subsequent GETs round-robin
        // — assert via "we see worker IDs other than post_worker at least
        // once across many attempts".
        let mut seen_other_worker = false;
        for _ in 0..10 {
            let (_, w) = get_video(&app, &task_id).await;
            if let Some(w) = w {
                if w != post_worker {
                    seen_other_worker = true;
                    break;
                }
            }
        }
        assert!(
            seen_other_worker,
            "after DELETE, sticky entry should be gone; \
             still pinned to {post_worker}"
        );

        ctx.shutdown().await;
    }

    /// `?model=...` in the URL is the documented playground request shape.
    /// Verify it routes correctly even when the body is absent or has no
    /// `model` field — the query-string fallback should kick in.
    #[tokio::test]
    async fn model_in_query_string_is_accepted() {
        let config = TestRouterConfig::round_robin(3855);
        let workers = TestWorkerConfig::healthy_workers(19991, 2);
        let ctx = AppTestContext::new_with_config(config, workers).await;
        let app = ctx.create_app();

        // POST with model only in the query string.
        let req = Request::builder()
            .method("POST")
            .uri("/v1/videos?model=mock-model")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(r#"{"prompt":"hi"}"#))
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "?model= fallback should let POST succeed"
        );

        ctx.shutdown().await;
    }
}
