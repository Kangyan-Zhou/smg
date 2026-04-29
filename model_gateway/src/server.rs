use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};

use axum::{
    body::Body,
    extract::{multipart::MultipartError, Extension, Multipart, Path, Query, Request, State},
    http::{self, header::InvalidHeaderName, HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{delete, get, post},
    Json, Router,
};
use llm_tokenizer::TokenizerRegistry;
use openai_protocol::{
    chat::ChatCompletionRequest,
    classify::ClassifyRequest,
    completion::CompletionRequest,
    embedding::EmbeddingRequest,
    generate::GenerateRequest,
    interactions::InteractionsRequest,
    messages::CreateMessageRequest,
    parser::{ParseFunctionCallRequest, SeparateReasoningRequest},
    realtime_session::{
        RealtimeClientSecretCreateRequest, RealtimeSessionCreateRequest,
        RealtimeTranscriptionSessionCreateRequest,
    },
    rerank::{RerankRequest, V1RerankReqInput},
    responses::ResponsesRequest,
    skills::{
        SkillGetQuery, SkillPatchRequest, SkillVersionPatchRequest, SkillVersionsListQuery,
        SkillsListQuery,
    },
    tokenize::{AddTokenizerRequest, DetokenizeRequest, TokenizeRequest},
    transcription::TranscriptionRequest,
    validated::ValidatedJson,
    worker::{WorkerSpec, WorkerUpdateRequest},
};
use rustls::crypto::ring;
use serde::Deserialize;
use serde_json::{json, Value};
use smg_mesh::{MeshServerBuilder, MeshServerConfig, MeshServerHandler, WorkerStateSubscriber};
use tokio::{signal, spawn, sync::mpsc};
use tracing::{debug, error, info, warn, Level};
use wfaas::LoggingSubscriber;

use crate::{
    app_context::AppContext,
    config::{RouterConfig, RoutingMode},
    diffusion::TaskId,
    middleware::{self, AuthConfig, QueuedRequest},
    observability::{
        logging::{self, LoggingConfig},
        metrics::{self, PrometheusConfig},
        metrics_server,
        metrics_ws::{collectors, registry::WatchRegistry},
        otel_trace,
    },
    routers::{
        common::multipart::extract_model_from_multipart,
        conversations, error,
        mesh::{
            get_app_config, get_cluster_status, get_global_rate_limit, get_global_rate_limit_stats,
            get_mesh_health, get_policy_state, get_policy_states, get_worker_state,
            get_worker_states, set_global_rate_limit, trigger_graceful_shutdown, update_app_config,
        },
        openai::realtime::ws::RealtimeQueryParams,
        parse, responses as response_handlers,
        router_manager::RouterManager,
        skills, tokenize, AudioFile, RouterTrait,
    },
    service_discovery::{start_service_discovery, ServiceDiscoveryConfig},
    wasm::route::{add_wasm_module, list_wasm_modules, remove_wasm_module},
    worker::{
        manager::{WorkerManager, WorkerManagerConfig},
        worker::WorkerType,
    },
    workflow::{
        job_queue::{JobQueue, JobQueueConfig},
        Job, TokenizerConfigRequest, WorkflowEngines,
    },
};
#[derive(Clone)]
pub struct AppState {
    pub router: Arc<dyn RouterTrait>,
    pub context: Arc<AppContext>,
    pub concurrency_queue_tx: Option<mpsc::Sender<QueuedRequest>>,
    pub router_manager: Option<Arc<RouterManager>>,
    pub mesh_handler: Option<Arc<MeshServerHandler>>,
    /// Sticky `task_id → worker_url` map for diffusion follow-up requests.
    pub diffusion_task_map: crate::diffusion::TaskWorkerMap,
}

async fn parse_function_call(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ParseFunctionCallRequest>,
) -> Response {
    parse::parse_function_call(&state.context, &req).await
}

async fn parse_reasoning(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SeparateReasoningRequest>,
) -> Response {
    parse::parse_reasoning(&state.context, &req).await
}

async fn sink_handler() -> Response {
    StatusCode::NOT_FOUND.into_response()
}

// ---------------------------------------------------------------------------
// Diffusion endpoint handlers — /v1/videos and /v1/images/*
//
// Diffusion requests carry the model in the body (`model` form field for
// multipart, top-level `model` for JSON) or, for backwards compatibility with
// the playground README, in the `?model=` query string. POSTs are routed to a
// worker that hosts the model; for `/v1/videos` we additionally record the
// `task_id → worker_url` mapping so subsequent GET / DELETE / `/content`
// requests on that id reach the worker that actually owns the task.
//
// `/v1/images/*` endpoints are synchronous (no follow-up GET), so we route
// them by model but skip the body-buffering needed to extract a task id —
// that buffering would regress large `b64_json` responses (~3-30 MB).
// ---------------------------------------------------------------------------

/// Hard cap for diffusion request bodies — covers large image/video uploads.
const MAX_DIFFUSION_BODY_BYTES: usize = 500 * 1024 * 1024;

/// Buffer the full request body, enforcing [`MAX_DIFFUSION_BODY_BYTES`].
async fn read_body(req: Request) -> Result<bytes::Bytes, Response> {
    axum::body::to_bytes(req.into_body(), MAX_DIFFUSION_BODY_BYTES)
        .await
        .map_err(|e| {
            (
                StatusCode::BAD_REQUEST,
                format!("Failed to read request body: {e}"),
            )
                .into_response()
        })
}

/// Cap on `/v1/videos` POST response bodies we re-buffer to extract the task
/// id. Real responses are tiny JSON envelopes (a few hundred bytes); 1 MiB is
/// orders of magnitude over the steady state and protects against an
/// unexpected media payload mis-routing through this path.
const MAX_VIDEO_RESPONSE_BYTES: usize = 1024 * 1024;

/// Pull the `model` field from a diffusion payload without re-encoding it.
///
/// Tries the request body first (multipart `model` form field or JSON `model`
/// key), then falls back to `?model=` in the query string for compatibility
/// with the documented playground request shape.
fn extract_model_from_payload(
    body: &bytes::Bytes,
    content_type: &str,
    query: Option<&str>,
) -> Option<String> {
    let from_body = if content_type.contains("multipart/form-data") {
        extract_model_from_multipart(body, content_type)
    } else {
        serde_json::from_slice::<Value>(body)
            .ok()
            .and_then(|v| v.get("model")?.as_str().map(str::to_string))
    };
    from_body.or_else(|| extract_model_from_query(query?))
}

/// Extract the `model` key from a URL-encoded query string. Handles both
/// `+`-as-space and arbitrary `%XX` escapes via the standard parser.
fn extract_model_from_query(query: &str) -> Option<String> {
    url::form_urlencoded::parse(query.as_bytes())
        .find_map(|(k, v)| (k == "model" && !v.is_empty()).then(|| v.into_owned()))
}

/// Parse a response body for `{"id": "..."}`. Used to record sticky task
/// routing for `/v1/videos` POSTs. Returns `None` if the body isn't JSON,
/// lacks a top-level `id`, or `id` isn't a string — in which case we log
/// the anomaly so a misbehaving upstream is visible to operators.
fn parse_response_id(body: &bytes::Bytes) -> Option<String> {
    match serde_json::from_slice::<Value>(body) {
        Ok(v) => match v.get("id") {
            Some(Value::String(s)) if !s.is_empty() => Some(s.clone()),
            _ => {
                warn!(
                    error_id = "diffusion_response_missing_id",
                    body_len = body.len(),
                    "diffusion POST response had no top-level string `id`; sticky routing skipped"
                );
                None
            }
        },
        Err(e) => {
            warn!(
                error_id = "diffusion_response_id_parse_failed",
                body_len = body.len(),
                error = %e,
                "could not parse diffusion response body as JSON; sticky routing skipped"
            );
            None
        }
    }
}

/// Forward a POST that creates a diffusion task and record the resulting
/// `task_id → worker_url` mapping for follow-up GET / DELETE / `/content`
/// requests.
///
/// Failure modes:
/// * `model_id` could not be resolved → `400 diffusion_model_required`. We
///   do not silently round-robin: that's how the bug this PR fixes was
///   created in the first place.
/// * Model resolved but no worker hosts it → `404 model_not_found`.
/// * Picked worker disappears in the TOCTOU window between selection and
///   forward (deregister, drain, health flip) → retry once via the
///   policy-driven path so the user gets a useful response instead of 503.
/// * Upstream returns non-2xx → forward the response unchanged; record nothing.
/// * Upstream returns 2xx but the body exceeds [`MAX_VIDEO_RESPONSE_BYTES`]
///   → log and forward what we have without recording. We never substitute a
///   500 for a successful upstream because that would silently produce
///   duplicate tasks on client retry.
async fn forward_and_record(
    state: &AppState,
    headers: &HeaderMap,
    body: bytes::Bytes,
    route: &str,
    model_id: Option<&str>,
    method: &http::Method,
) -> Response {
    let Some(model) = model_id else {
        return error::bad_request(
            "diffusion_model_required",
            "diffusion endpoints require `model` in the request body or `?model=` query",
        );
    };
    let Some(worker_url) = state
        .router
        .pick_worker_url_for_model(Some(headers), Some(model))
        .await
    else {
        warn!(
            error_id = "diffusion_no_worker_for_model",
            model = model,
            route = route,
            "no healthy worker for diffusion model; cannot record sticky routing"
        );
        return error::model_not_found(model);
    };

    let response = state
        .router
        .route_raw_request_to_worker_url(Some(headers), body.clone(), route, &worker_url, method)
        .await;

    // TOCTOU: the worker might have been deregistered (or gone unavailable)
    // between selection and forward. Detect via the gateway-internal error
    // code and retry once via the policy-driven path before giving up.
    if response.status() == StatusCode::SERVICE_UNAVAILABLE {
        let code = error::extract_error_code_from_response(&response);
        if matches!(code, "worker_not_found" | "worker_unavailable") {
            warn!(
                error_id = "diffusion_post_worker_disappeared",
                worker_url = %worker_url,
                model = model,
                gateway_code = code,
                "picked worker no longer available; retrying with policy-driven selection"
            );
            return state
                .router
                .route_raw_request(Some(headers), body, route, Some(model), method)
                .await;
        }
    }

    if !response.status().is_success() {
        return response;
    }

    // Re-buffer the body so we can both peek for `id` and forward unchanged.
    // On overflow, log and forward what we have without recording — never
    // mask a successful upstream with a synthetic 5xx.
    let (parts, body) = response.into_parts();
    let body_bytes = match axum::body::to_bytes(body, MAX_VIDEO_RESPONSE_BYTES).await {
        Ok(b) => b,
        Err(e) => {
            warn!(
                error_id = "diffusion_response_buffer_failed",
                route = route,
                worker_url = %worker_url,
                limit = MAX_VIDEO_RESPONSE_BYTES,
                error = %e,
                "could not buffer diffusion response; forwarding without recording sticky entry"
            );
            return error::bad_gateway(
                "diffusion_response_buffer_failed",
                "diffusion response could not be buffered to record routing; please retry",
            );
        }
    };
    if let Some(id) = parse_response_id(&body_bytes) {
        let prior = state
            .diffusion_task_map
            .record(&TaskId::from(id), &worker_url);
        if let Some(prior_url) = prior {
            if prior_url != worker_url {
                warn!(
                    error_id = "diffusion_task_routing_churn",
                    prior_worker_url = %prior_url,
                    new_worker_url = %worker_url,
                    "task id was already routed to a different worker; overwriting"
                );
            }
        }
    }
    Response::from_parts(parts, Body::from(body_bytes))
}

/// Route a diffusion POST that has no follow-up endpoints (`/v1/images/*`).
/// Picks a worker by model; does not buffer the response or record anything.
/// We keep this separate from [`forward_and_record`] so synchronous image
/// generations (which can return ~30 MB of base64) aren't double-buffered.
async fn forward_diffusion_post_no_record(
    state: &AppState,
    headers: &HeaderMap,
    body: bytes::Bytes,
    route: &str,
    model_id: Option<&str>,
    method: &http::Method,
) -> Response {
    let Some(model) = model_id else {
        return error::bad_request(
            "diffusion_model_required",
            "diffusion endpoints require `model` in the request body or `?model=` query",
        );
    };
    state
        .router
        .route_raw_request(Some(headers), body, route, Some(model), method)
        .await
}

/// Forward a follow-up request (GET / DELETE / `/content`) for an existing
/// diffusion task, using the sticky map when possible.
///
/// Eviction policy:
/// * Upstream `404` → engine restarted and lost its in-memory store; drop the
///   sticky entry and forward the 404 to the client.
/// * Gateway-internal `503` (`worker_not_found` / `worker_unavailable`) →
///   worker deregistered or unhealthy; drop the entry and fall through to
///   policy-driven selection so the client gets a successful response if any
///   worker happens to know the task.
/// * Other status (5xx upstream, etc.) → return as-is and *keep* the entry.
///   We log a warning so on-call can see when sticky pinning lands on a
///   misbehaving worker; consecutive-failure circuit breaking is a follow-up.
async fn forward_followup(
    state: &AppState,
    headers: &HeaderMap,
    route: &str,
    method: &http::Method,
    task_id: &TaskId,
) -> Response {
    if let Some(worker_url) = state.diffusion_task_map.get(task_id) {
        let response = state
            .router
            .route_raw_request_to_worker_url(
                Some(headers),
                bytes::Bytes::new(),
                route,
                &worker_url,
                method,
            )
            .await;

        let status = response.status();
        if status == StatusCode::NOT_FOUND {
            // Engine restarted; the task is gone from its in-memory store.
            state.diffusion_task_map.remove(task_id);
            return response;
        }
        if status == StatusCode::SERVICE_UNAVAILABLE {
            let code = error::extract_error_code_from_response(&response);
            if matches!(code, "worker_not_found" | "worker_unavailable") {
                // Gateway-internal: the worker URL no longer routes anywhere.
                // Evict and fall through to policy-driven selection.
                warn!(
                    error_id = "diffusion_sticky_worker_gone",
                    task_id = %task_id,
                    worker_url = %worker_url,
                    "sticky worker no longer registered; falling back to policy-driven routing"
                );
                state.diffusion_task_map.remove(task_id);
            } else {
                // Upstream-emitted 503 (queue full, draining, model loading).
                // Likely transient — keep the sticky entry and pass through.
                warn!(
                    error_id = "diffusion_sticky_upstream_busy",
                    task_id = %task_id,
                    worker_url = %worker_url,
                    "sticky worker returned upstream 503; preserving sticky entry"
                );
                return response;
            }
        } else if !status.is_success() {
            warn!(
                error_id = "diffusion_sticky_upstream_5xx",
                task_id = %task_id,
                worker_url = %worker_url,
                status = status.as_u16(),
                "sticky worker returned non-success status; preserving sticky entry"
            );
            return response;
        } else {
            return response;
        }
    }

    state
        .router
        .route_raw_request(Some(headers), bytes::Bytes::new(), route, None, method)
        .await
}

/// POST /v1/videos — create a video generation job (multipart or JSON).
async fn v1_videos_create(State(state): State<Arc<AppState>>, req: Request) -> Response {
    let route = req.uri().path().to_string();
    let query = req.uri().query().map(str::to_string);
    let method = req.method().clone();
    let content_type = req
        .headers()
        .get(http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    let headers = req.headers().clone();

    let body = match read_body(req).await {
        Ok(b) => b,
        Err(e) => return e,
    };

    let model_id = extract_model_from_payload(&body, &content_type, query.as_deref());
    forward_and_record(&state, &headers, body, &route, model_id.as_deref(), &method).await
}

/// GET /v1/videos — list video jobs. The list is per-worker (each engine
/// stores jobs in process-local memory), so this only returns one worker's
/// view in a multi-engine deployment. Fanning out and merging is a planned
/// follow-up; tracked via the diffusion task map docs.
async fn v1_videos_list(State(state): State<Arc<AppState>>, req: Request) -> Response {
    let route = req
        .uri()
        .path_and_query()
        .map(|p| p.as_str())
        .unwrap_or("/v1/videos")
        .to_string();
    let method = req.method().clone();
    let headers = req.headers().clone();
    state
        .router
        .route_raw_request(Some(&headers), bytes::Bytes::new(), &route, None, &method)
        .await
}

/// GET /v1/videos/{id} — poll job status on the worker that owns it.
async fn v1_videos_get(
    State(state): State<Arc<AppState>>,
    Path(video_id): Path<String>,
    req: Request,
) -> Response {
    let route = format!("/v1/videos/{video_id}");
    let method = req.method().clone();
    let headers = req.headers().clone();
    let task_id = TaskId::from(video_id);
    forward_followup(&state, &headers, &route, &method, &task_id).await
}

/// DELETE /v1/videos/{id} — cancel/delete a job on its owning worker.
async fn v1_videos_delete(
    State(state): State<Arc<AppState>>,
    Path(video_id): Path<String>,
    req: Request,
) -> Response {
    let route = format!("/v1/videos/{video_id}");
    let method = req.method().clone();
    let headers = req.headers().clone();
    let task_id = TaskId::from(video_id);
    let response = forward_followup(&state, &headers, &route, &method, &task_id).await;
    // Successful delete on the upstream means we should drop our sticky entry too.
    if response.status().is_success() {
        state.diffusion_task_map.remove(&task_id);
    }
    response
}

/// GET /v1/videos/{id}/content — download completed video from its owning worker.
async fn v1_videos_content(
    State(state): State<Arc<AppState>>,
    Path(video_id): Path<String>,
    req: Request,
) -> Response {
    let route = format!("/v1/videos/{video_id}/content");
    let method = req.method().clone();
    let headers = req.headers().clone();
    let task_id = TaskId::from(video_id);
    forward_followup(&state, &headers, &route, &method, &task_id).await
}

/// POST /v1/images/edits — image editing (multipart/form-data).
/// Synchronous endpoint — no follow-ups, so we don't buffer the response.
async fn v1_images_edits(State(state): State<Arc<AppState>>, req: Request) -> Response {
    let route = "/v1/images/edits".to_string();
    let query = req.uri().query().map(str::to_string);
    let method = req.method().clone();
    let content_type = req
        .headers()
        .get(http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    let headers = req.headers().clone();

    let body = match read_body(req).await {
        Ok(b) => b,
        Err(e) => return e,
    };

    let model_id = extract_model_from_payload(&body, &content_type, query.as_deref());
    forward_diffusion_post_no_record(&state, &headers, body, &route, model_id.as_deref(), &method)
        .await
}

/// POST /v1/images/generations — image generation (JSON body).
/// Synchronous endpoint — no follow-ups, so we don't buffer the response.
async fn v1_images_generations(State(state): State<Arc<AppState>>, req: Request) -> Response {
    let route = "/v1/images/generations".to_string();
    let query = req.uri().query().map(str::to_string);
    let method = req.method().clone();
    let content_type = req
        .headers()
        .get(http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    let headers = req.headers().clone();

    let body = match read_body(req).await {
        Ok(b) => b,
        Err(e) => return e,
    };

    let model_id = extract_model_from_payload(&body, &content_type, query.as_deref());
    forward_diffusion_post_no_record(&state, &headers, body, &route, model_id.as_deref(), &method)
        .await
}

async fn liveness() -> Response {
    (StatusCode::OK, "OK").into_response()
}

async fn readiness(State(state): State<Arc<AppState>>) -> Response {
    let workers = state.context.worker_registry.get_all();
    let healthy_workers: Vec<_> = workers.iter().filter(|w| w.is_healthy()).collect();

    let is_ready = if state.context.router_config.enable_igw {
        !healthy_workers.is_empty()
    } else {
        match &state.context.router_config.mode {
            RoutingMode::PrefillDecode { .. } => {
                let has_prefill = healthy_workers
                    .iter()
                    .any(|w| matches!(w.worker_type(), WorkerType::Prefill));
                let has_decode = healthy_workers
                    .iter()
                    .any(|w| matches!(w.worker_type(), WorkerType::Decode));
                has_prefill && has_decode
            }
            RoutingMode::Regular { .. } => !healthy_workers.is_empty(),
            RoutingMode::OpenAI { .. } => !healthy_workers.is_empty(),
            RoutingMode::Anthropic { .. } => !healthy_workers.is_empty(),
            RoutingMode::Gemini { .. } => !healthy_workers.is_empty(),
        }
    };

    if is_ready {
        (
            StatusCode::OK,
            Json(json!({
                "status": "ready",
                "healthy_workers": healthy_workers.len(),
                "total_workers": workers.len()
            })),
        )
            .into_response()
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({
                "status": "not ready",
                "reason": "insufficient healthy workers"
            })),
        )
            .into_response()
    }
}

async fn health(_state: State<Arc<AppState>>) -> Response {
    liveness().await
}

async fn health_generate(State(state): State<Arc<AppState>>, req: Request) -> Response {
    state.router.health_generate(req).await
}

async fn engine_metrics(State(state): State<Arc<AppState>>) -> Response {
    WorkerManager::get_engine_metrics(&state.context.worker_registry, &state.context.client)
        .await
        .into_response()
}

async fn get_server_info(State(state): State<Arc<AppState>>, req: Request) -> Response {
    state.router.get_server_info(req).await
}

async fn v1_models(State(state): State<Arc<AppState>>, req: Request) -> Response {
    state.router.get_models(req).await
}

async fn get_model_info(State(state): State<Arc<AppState>>, req: Request) -> Response {
    state.router.get_model_info(req).await
}

async fn generate(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Extension(tenant_meta): Extension<middleware::TenantRequestMeta>,
    Json(body): Json<GenerateRequest>,
) -> Response {
    state
        .router
        .route_generate(Some(&headers), &tenant_meta, &body, &body.model)
        .await
}

async fn v1_chat_completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Extension(tenant_meta): Extension<middleware::TenantRequestMeta>,
    ValidatedJson(body): ValidatedJson<ChatCompletionRequest>,
) -> Response {
    state
        .router
        .route_chat(Some(&headers), &tenant_meta, &body, &body.model)
        .await
}

async fn v1_completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Extension(tenant_meta): Extension<middleware::TenantRequestMeta>,
    ValidatedJson(body): ValidatedJson<CompletionRequest>,
) -> Response {
    state
        .router
        .route_completion(Some(&headers), &tenant_meta, &body, &body.model)
        .await
}

async fn rerank(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Extension(tenant_meta): Extension<middleware::TenantRequestMeta>,
    ValidatedJson(body): ValidatedJson<RerankRequest>,
) -> Response {
    state
        .router
        .route_rerank(Some(&headers), &tenant_meta, &body, &body.model)
        .await
}

async fn v1_rerank(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Extension(tenant_meta): Extension<middleware::TenantRequestMeta>,
    Json(body): Json<V1RerankReqInput>,
) -> Response {
    let rerank_body: RerankRequest = body.into();
    state
        .router
        .route_rerank(
            Some(&headers),
            &tenant_meta,
            &rerank_body,
            &rerank_body.model,
        )
        .await
}

async fn v1_responses(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Extension(tenant_meta): Extension<middleware::TenantRequestMeta>,
    ValidatedJson(body): ValidatedJson<ResponsesRequest>,
) -> Response {
    state
        .router
        .route_responses(Some(&headers), &tenant_meta, &body, &body.model)
        .await
}

async fn v1_interactions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Extension(tenant_meta): Extension<middleware::TenantRequestMeta>,
    ValidatedJson(body): ValidatedJson<InteractionsRequest>,
) -> Response {
    let model_id = body.model.as_deref().or(body.agent.as_deref());
    state
        .router
        .route_interactions(Some(&headers), &tenant_meta, &body, model_id)
        .await
}

async fn v1_embeddings(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Extension(tenant_meta): Extension<middleware::TenantRequestMeta>,
    Json(body): Json<EmbeddingRequest>,
) -> Response {
    state
        .router
        .route_embeddings(Some(&headers), &tenant_meta, &body, &body.model)
        .await
}

async fn v1_messages(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Extension(tenant_meta): Extension<middleware::TenantRequestMeta>,
    ValidatedJson(body): ValidatedJson<CreateMessageRequest>,
) -> Response {
    state
        .router
        .route_messages(Some(&headers), &tenant_meta, &body, &body.model)
        .await
}

async fn v1_classify(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Extension(tenant_meta): Extension<middleware::TenantRequestMeta>,
    Json(body): Json<ClassifyRequest>,
) -> Response {
    state
        .router
        .route_classify(Some(&headers), &tenant_meta, &body, &body.model)
        .await
}

async fn v1_audio_transcriptions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Extension(tenant_meta): Extension<middleware::TenantRequestMeta>,
    mut multipart: Multipart,
) -> Response {
    let mut file_bytes: Option<bytes::Bytes> = None;
    let mut file_name: Option<String> = None;
    let mut file_content_type: Option<String> = None;
    let mut req = TranscriptionRequest::default();
    let mut timestamp_granularities: Vec<String> = Vec::new();

    loop {
        let field = match multipart.next_field().await {
            Ok(Some(f)) => f,
            Ok(None) => break,
            Err(e) => {
                return (
                    StatusCode::BAD_REQUEST,
                    format!("Failed to read multipart field: {e}"),
                )
                    .into_response();
            }
        };

        let name = field.name().unwrap_or("").to_string();
        match name.as_str() {
            "file" => {
                file_name = field.file_name().map(str::to_string);
                file_content_type = field.content_type().map(str::to_string);
                match field.bytes().await {
                    Ok(b) => file_bytes = Some(b),
                    Err(e) => {
                        return (
                            StatusCode::BAD_REQUEST,
                            format!("Failed to read audio file bytes: {e}"),
                        )
                            .into_response();
                    }
                }
            }
            "model" => match field.text().await {
                Ok(t) => req.model = t,
                Err(e) => return bad_text_field("model", e),
            },
            "language" => match field.text().await {
                Ok(t) => req.language = Some(t),
                Err(e) => return bad_text_field("language", e),
            },
            "prompt" => match field.text().await {
                Ok(t) => req.prompt = Some(t),
                Err(e) => return bad_text_field("prompt", e),
            },
            "response_format" => match field.text().await {
                Ok(t) => req.response_format = Some(t),
                Err(e) => return bad_text_field("response_format", e),
            },
            "temperature" => match field.text().await {
                Ok(t) => match t.trim().parse::<f32>() {
                    Ok(v) if v.is_finite() && (0.0..=1.0).contains(&v) => {
                        req.temperature = Some(v);
                    }
                    Ok(v) => {
                        return (
                            StatusCode::BAD_REQUEST,
                            format!(
                                "Invalid 'temperature' value: {v} (must be a finite number in [0.0, 1.0])"
                            ),
                        )
                            .into_response();
                    }
                    Err(e) => {
                        return (
                            StatusCode::BAD_REQUEST,
                            format!("Invalid 'temperature' value: {e}"),
                        )
                            .into_response();
                    }
                },
                Err(e) => return bad_text_field("temperature", e),
            },
            "timestamp_granularities" | "timestamp_granularities[]" => match field.text().await {
                Ok(t) => timestamp_granularities.push(t),
                Err(e) => return bad_text_field("timestamp_granularities", e),
            },
            "stream" => match field.text().await {
                Ok(t) => match t.as_str() {
                    "true" | "True" | "TRUE" | "1" => req.stream = Some(true),
                    "false" | "False" | "FALSE" | "0" => req.stream = Some(false),
                    other => {
                        return (
                            StatusCode::BAD_REQUEST,
                            format!("Invalid 'stream' value: '{other}' (expected true/false/1/0)"),
                        )
                            .into_response();
                    }
                },
                Err(e) => return bad_text_field("stream", e),
            },
            _ => {
                // Unknown field; drain to free resources but otherwise ignore.
                let _ = field.bytes().await;
            }
        }
    }

    // Reject blank/whitespace-only `model` before it reaches worker selection.
    if req.model.trim().is_empty() {
        return (StatusCode::BAD_REQUEST, "Missing required 'model' field").into_response();
    }
    req.model = req.model.trim().to_string();
    let bytes = match file_bytes {
        Some(b) if !b.is_empty() => b,
        Some(_) => {
            return (StatusCode::BAD_REQUEST, "Uploaded 'file' part is empty").into_response();
        }
        None => {
            return (StatusCode::BAD_REQUEST, "Missing required 'file' part").into_response();
        }
    };

    if !timestamp_granularities.is_empty() {
        req.timestamp_granularities = Some(timestamp_granularities);
    }

    let audio = AudioFile {
        bytes,
        file_name: file_name.unwrap_or_else(|| "audio".to_string()),
        content_type: file_content_type,
    };

    state
        .router
        .route_audio_transcriptions(Some(&headers), &tenant_meta, &req, audio, &req.model)
        .await
}

fn bad_text_field(field: &str, e: MultipartError) -> Response {
    (
        StatusCode::BAD_REQUEST,
        format!("Failed to read '{field}' field: {e}"),
    )
        .into_response()
}

async fn v1_responses_get(
    State(state): State<Arc<AppState>>,
    Path(response_id): Path<String>,
) -> Response {
    response_handlers::get_response(&state.context.response_storage, &response_id).await
}

async fn v1_responses_cancel(
    State(state): State<Arc<AppState>>,
    Path(response_id): Path<String>,
    headers: HeaderMap,
) -> Response {
    state
        .router
        .cancel_response(Some(&headers), &response_id)
        .await
}

async fn v1_responses_delete(
    State(state): State<Arc<AppState>>,
    Path(response_id): Path<String>,
) -> Response {
    response_handlers::delete_response(&state.context.response_storage, &response_id).await
}

async fn v1_responses_list_input_items(
    State(state): State<Arc<AppState>>,
    Path(response_id): Path<String>,
) -> Response {
    response_handlers::list_response_input_items(&state.context.response_storage, &response_id)
        .await
}

async fn v1_conversations_create(
    State(state): State<Arc<AppState>>,
    Json(body): Json<Value>,
) -> Response {
    conversations::create_conversation(&state.context.conversation_storage, body).await
}

async fn v1_conversations_get(
    State(state): State<Arc<AppState>>,
    Path(conversation_id): Path<String>,
) -> Response {
    conversations::get_conversation(&state.context.conversation_storage, &conversation_id).await
}

async fn v1_conversations_update(
    State(state): State<Arc<AppState>>,
    Path(conversation_id): Path<String>,
    Json(body): Json<Value>,
) -> Response {
    conversations::update_conversation(&state.context.conversation_storage, &conversation_id, body)
        .await
}

async fn v1_conversations_delete(
    State(state): State<Arc<AppState>>,
    Path(conversation_id): Path<String>,
) -> Response {
    conversations::delete_conversation(&state.context.conversation_storage, &conversation_id).await
}

#[derive(Deserialize, Default)]
struct ListItemsQuery {
    limit: Option<usize>,
    order: Option<String>,
    after: Option<String>,
}

async fn v1_conversations_list_items(
    State(state): State<Arc<AppState>>,
    Path(conversation_id): Path<String>,
    Query(ListItemsQuery {
        limit,
        order,
        after,
    }): Query<ListItemsQuery>,
) -> Response {
    conversations::list_conversation_items(
        &state.context.conversation_storage,
        &state.context.conversation_item_storage,
        &conversation_id,
        limit,
        order.as_deref(),
        after.as_deref(),
    )
    .await
}

#[derive(Deserialize, Default)]
struct GetItemQuery {
    /// Additional fields to include in response (not yet implemented)
    include: Option<Vec<String>>,
}

async fn v1_conversations_create_items(
    State(state): State<Arc<AppState>>,
    Path(conversation_id): Path<String>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    let memory_execution_context =
        middleware::build_memory_execution_context(&state.context.router_config, &headers);

    conversations::create_conversation_items_with_headers(
        &state.context.conversation_storage,
        &state.context.conversation_item_storage,
        &conversation_id,
        body,
        memory_execution_context,
    )
    .await
}

async fn v1_conversations_get_item(
    State(state): State<Arc<AppState>>,
    Path((conversation_id, item_id)): Path<(String, String)>,
    Query(query): Query<GetItemQuery>,
) -> Response {
    conversations::get_conversation_item(
        &state.context.conversation_storage,
        &state.context.conversation_item_storage,
        &conversation_id,
        &item_id,
        query.include,
    )
    .await
}

async fn v1_conversations_delete_item(
    State(state): State<Arc<AppState>>,
    Path((conversation_id, item_id)): Path<(String, String)>,
) -> Response {
    conversations::delete_conversation_item(
        &state.context.conversation_storage,
        &state.context.conversation_item_storage,
        &conversation_id,
        &item_id,
    )
    .await
}

async fn v1_realtime_webrtc(
    State(state): State<Arc<AppState>>,
    Query(params): Query<RealtimeQueryParams>,
    req: Request,
) -> Response {
    // Model may come from query param (application/sdp) or session body
    // (multipart/form-data). Let the handler validate per content type.
    let model = params.model.unwrap_or_default();
    state.router.route_realtime_webrtc(req, &model).await
}

async fn v1_realtime_ws(
    State(state): State<Arc<AppState>>,
    Query(params): Query<RealtimeQueryParams>,
    req: Request,
) -> Response {
    let model = match params.model {
        Some(m) if !m.trim().is_empty() => m,
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                "Missing required 'model' query parameter",
            )
                .into_response();
        }
    };
    state.router.route_realtime_ws(req, &model).await
}

async fn v1_realtime_session(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    ValidatedJson(body): ValidatedJson<RealtimeSessionCreateRequest>,
) -> Response {
    state
        .router
        .route_realtime_session(Some(&headers), &body)
        .await
}

async fn v1_realtime_client_secret(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    ValidatedJson(body): ValidatedJson<RealtimeClientSecretCreateRequest>,
) -> Response {
    state
        .router
        .route_realtime_client_secret(Some(&headers), &body)
        .await
}

async fn v1_realtime_transcription_session(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    ValidatedJson(body): ValidatedJson<RealtimeTranscriptionSessionCreateRequest>,
) -> Response {
    state
        .router
        .route_realtime_transcription_session(Some(&headers), &body)
        .await
}

async fn flush_cache(State(state): State<Arc<AppState>>, _req: Request) -> Response {
    WorkerManager::flush_cache_all(&state.context.worker_registry, &state.context.client)
        .await
        .into_response()
}

async fn get_loads(State(state): State<Arc<AppState>>, _req: Request) -> Response {
    WorkerManager::get_all_worker_loads(&state.context.worker_registry, &state.context.client)
        .await
        .into_response()
}

async fn create_worker(
    State(state): State<Arc<AppState>>,
    Json(config): Json<WorkerSpec>,
) -> Response {
    match state.context.worker_service.create_worker(config).await {
        Ok(result) => result.into_response(),
        Err(err) => err.into_response(),
    }
}

async fn list_workers_rest(State(state): State<Arc<AppState>>) -> Response {
    state.context.worker_service.list_workers().into_response()
}

async fn get_worker(
    State(state): State<Arc<AppState>>,
    Path(worker_id_raw): Path<String>,
) -> Response {
    match state.context.worker_service.get_worker(&worker_id_raw) {
        Ok(result) => result.into_response(),
        Err(err) => err.into_response(),
    }
}

async fn delete_worker(
    State(state): State<Arc<AppState>>,
    Path(worker_id_raw): Path<String>,
) -> Response {
    match state
        .context
        .worker_service
        .delete_worker(&worker_id_raw)
        .await
    {
        Ok(result) => result.into_response(),
        Err(err) => err.into_response(),
    }
}

async fn update_worker(
    State(state): State<Arc<AppState>>,
    Path(worker_id_raw): Path<String>,
    Json(update): Json<WorkerUpdateRequest>,
) -> Response {
    match state
        .context
        .worker_service
        .update_worker(&worker_id_raw, update)
        .await
    {
        Ok(result) => result.into_response(),
        Err(err) => err.into_response(),
    }
}

async fn replace_worker(
    State(state): State<Arc<AppState>>,
    Path(worker_id_raw): Path<String>,
    Json(config): Json<WorkerSpec>,
) -> Response {
    match state
        .context
        .worker_service
        .replace_worker(&worker_id_raw, config)
        .await
    {
        Ok(result) => result.into_response(),
        Err(err) => err.into_response(),
    }
}

// ============================================================================
// Tokenize / Detokenize Handlers
// ============================================================================

async fn v1_tokenize(
    State(state): State<Arc<AppState>>,
    Json(request): Json<TokenizeRequest>,
) -> Response {
    tokenize::tokenize(&state.context.tokenizer_registry, request).await
}

async fn v1_detokenize(
    State(state): State<Arc<AppState>>,
    Json(request): Json<DetokenizeRequest>,
) -> Response {
    tokenize::detokenize(&state.context.tokenizer_registry, request).await
}

async fn v1_tokenizers_add(
    State(state): State<Arc<AppState>>,
    Json(request): Json<AddTokenizerRequest>,
) -> Response {
    tokenize::add_tokenizer(&state.context, request).await
}

async fn v1_tokenizers_list(State(state): State<Arc<AppState>>) -> Response {
    tokenize::list_tokenizers(&state.context.tokenizer_registry).await
}

async fn v1_tokenizers_get(
    State(state): State<Arc<AppState>>,
    Path(tokenizer_id): Path<String>,
) -> Response {
    tokenize::get_tokenizer_info(&state.context, &tokenizer_id).await
}

async fn v1_tokenizers_status(
    State(state): State<Arc<AppState>>,
    Path(tokenizer_id): Path<String>,
) -> Response {
    tokenize::get_tokenizer_status(&state.context, &tokenizer_id).await
}

async fn v1_tokenizers_remove(
    State(state): State<Arc<AppState>>,
    Path(tokenizer_id): Path<String>,
) -> Response {
    tokenize::remove_tokenizer(&state.context, &tokenizer_id).await
}

async fn v1_skills_create(State(state): State<Arc<AppState>>, multipart: Multipart) -> Response {
    skills::create_skill(State(state), multipart).await
}

async fn v1_skills_list(
    State(state): State<Arc<AppState>>,
    query: Query<SkillsListQuery>,
    headers: HeaderMap,
) -> Response {
    skills::list_skills(State(state), query, headers).await
}

async fn v1_skills_get(
    State(state): State<Arc<AppState>>,
    Path(skill_id): Path<String>,
    query: Query<SkillGetQuery>,
    headers: HeaderMap,
) -> Response {
    skills::get_skill(State(state), Path(skill_id), query, headers).await
}

async fn v1_skills_patch(
    State(state): State<Arc<AppState>>,
    Path(skill_id): Path<String>,
    query: Query<SkillGetQuery>,
    ValidatedJson(body): ValidatedJson<SkillPatchRequest>,
) -> Response {
    skills::patch_skill(State(state), Path(skill_id), query, Json(body)).await
}

async fn v1_skills_create_version(
    State(state): State<Arc<AppState>>,
    Path(skill_id): Path<String>,
    multipart: Multipart,
) -> Response {
    skills::create_skill_version(State(state), Path(skill_id), multipart).await
}

async fn v1_skills_list_versions(
    State(state): State<Arc<AppState>>,
    Path(skill_id): Path<String>,
    query: Query<SkillVersionsListQuery>,
    headers: HeaderMap,
) -> Response {
    skills::list_skill_versions(State(state), Path(skill_id), query, headers).await
}

async fn v1_skills_get_version(
    State(state): State<Arc<AppState>>,
    Path((skill_id, version)): Path<(String, String)>,
    query: Query<SkillGetQuery>,
    headers: HeaderMap,
) -> Response {
    skills::get_skill_version(State(state), Path((skill_id, version)), query, headers).await
}

async fn v1_skills_patch_version(
    State(state): State<Arc<AppState>>,
    Path((skill_id, version)): Path<(String, String)>,
    query: Query<SkillGetQuery>,
    ValidatedJson(body): ValidatedJson<SkillVersionPatchRequest>,
) -> Response {
    skills::patch_skill_version(State(state), Path((skill_id, version)), query, Json(body)).await
}

async fn v1_skills_delete(
    State(state): State<Arc<AppState>>,
    Path(skill_id): Path<String>,
    query: Query<SkillGetQuery>,
) -> Response {
    skills::delete_skill(State(state), Path(skill_id), query).await
}

async fn v1_skills_delete_version(
    State(state): State<Arc<AppState>>,
    Path((skill_id, version)): Path<(String, String)>,
    query: Query<SkillGetQuery>,
) -> Response {
    skills::delete_skill_version(State(state), Path((skill_id, version)), query).await
}

pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub router_config: RouterConfig,
    pub max_payload_size: usize,
    pub log_dir: Option<String>,
    pub log_level: Option<String>,
    pub log_json: bool,
    pub service_discovery_config: Option<ServiceDiscoveryConfig>,
    pub prometheus_config: Option<PrometheusConfig>,
    pub request_timeout_secs: u64,
    pub request_id_headers: Option<Vec<String>>,
    pub shutdown_grace_period_secs: u64,
    /// Control plane authentication configuration
    pub control_plane_auth: Option<smg_auth::ControlPlaneAuthConfig>,
    pub mesh_server_config: Option<MeshServerConfig>,
    /// Bind address for WebRTC UDP sockets.
    /// `None` means use the default (0.0.0.0, auto-detect candidate IP).
    pub webrtc_bind_addr: Option<std::net::IpAddr>,
    /// STUN server for ICE candidate gathering (host:port).
    /// `None` means use the default (stun.l.google.com:19302).
    pub webrtc_stun_server: Option<String>,
}

pub fn build_app(
    app_state: Arc<AppState>,
    auth_config: AuthConfig,
    control_plane_auth_state: Option<smg_auth::ControlPlaneAuthState>,
    max_payload_size: usize,
    request_id_headers: Vec<String>,
    cors_allowed_origins: Vec<String>,
) -> Result<Router, InvalidHeaderName> {
    // Pending (upgrade not completed): 30s TTL
    // Disconnected: 60 min TTL
    app_state.context.realtime_registry.start_reaper(
        Duration::from_secs(3600),
        Duration::from_secs(30),
        Duration::from_secs(60),
    );

    let tenant_resolution_state =
        middleware::TenantResolutionState::new(&app_state.context.router_config)?
            .with_tenant_alias_store(
                app_state
                    .context
                    .skill_service
                    .as_ref()
                    .and_then(|skill_service| skill_service.tenant_alias_store()),
            );

    let protected_routes = Router::new()
        .route("/v1/responses", post(v1_responses))
        .route("/v1/responses/{response_id}", get(v1_responses_get))
        .route(
            "/v1/responses/{response_id}/cancel",
            post(v1_responses_cancel),
        )
        .route("/v1/responses/{response_id}", delete(v1_responses_delete))
        .route(
            "/v1/responses/{response_id}/input_items",
            get(v1_responses_list_input_items),
        )
        .route("/v1/conversations", post(v1_conversations_create))
        .route(
            "/v1/conversations/{conversation_id}",
            get(v1_conversations_get)
                .post(v1_conversations_update)
                .delete(v1_conversations_delete),
        )
        .route(
            "/v1/conversations/{conversation_id}/items",
            get(v1_conversations_list_items).post(v1_conversations_create_items),
        )
        .route(
            "/v1/conversations/{conversation_id}/items/{item_id}",
            get(v1_conversations_get_item).delete(v1_conversations_delete_item),
        )
        .route_layer(axum::middleware::from_fn_with_state(
            app_state.clone(),
            middleware::storage_context_middleware,
        ))
        .route("/generate", post(generate))
        .route("/v1/chat/completions", post(v1_chat_completions))
        .route("/v1/completions", post(v1_completions))
        .route("/rerank", post(rerank))
        .route("/v1/rerank", post(v1_rerank))
        .route("/v1/embeddings", post(v1_embeddings))
        .route("/v1/messages", post(v1_messages))
        .route("/v1/interactions", post(v1_interactions))
        .route("/v1/classify", post(v1_classify))
        // Tokenize / Detokenize endpoints
        .route("/v1/tokenize", post(v1_tokenize))
        .route("/v1/detokenize", post(v1_detokenize))
        // Realtime REST endpoints (same middleware as other protected routes)
        .route("/v1/realtime/sessions", post(v1_realtime_session))
        .route(
            "/v1/realtime/client_secrets",
            post(v1_realtime_client_secret),
        )
        .route(
            "/v1/realtime/transcription_sessions",
            post(v1_realtime_transcription_session),
        )
        .route_layer(axum::middleware::from_fn_with_state(
            app_state.clone(),
            middleware::concurrency_limit_middleware,
        ))
        .route_layer(axum::middleware::from_fn_with_state(
            tenant_resolution_state.clone(),
            middleware::route_request_meta_middleware,
        ))
        .route_layer(axum::middleware::from_fn_with_state(
            auth_config.clone(),
            middleware::auth_middleware,
        ))
        .route_layer(axum::middleware::from_fn_with_state(
            app_state.clone(),
            middleware::wasm_middleware,
        ));

    // WebSocket and WebRTC routes: auth + concurrency but NO WASM middleware.
    // WASM OnResponse reconstructs the response from status/headers/body,
    // dropping the response extensions that carry the WebSocket upgrade future.
    let realtime_routes = Router::new()
        .route("/v1/realtime", get(v1_realtime_ws))
        .route("/v1/realtime/calls", post(v1_realtime_webrtc))
        .route_layer(axum::middleware::from_fn_with_state(
            app_state.clone(),
            middleware::concurrency_limit_middleware,
        ))
        .route_layer(axum::middleware::from_fn_with_state(
            tenant_resolution_state.clone(),
            middleware::route_request_meta_middleware,
        ))
        .route_layer(axum::middleware::from_fn_with_state(
            auth_config.clone(),
            middleware::auth_middleware,
        ));

    // Multipart upload routes: auth + concurrency but NO WASM middleware.
    // The WASM OnRequest phase buffers the full body into a `Vec<u8>` subject
    // to the WASM manager's `max_body_size` (10MB default). Audio uploads
    // routinely exceed that, so WASM middleware would reject them with 400
    // before reaching the handler.
    let multipart_upload_routes = Router::new()
        .route("/v1/audio/transcriptions", post(v1_audio_transcriptions))
        // Diffusion endpoints — video generation. POST is multipart/JSON,
        // management verbs are proxied verbatim to the worker. Kept outside
        // the WASM middleware for the same reason as audio: bodies can be
        // larger than the default WASM max_body_size (10 MB).
        .route("/v1/videos", post(v1_videos_create).get(v1_videos_list))
        .route(
            "/v1/videos/{video_id}",
            get(v1_videos_get).delete(v1_videos_delete),
        )
        .route("/v1/videos/{video_id}/content", get(v1_videos_content))
        // Diffusion endpoints — image generation. `/edits` is multipart,
        // `/generations` is JSON but shares the same raw-forwarding path so
        // we keep them together outside WASM.
        .route("/v1/images/generations", post(v1_images_generations))
        .route("/v1/images/edits", post(v1_images_edits))
        .route_layer(axum::middleware::from_fn_with_state(
            app_state.clone(),
            middleware::concurrency_limit_middleware,
        ))
        .route_layer(axum::middleware::from_fn_with_state(
            tenant_resolution_state,
            middleware::route_request_meta_middleware,
        ))
        .route_layer(axum::middleware::from_fn_with_state(
            auth_config.clone(),
            middleware::auth_middleware,
        ));

    let public_routes = Router::new()
        .route("/liveness", get(liveness))
        .route("/readiness", get(readiness))
        .route("/health", get(health))
        .route("/health_generate", get(health_generate))
        .route("/engine_metrics", get(engine_metrics))
        .route("/v1/models", get(v1_models))
        .route("/get_model_info", get(get_model_info))
        .route("/get_server_info", get(get_server_info));

    // Build admin routes with control plane auth if configured, otherwise use simple API key auth
    let mut admin_routes = Router::new()
        .route("/flush_cache", post(flush_cache))
        .route("/get_loads", get(get_loads))
        .route("/parse/function_call", post(parse_function_call))
        .route("/parse/reasoning", post(parse_reasoning))
        .route("/wasm", post(add_wasm_module))
        .route("/wasm/{module_uuid}", delete(remove_wasm_module))
        .route("/wasm", get(list_wasm_modules))
        // Tokenizer management endpoints
        .route(
            "/v1/tokenizers",
            post(v1_tokenizers_add).get(v1_tokenizers_list),
        )
        .route(
            "/v1/tokenizers/{tokenizer_id}",
            get(v1_tokenizers_get).delete(v1_tokenizers_remove),
        )
        .route(
            "/v1/tokenizers/{tokenizer_id}/status",
            get(v1_tokenizers_status),
        );

    if app_state.context.router_config.skills_enabled
        && app_state
            .context
            .router_config
            .skills
            .as_ref()
            .is_some_and(|skills_config| skills_config.admin.enabled)
        && app_state.context.skill_service.is_some()
    {
        admin_routes = admin_routes
            .route("/v1/skills", post(v1_skills_create).get(v1_skills_list))
            .route(
                "/v1/skills/{skill_id}",
                get(v1_skills_get)
                    .patch(v1_skills_patch)
                    .delete(v1_skills_delete),
            )
            .route(
                "/v1/skills/{skill_id}/versions",
                post(v1_skills_create_version).get(v1_skills_list_versions),
            )
            .route(
                "/v1/skills/{skill_id}/versions/{version}",
                get(v1_skills_get_version)
                    .patch(v1_skills_patch_version)
                    .delete(v1_skills_delete_version),
            );
    }

    // Build worker routes
    let worker_routes = Router::new()
        .route("/workers", post(create_worker).get(list_workers_rest))
        .route(
            "/workers/{worker_id}",
            get(get_worker)
                .put(replace_worker)
                .patch(update_worker)
                .delete(delete_worker),
        );

    // Apply authentication middleware to control plane routes
    let apply_control_plane_auth = |routes: Router<Arc<AppState>>| {
        if let Some(ref cp_state) = control_plane_auth_state {
            routes.route_layer(axum::middleware::from_fn_with_state(
                cp_state.clone(),
                smg_auth::control_plane_auth_middleware,
            ))
        } else {
            routes.route_layer(axum::middleware::from_fn_with_state(
                auth_config.clone(),
                middleware::auth_middleware,
            ))
        }
    };
    let admin_routes = apply_control_plane_auth(admin_routes);
    let worker_routes = apply_control_plane_auth(worker_routes);

    // HA management routes
    let mesh_routes = Router::new()
        .route("/ha/status", get(get_cluster_status))
        .route("/ha/health", get(get_mesh_health))
        .route("/ha/workers", get(get_worker_states))
        .route("/ha/workers/{worker_id}", get(get_worker_state))
        .route("/ha/policies", get(get_policy_states))
        .route("/ha/policies/{model_id}", get(get_policy_state))
        .route("/ha/config/{key}", get(get_app_config))
        .route("/ha/config", post(update_app_config))
        .route("/ha/rate-limit", post(set_global_rate_limit))
        .route("/ha/rate-limit", get(get_global_rate_limit))
        .route("/ha/rate-limit/stats", get(get_global_rate_limit_stats))
        .route("/ha/shutdown", post(trigger_graceful_shutdown))
        .route_layer(axum::middleware::from_fn_with_state(
            auth_config.clone(),
            middleware::auth_middleware,
        ));

    Ok(Router::new()
        .merge(protected_routes)
        .merge(realtime_routes)
        .merge(multipart_upload_routes)
        .merge(public_routes)
        .merge(admin_routes)
        .merge(worker_routes)
        .merge(mesh_routes)
        .layer(axum::extract::DefaultBodyLimit::max(max_payload_size))
        .layer(tower_http::limit::RequestBodyLimitLayer::new(
            max_payload_size,
        ))
        .layer(middleware::create_logging_layer())
        .layer(middleware::HttpMetricsLayer::new(
            app_state.context.inflight_tracker.clone(),
        ))
        .layer(middleware::RequestIdLayer::new(request_id_headers))
        .layer(create_cors_layer(cors_allowed_origins))
        .fallback(sink_handler)
        .with_state(app_state))
}

pub async fn startup(config: ServerConfig) -> Result<(), Box<dyn std::error::Error>> {
    static LOGGING_INITIALIZED: AtomicBool = AtomicBool::new(false);

    if let Some(trace_config) = &config.router_config.trace_config {
        otel_trace::otel_tracing_init(
            trace_config.enable_trace,
            Some(&trace_config.otlp_traces_endpoint),
        )?;
    }

    let _log_guard = if LOGGING_INITIALIZED.swap(true, Ordering::SeqCst) {
        None
    } else {
        Some(logging::init_logging(
            LoggingConfig {
                level: config
                    .log_level
                    .as_deref()
                    .and_then(|s| match s.to_uppercase().parse::<Level>() {
                        Ok(l) => Some(l),
                        Err(_) => {
                            warn!("Invalid log level string: '{s}'. Defaulting to INFO.");
                            None
                        }
                    })
                    .unwrap_or(Level::INFO),
                json_format: config.log_json,
                log_dir: config.log_dir.clone(),
                colorize: true,
                log_file_name: "smg".to_string(),
                log_targets: None,
            },
            config.router_config.trace_config.clone(),
        ))
    };

    // Start metrics server and collectors.
    // Metrics server binds the port now; collectors start after AppContext is built.
    let (prometheus_handle, watch_registry) =
        if let Some(prometheus_config) = &config.prometheus_config {
            let handle = metrics::start_prometheus(prometheus_config.clone());
            let registry = Arc::new(WatchRegistry::new());
            let _server_handle = metrics_server::start_metrics_server(
                handle.clone(),
                prometheus_config.host.clone(),
                prometheus_config.port,
                registry.clone(),
                metrics_server::DEFAULT_MAX_WS_CONNECTIONS,
            )
            .await;
            (Some(handle), Some(registry))
        } else {
            (None, None)
        };

    // Initialize mesh server if configured, it will return a handler for mesh management
    let mesh_handler = if let Some(mesh_server_config) = &config.mesh_server_config {
        // Create mesh server builder and build with stores
        let (mesh_server, handler) = MeshServerBuilder::from(mesh_server_config).build();

        // Start rate limit window reset task (managed by handler)
        handler.start_rate_limit_task(1); // Reset every 1 second

        #[expect(
            clippy::disallowed_methods,
            reason = "mesh server runs for the lifetime of the process; shutdown is handled by the mesh handler"
        )]
        spawn(async move {
            if let Err(e) = mesh_server.start().await {
                tracing::error!("Mesh server failed: {}", e);
            }
        });

        Some(Arc::new(handler))
    } else {
        None
    };

    info!(
        "Starting router on {}:{} | mode: {:?} | policy: {:?} | max_payload: {}MB",
        config.host,
        config.port,
        config.router_config.mode,
        config.router_config.policy,
        config.max_payload_size / (1024 * 1024)
    );

    let app_context = Arc::new(
        AppContext::from_config(
            config.router_config.clone(),
            config.request_timeout_secs,
            config.webrtc_bind_addr,
            config.webrtc_stun_server.clone(),
        )
        .await?,
    );

    if config.prometheus_config.is_some() {
        app_context.inflight_tracker.start_sampler(20);
    }

    // Start WS metrics collectors now that AppContext is available.
    let _collector_handles = match (&prometheus_handle, &watch_registry) {
        (Some(handle), Some(registry)) => Some(collectors::start_collectors(
            app_context.clone(),
            registry.clone(),
            collectors::CollectorConfig::default(),
            handle.clone(),
        )),
        _ => None,
    };

    let weak_context = Arc::downgrade(&app_context);
    let worker_job_queue = JobQueue::new(JobQueueConfig::default(), weak_context);
    #[expect(
        clippy::expect_used,
        reason = "OnceLock initialization during startup; double-init is a fatal bug"
    )]
    app_context
        .worker_job_queue
        .set(worker_job_queue)
        .expect("JobQueue should only be initialized once");

    // Initialize typed workflow engines
    let engines = WorkflowEngines::new(&config.router_config);

    // Subscribe logging to all workflow engines
    engines.subscribe_all(Arc::new(LoggingSubscriber)).await;

    #[expect(
        clippy::expect_used,
        reason = "OnceLock initialization during startup; double-init is a fatal bug"
    )]
    app_context
        .workflow_engines
        .set(engines)
        .expect("WorkflowEngines should only be initialized once");
    debug!(
        "Workflow engines initialized (health check timeout: {}s)",
        config.router_config.health_check.timeout_secs
    );

    // Submit startup tokenizer job if tokenizer path is configured
    // This runs before worker initialization to ensure tokenizer is available
    if config.router_config.disable_tokenizer_autoload {
        info!("Tokenizer autoload disabled via config; skipping startup tokenizer load");
    } else if let Some(tokenizer_source) = config
        .router_config
        .tokenizer_path
        .as_ref()
        .or(config.router_config.model_path.as_ref())
    {
        info!("Loading startup tokenizer from: {}", tokenizer_source);

        #[expect(
            clippy::expect_used,
            reason = "JobQueue was just initialized above; absence is unreachable"
        )]
        let job_queue = app_context
            .worker_job_queue
            .get()
            .expect("JobQueue should be initialized");

        let tokenizer_config = TokenizerConfigRequest {
            id: TokenizerRegistry::generate_id(),
            name: tokenizer_source.clone(),
            source: tokenizer_source.clone(),
            chat_template_path: config.router_config.chat_template.clone(),
            cache_config: config.router_config.tokenizer_cache.to_option(),
            fail_on_duplicate: false,
        };

        let job = Job::AddTokenizer {
            config: Box::new(tokenizer_config),
        };

        job_queue
            .submit(job)
            .await
            .map_err(|e| format!("Failed to submit startup tokenizer job: {e}"))?;

        info!("Startup tokenizer job submitted (will complete in background)");
    }

    info!(
        "Initializing workers for routing mode: {:?}",
        config.router_config.mode
    );

    // Submit worker initialization job to queue
    #[expect(
        clippy::expect_used,
        reason = "JobQueue was initialized above; absence is unreachable"
    )]
    let job_queue = app_context
        .worker_job_queue
        .get()
        .expect("JobQueue should be initialized");
    let job = Job::InitializeWorkersFromConfig {
        router_config: Box::new(config.router_config.clone()),
    };
    job_queue
        .submit(job)
        .await
        .map_err(|e| format!("Failed to submit worker initialization job: {e}"))?;

    info!("Worker initialization job submitted (will complete in background)");

    if let Some(mcp_config) = &config.router_config.mcp_config {
        info!("Found {} MCP server(s) in config", mcp_config.servers.len());
        let mcp_job = Job::InitializeMcpServers {
            mcp_config: Box::new(mcp_config.clone()),
        };
        job_queue
            .submit(mcp_job)
            .await
            .map_err(|e| format!("Failed to submit MCP initialization job: {e}"))?;
    } else {
        info!("No MCP config provided, skipping MCP server initialization");
    }

    // Note: MCP orchestrator handles background refresh internally via refresh channel
    // configured by inventory.refresh_interval in mcp.yaml

    let worker_stats = app_context.worker_registry.stats();
    info!(
        "Workers initialized: {} total, {} healthy",
        worker_stats.total_workers, worker_stats.healthy_workers
    );

    let router_manager = RouterManager::from_config(&config, &app_context).await?;
    let router: Arc<dyn RouterTrait> = router_manager.clone();

    // WorkerManager owns the background health check loop. Its handle must
    // outlive the server to keep the task alive — bind it here so its Drop
    // (which aborts the task) runs at server shutdown.
    let _worker_manager = if config.router_config.health_check.disable_health_check {
        info!("Global health checks disabled via CLI/config; skipping WorkerManager");
        None
    } else {
        let manager = WorkerManager::start(
            app_context.worker_registry.clone(),
            WorkerManagerConfig {
                default_check_interval_secs: config.router_config.health_check.check_interval_secs,
                remove_unhealthy: config.router_config.health_check.remove_unhealthy_workers,
            },
            app_context.worker_job_queue.get().cloned(),
        );
        debug!(
            "Started WorkerManager health check loop with {}s default interval",
            config.router_config.health_check.check_interval_secs
        );
        Some(manager)
    };

    // WorkerMonitor subscribes to registry events. Starting its event
    // loop here (after the synchronous worker population in
    // RouterManager::from_config above) means the bootstrap reconcile
    // captures every worker that exists at this point and the event
    // task picks up everything registered afterwards.
    if let Some(ref worker_monitor) = app_context.worker_monitor {
        worker_monitor.start_event_loop();
        debug!("Started WorkerMonitor event loop");
    }

    let (limiter, processor) = middleware::ConcurrencyLimiter::new(
        app_context.rate_limiter.clone(),
        config.router_config.queue_size,
        Duration::from_secs(config.router_config.queue_timeout_secs),
    );

    if app_context.rate_limiter.is_none() {
        info!("Rate limiting is disabled (max_concurrent_requests = -1)");
    }

    match processor {
        Some(proc) => {
            #[expect(
                clippy::disallowed_methods,
                reason = "request queue processor runs for the lifetime of the server"
            )]
            spawn(proc.run());
            debug!(
                "Started request queue (size: {}, timeout: {}s)",
                config.router_config.queue_size, config.router_config.queue_timeout_secs
            );
        }
        None => {
            debug!(
                "Rate limiting enabled (max_concurrent_requests = {}, queue disabled)",
                config.router_config.max_concurrent_requests
            );
        }
    }

    // Set mesh sync manager to worker registry and policy registry if mesh is enabled
    // This allows these components to sync state across mesh nodes when mesh is enabled,
    // but they work independently without mesh when mesh is disabled.
    // Using thread-safe set_mesh_sync method that works with Arc-wrapped registries
    if let Some(ref handle) = mesh_handler {
        app_context
            .worker_registry
            .set_mesh_sync(Some(handle.sync_manager.clone()));
        handle
            .sync_manager
            .register_worker_state_subscriber(app_context.worker_registry.clone());
        // Replay workers already in the CRDT store — they arrived between
        // mesh server start and subscriber registration above.
        for state in handle.sync_manager.get_all_worker_states() {
            app_context.worker_registry.on_remote_worker_state(&state);
        }
        info!("Mesh sync manager set on worker registry");

        handle
            .sync_manager
            .register_tree_state_subscriber(app_context.policy_registry.clone());
        app_context
            .policy_registry
            .set_mesh_sync(Some(handle.sync_manager.clone()));
        info!("Mesh sync manager set on policy registry");
    }

    // Get mesh cluster state and port before moving mesh_handler into app_state
    let mesh_cluster_state = mesh_handler.as_ref().map(|h| h.state.clone());
    let mesh_port = config
        .mesh_server_config
        .as_ref()
        .map(|c| c.advertise_addr.port());

    let diffusion_task_map = crate::diffusion::TaskWorkerMap::new();
    diffusion_task_map.spawn_sweeper();
    let app_state = Arc::new(AppState {
        router,
        context: app_context.clone(),
        concurrency_queue_tx: limiter.queue_tx.clone(),
        router_manager: Some(router_manager),
        mesh_handler,
        diffusion_task_map,
    });
    if let Some(service_discovery_config) = config.service_discovery_config {
        if service_discovery_config.enabled {
            let app_context_arc = Arc::clone(&app_state.context);

            match start_service_discovery(
                service_discovery_config,
                app_context_arc,
                mesh_cluster_state,
                mesh_port,
            )
            .await
            {
                Ok(handle) => {
                    info!("Service discovery started");
                    #[expect(
                        clippy::disallowed_methods,
                        reason = "service discovery runs for the lifetime of the server"
                    )]
                    spawn(async move {
                        if let Err(e) = handle.await {
                            error!("Service discovery task failed: {:?}", e);
                        }
                    });
                }
                Err(e) => {
                    error!("Failed to start service discovery: {e}");
                    warn!("Continuing without service discovery");
                }
            }
        }
    }

    info!(
        "Router ready | workers: {:?}",
        WorkerManager::get_worker_urls(&app_state.context.worker_registry)
    );

    let request_id_headers = config.request_id_headers.clone().unwrap_or_else(|| {
        vec![
            "x-request-id".to_string(),
            "x-correlation-id".to_string(),
            "x-trace-id".to_string(),
            "request-id".to_string(),
        ]
    });

    let auth_config = AuthConfig::new(config.router_config.api_key.clone());

    // Initialize control plane authentication if configured
    let control_plane_auth_state =
        smg_auth::ControlPlaneAuthState::try_init(config.control_plane_auth.as_ref()).await;

    let app = build_app(
        app_state,
        auth_config,
        control_plane_auth_state,
        config.max_payload_size,
        request_id_headers,
        config.router_config.cors_allowed_origins.clone(),
    )?;

    // TcpListener::bind accepts &str and handles IPv4/IPv6 via ToSocketAddrs
    let bind_addr = format!("{}:{}", config.host, config.port);
    info!("Starting server on {}", bind_addr);

    // Parse address and set up graceful shutdown (common to both TLS and non-TLS)
    let addr: std::net::SocketAddr = bind_addr
        .parse()
        .map_err(|e| format!("Invalid address: {e}"))?;

    let handle = axum_server::Handle::new();
    let handle_clone = handle.clone();
    let inflight_tracker = app_context.inflight_tracker.clone();
    let drain_timeout = Duration::from_secs(config.shutdown_grace_period_secs);
    #[expect(
        clippy::disallowed_methods,
        reason = "shutdown signal handler must outlive the server to trigger graceful shutdown"
    )]
    spawn(async move {
        shutdown_signal().await;

        // Phase 1: Gate — stop accepting new connections, mark as draining
        info!(
            in_flight = inflight_tracker.len(),
            "Beginning graceful shutdown: gating new connections"
        );
        inflight_tracker.begin_drain();
        handle_clone.graceful_shutdown(Some(drain_timeout));

        // Phase 2: Drain — wait for in-flight requests to complete
        // Re-check after gating to catch requests that arrived between the
        // snapshot and graceful_shutdown stopping the accept loop.
        if !inflight_tracker.is_empty() {
            let drained = inflight_tracker.wait_for_drain(drain_timeout).await;
            if drained {
                info!("All in-flight requests drained");
            } else {
                warn!(
                    remaining = inflight_tracker.len(),
                    timeout_secs = drain_timeout.as_secs(),
                    "Drain timed out, forcing shutdown with requests still in-flight"
                );
            }
        }
        // Phase 3: Teardown proceeds after axum server stops (in the main task)
    });

    let server_result = if let (Some(cert), Some(key)) = (
        &config.router_config.server_cert,
        &config.router_config.server_key,
    ) {
        info!("TLS enabled");
        ring::default_provider()
            .install_default()
            .map_err(|e| format!("Failed to install rustls ring provider: {e:?}"))?;

        let tls_config = axum_server::tls_rustls::RustlsConfig::from_pem(cert.clone(), key.clone())
            .await
            .map_err(|e| format!("Failed to create TLS config: {e}"))?;

        axum_server::bind_rustls(addr, tls_config)
            .handle(handle)
            .serve(app.into_make_service_with_connect_info::<std::net::SocketAddr>())
            .await
    } else {
        axum_server::bind(addr)
            .handle(handle)
            .serve(app.into_make_service_with_connect_info::<std::net::SocketAddr>())
            .await
    };

    // Graceful Shutdown

    info!("HTTP server stopped. Starting component cleanup...");

    // This triggers background task cancellation, waits for tools, and denies approvals
    if let Some(orchestrator) = app_context.mcp_orchestrator.get() {
        orchestrator.shutdown().await;
    }

    info!("Cleanup complete. Process exiting.");

    // Return original server error if any, otherwise Ok
    server_result.map_err(|e| Box::new(e) as Box<dyn std::error::Error>)
}

#[expect(
    clippy::expect_used,
    reason = "signal handler installation is infallible on supported platforms; failure is fatal"
)]
async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        () = ctrl_c => {
            info!("Received Ctrl+C, starting graceful shutdown");
        },
        () = terminate => {
            info!("Received terminate signal, starting graceful shutdown");
        },
    }
}

fn create_cors_layer(allowed_origins: Vec<String>) -> tower_http::cors::CorsLayer {
    use tower_http::cors::Any;

    let cors = if allowed_origins.is_empty() {
        tower_http::cors::CorsLayer::new()
            .allow_origin(Any)
            .allow_methods(Any)
            .allow_headers(Any)
            .expose_headers(Any)
    } else {
        let origins: Vec<http::HeaderValue> = allowed_origins
            .into_iter()
            .filter_map(|origin| origin.parse().ok())
            .collect();

        tower_http::cors::CorsLayer::new()
            .allow_origin(origins)
            .allow_methods([
                http::Method::GET,
                http::Method::POST,
                http::Method::PATCH,
                http::Method::DELETE,
                http::Method::OPTIONS,
            ])
            .allow_headers([http::header::CONTENT_TYPE, http::header::AUTHORIZATION])
            .expose_headers([http::header::HeaderName::from_static("x-request-id")])
    };

    cors.max_age(Duration::from_secs(3600))
}

#[cfg(test)]
mod diffusion_helper_tests {
    //! Unit tests for the small helpers backing the diffusion sticky-routing
    //! handlers. The full request flow is covered by router-integration tests.

    use bytes::Bytes;

    use super::{extract_model_from_payload, extract_model_from_query, parse_response_id};

    #[test]
    fn extract_model_from_json_body() {
        let body = Bytes::from(r#"{"model":"foo/bar","prompt":"hi"}"#);
        let model = extract_model_from_payload(&body, "application/json", None);
        assert_eq!(model.as_deref(), Some("foo/bar"));
    }

    #[test]
    fn extract_model_from_multipart_body() {
        let boundary = "BOUNDARY";
        let body = Bytes::from(format!(
            "--{boundary}\r\n\
             Content-Disposition: form-data; name=\"model\"\r\n\r\n\
             diffusion/wan-2-2\r\n\
             --{boundary}--\r\n"
        ));
        let model =
            extract_model_from_payload(&body, "multipart/form-data; boundary=BOUNDARY", None);
        assert_eq!(model.as_deref(), Some("diffusion/wan-2-2"));
    }

    #[test]
    fn falls_back_to_query_when_body_lacks_model() {
        let body = Bytes::from(r#"{"prompt":"hi"}"#);
        let model =
            extract_model_from_payload(&body, "application/json", Some("model=foo%2Fbar&n=1"));
        assert_eq!(model.as_deref(), Some("foo/bar"));
    }

    #[test]
    fn body_takes_precedence_over_query() {
        let body = Bytes::from(r#"{"model":"from-body"}"#);
        let model = extract_model_from_payload(&body, "application/json", Some("model=from-query"));
        assert_eq!(model.as_deref(), Some("from-body"));
    }

    #[test]
    fn no_model_anywhere_returns_none() {
        let body = Bytes::from(r#"{"prompt":"hi"}"#);
        let model = extract_model_from_payload(&body, "application/json", Some("n=1"));
        assert!(model.is_none());
    }

    #[test]
    fn extract_model_from_query_handles_first_param() {
        assert_eq!(
            extract_model_from_query("model=abc&foo=bar").as_deref(),
            Some("abc")
        );
    }

    #[test]
    fn extract_model_from_query_handles_later_param() {
        assert_eq!(
            extract_model_from_query("foo=bar&model=abc").as_deref(),
            Some("abc")
        );
    }

    #[test]
    fn extract_model_from_query_decodes_percent_slash() {
        assert_eq!(
            extract_model_from_query("model=foo%2Fbar%2Fbaz").as_deref(),
            Some("foo/bar/baz")
        );
    }

    #[test]
    fn extract_model_from_query_returns_none_when_missing() {
        assert!(extract_model_from_query("foo=bar").is_none());
    }

    #[test]
    fn parse_response_id_finds_top_level_id() {
        let body = Bytes::from(r#"{"id":"task-123","status":"queued"}"#);
        assert_eq!(parse_response_id(&body).as_deref(), Some("task-123"));
    }

    #[test]
    fn parse_response_id_returns_none_when_no_id() {
        let body = Bytes::from(r#"{"status":"queued"}"#);
        assert!(parse_response_id(&body).is_none());
    }

    #[test]
    fn parse_response_id_returns_none_for_non_json() {
        let body = Bytes::from(b"<html>error</html>".as_slice());
        assert!(parse_response_id(&body).is_none());
    }

    #[test]
    fn parse_response_id_ignores_non_string_id() {
        // OpenAI's spec says id is a string, but defensively reject numeric.
        let body = Bytes::from(r#"{"id":42}"#);
        assert!(parse_response_id(&body).is_none());
    }
}
