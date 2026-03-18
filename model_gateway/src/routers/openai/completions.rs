//! Completions routing for the OpenAI router.

use std::{sync::Arc, time::Instant};

use axum::{
    body::Body,
    http::{header::CONTENT_TYPE, HeaderMap, HeaderValue, StatusCode},
    response::Response,
};
use futures_util::StreamExt;
use openai_protocol::completion::CompletionRequest;
use serde_json::to_value;
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;

use super::{context::SharedComponents, provider::ProviderRegistry, router::resolve_provider};
use crate::{
    config::types::RetryConfig,
    core::{is_retryable_status, Endpoint, ProviderType, RetryExecutor, WorkerRegistry},
    observability::metrics::{bool_to_static_str, metrics_labels, Metrics},
    routers::{
        error,
        header_utils::{apply_provider_headers, extract_auth_header},
        worker_selection::{SelectWorkerRequest, WorkerSelector},
    },
};

/// Shared context passed to completion routing functions.
pub(super) struct CompletionRouterContext<'a> {
    pub worker_registry: &'a WorkerRegistry,
    pub provider_registry: &'a ProviderRegistry,
    pub shared_components: &'a Arc<SharedComponents>,
    pub client: &'a reqwest::Client,
    pub retry_config: &'a RetryConfig,
}

/// Route a completion request to the appropriate upstream worker.
pub(super) async fn route_completion(
    deps: &CompletionRouterContext<'_>,
    headers: Option<&HeaderMap>,
    body: &CompletionRequest,
    model_id: Option<&str>,
) -> Response {
    let start = Instant::now();
    let model = model_id.unwrap_or(body.model.as_str());
    let streaming = body.stream;

    Metrics::record_router_request(
        metrics_labels::ROUTER_OPENAI,
        metrics_labels::BACKEND_EXTERNAL,
        metrics_labels::CONNECTION_HTTP,
        model,
        metrics_labels::ENDPOINT_COMPLETIONS,
        bool_to_static_str(streaming),
    );

    let selector = WorkerSelector::new(deps.worker_registry, deps.client);
    let worker = match selector
        .select_worker(&SelectWorkerRequest {
            model_id: model,
            headers,
            provider: Some(ProviderType::OpenAI),
            ..Default::default()
        })
        .await
    {
        Ok(w) => w,
        Err(response) => {
            Metrics::record_router_error(
                metrics_labels::ROUTER_OPENAI,
                metrics_labels::BACKEND_EXTERNAL,
                metrics_labels::CONNECTION_HTTP,
                model,
                metrics_labels::ENDPOINT_COMPLETIONS,
                metrics_labels::ERROR_NO_WORKERS,
            );
            return response;
        }
    };

    let mut payload = match to_value(body) {
        Ok(v) => v,
        Err(e) => {
            Metrics::record_router_error(
                metrics_labels::ROUTER_OPENAI,
                metrics_labels::BACKEND_EXTERNAL,
                metrics_labels::CONNECTION_HTTP,
                model,
                metrics_labels::ENDPOINT_COMPLETIONS,
                metrics_labels::ERROR_VALIDATION,
            );
            return error::bad_request(
                "invalid_request",
                format!("Failed to serialize request: {e}"),
            );
        }
    };

    // When model_id overrides the body model, patch the serialized payload
    // so the upstream request uses the effective model consistently.
    if model_id.is_some() {
        payload["model"] = serde_json::Value::String(model.to_owned());
    }

    let provider = resolve_provider(deps.provider_registry, worker.as_ref(), model);
    if let Err(e) = provider.transform_request(&mut payload, Endpoint::Completions) {
        Metrics::record_router_error(
            metrics_labels::ROUTER_OPENAI,
            metrics_labels::BACKEND_EXTERNAL,
            metrics_labels::CONNECTION_HTTP,
            model,
            metrics_labels::ENDPOINT_COMPLETIONS,
            metrics_labels::ERROR_VALIDATION,
        );
        return error::bad_request("invalid_request", format!("Provider transform error: {e}"));
    }

    let url = format!("{}/v1/completions", worker.url());

    // Wrap values in Arc to avoid cloning large objects on each retry attempt
    let payload_json = Arc::new(payload);
    let client = deps.shared_components.client.clone();
    let headers_cloned = Arc::new(headers.cloned());
    let worker_api_key = Arc::new(worker.api_key().cloned());

    let response = RetryExecutor::execute_response_with_retry(
        deps.retry_config,
        |_attempt| {
            let client = client.clone();
            let url = url.clone();
            let payload = Arc::clone(&payload_json);
            let headers = Arc::clone(&headers_cloned);
            let worker_api_key = Arc::clone(&worker_api_key);
            let worker = Arc::clone(&worker);

            async move {
                let mut req = client.post(&url).json(&*payload);
                let auth_header =
                    extract_auth_header((*headers).as_ref(), (*worker_api_key).as_ref());
                req = apply_provider_headers(req, &url, auth_header.as_ref());

                if streaming {
                    req = req.header("Accept", "text/event-stream");
                }

                let resp = match req.send().await {
                    Ok(r) => r,
                    Err(e) => {
                        worker.circuit_breaker().record_failure();
                        return error::service_unavailable(
                            "upstream_error",
                            format!("Failed to contact upstream: {e}"),
                        );
                    }
                };

                let status = StatusCode::from_u16(resp.status().as_u16())
                    .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

                if !status.is_success() {
                    worker.circuit_breaker().record_failure();
                }

                if streaming {
                    if status.is_success() {
                        worker.circuit_breaker().record_success();
                    }
                    let stream = resp.bytes_stream();
                    let (tx, rx) = mpsc::unbounded_channel();
                    #[expect(clippy::disallowed_methods, reason = "fire-and-forget stream relay; gateway shutdown need not wait for individual stream forwarding")]
                    tokio::spawn(async move {
                        let mut s = stream;
                        while let Some(chunk) = s.next().await {
                            match chunk {
                                Ok(bytes) => {
                                    if tx.send(Ok(bytes)).is_err() {
                                        break;
                                    }
                                }
                                Err(e) => {
                                    let _ = tx.send(Err(format!("Stream error: {e}")));
                                    break;
                                }
                            }
                        }
                    });
                    let mut response =
                        Response::new(Body::from_stream(UnboundedReceiverStream::new(rx)));
                    *response.status_mut() = status;
                    response
                        .headers_mut()
                        .insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
                    response
                } else {
                    let content_type = resp.headers().get(CONTENT_TYPE).cloned();
                    match resp.bytes().await {
                        Ok(body) => {
                            if status.is_success() {
                                worker.circuit_breaker().record_success();
                            }
                            let mut response = Response::new(Body::from(body));
                            *response.status_mut() = status;
                            if let Some(ct) = content_type {
                                response.headers_mut().insert(CONTENT_TYPE, ct);
                            }
                            response
                        }
                        Err(e) => {
                            worker.circuit_breaker().record_failure();
                            error::internal_error(
                                "upstream_error",
                                format!("Failed to read response: {e}"),
                            )
                        }
                    }
                }
            }
        },
        |res, _attempt| is_retryable_status(res.status()),
        |delay, attempt| {
            Metrics::record_worker_retry(
                metrics_labels::BACKEND_EXTERNAL,
                metrics_labels::ENDPOINT_COMPLETIONS,
            );
            Metrics::record_worker_retry_backoff(attempt, delay);
        },
        || {
            Metrics::record_worker_retries_exhausted(
                metrics_labels::BACKEND_EXTERNAL,
                metrics_labels::ENDPOINT_COMPLETIONS,
            );
        },
    )
    .await;

    if response.status().is_success() {
        Metrics::record_router_duration(
            metrics_labels::ROUTER_OPENAI,
            metrics_labels::BACKEND_EXTERNAL,
            metrics_labels::CONNECTION_HTTP,
            model,
            metrics_labels::ENDPOINT_COMPLETIONS,
            start.elapsed(),
        );
    } else {
        Metrics::record_router_error(
            metrics_labels::ROUTER_OPENAI,
            metrics_labels::BACKEND_EXTERNAL,
            metrics_labels::CONNECTION_HTTP,
            model,
            metrics_labels::ENDPOINT_COMPLETIONS,
            metrics_labels::ERROR_BACKEND,
        );
    }

    response
}
