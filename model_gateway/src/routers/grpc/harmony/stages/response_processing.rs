//! Harmony Response Processing Stage: Parse Harmony channels to ChatCompletionResponse

use std::{sync::Arc, time::Instant};

use async_trait::async_trait;
use axum::response::Response;
use tracing::error;

use super::super::{HarmonyResponseProcessor, HarmonyStreamingProcessor};
use crate::{
    core::AttachedBody,
    observability::metrics::{metrics_labels, Metrics, RequestMetricsParams},
    routers::{
        error,
        grpc::{
            common::stages::PipelineStage,
            context::{FinalResponse, RequestContext, RequestType},
        },
    },
};

/// Harmony Response Processing stage: Parse and format Harmony responses
///
/// Takes output tokens from execution and parses them using HarmonyParserAdapter
/// to extract analysis, tool calls, and final response text from Harmony channels.
pub(crate) struct HarmonyResponseProcessingStage {
    processor: HarmonyResponseProcessor,
    streaming_processor: Arc<HarmonyStreamingProcessor>,
}

impl HarmonyResponseProcessingStage {
    /// Create a new Harmony response processing stage
    pub fn new() -> Self {
        Self {
            processor: HarmonyResponseProcessor::new(),
            streaming_processor: Arc::new(HarmonyStreamingProcessor::new()),
        }
    }
}

impl Default for HarmonyResponseProcessingStage {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl PipelineStage for HarmonyResponseProcessingStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        let is_streaming = ctx.is_streaming();

        // Check request type to determine which processor method to call
        match &ctx.input.request_type {
            RequestType::Chat(_) => {
                // Get execution result (output tokens from model)
                let execution_result =
                    ctx.state.response.execution_result.take().ok_or_else(|| {
                        error!(
                            function = "HarmonyResponseProcessingStage::execute",
                            request_type = "Chat",
                            "No execution result available"
                        );
                        error::internal_error("no_execution_result", "No execution result")
                    })?;

                let dispatch = ctx.state.dispatch.as_ref().cloned().ok_or_else(|| {
                    error!(
                        function = "HarmonyResponseProcessingStage::execute",
                        request_type = "Chat",
                        "Dispatch metadata not set"
                    );
                    error::internal_error("dispatch_metadata_not_set", "Dispatch metadata not set")
                })?;

                // For streaming, delegate to streaming processor and return SSE response
                if is_streaming {
                    let response = self
                        .streaming_processor
                        .clone()
                        .process_streaming_chat_response(
                            execution_result,
                            ctx.chat_request_arc(),
                            dispatch,
                            ctx.state.pipeline_start,
                        );

                    // Attach load guards to response body for proper RAII lifecycle
                    let response = match ctx.state.load_guards.take() {
                        Some(guards) => AttachedBody::wrap_response(response, guards),
                        None => response,
                    };

                    return Ok(Some(response));
                }

                // For non-streaming, delegate to Harmony response processor to build ChatCompletionResponse
                let start_time = Instant::now();
                let chat_request = ctx.chat_request_arc();
                let response = self
                    .processor
                    .process_non_streaming_chat_response(
                        execution_result,
                        chat_request,
                        dispatch.clone(),
                    )
                    .await?;

                // Record non-streaming request metrics
                let (input_tokens, output_tokens) = response
                    .usage
                    .as_ref()
                    .map(|u| {
                        (
                            Some(u64::from(u.prompt_tokens)),
                            u64::from(u.completion_tokens),
                        )
                    })
                    .unwrap_or((None, 0));

                Metrics::record_request_metrics(RequestMetricsParams {
                    router_type: metrics_labels::ROUTER_GRPC,
                    backend_type: metrics_labels::BACKEND_HARMONY,
                    model_id: &dispatch.model,
                    endpoint: metrics_labels::ENDPOINT_CHAT,
                    ttft: None,
                    generation_duration: start_time.elapsed(),
                    input_tokens,
                    output_tokens,
                    itl_observations: &[],
                    e2e_latency: ctx.state.pipeline_start.map(|ps| ps.elapsed()),
                    queue_time: ctx
                        .state
                        .pipeline_start
                        .map(|ps| start_time.saturating_duration_since(ps)),
                });

                ctx.state.response.final_response = Some(FinalResponse::Chat(response));
                Ok(None)
            }
            RequestType::Responses(_) => {
                // For streaming Responses API, leave execution_result in context
                // for external streaming processor (serve_harmony_responses_stream)
                if is_streaming {
                    // Don't take execution_result - let the caller handle it
                    return Ok(None);
                }

                // For non-streaming, process normally
                let execution_result =
                    ctx.state.response.execution_result.take().ok_or_else(|| {
                        error!(
                            function = "HarmonyResponseProcessingStage::execute",
                            request_type = "Responses",
                            "No execution result available"
                        );
                        error::internal_error("no_execution_result", "No execution result")
                    })?;

                let dispatch = ctx.state.dispatch.as_ref().cloned().ok_or_else(|| {
                    error!(
                        function = "HarmonyResponseProcessingStage::execute",
                        request_type = "Responses",
                        "Dispatch metadata not set"
                    );
                    error::internal_error("dispatch_metadata_not_set", "Dispatch metadata not set")
                })?;

                let responses_request = ctx.responses_request_arc();
                let iteration_result = self
                    .processor
                    .process_responses_iteration(execution_result, responses_request, dispatch)
                    .await?;

                ctx.state.response.responses_iteration_result = Some(iteration_result);
                Ok(None)
            }
            request_type @ (RequestType::Generate(_)
            | RequestType::Embedding(_)
            | RequestType::Classify(_)
            | RequestType::Messages(_)) => {
                error!(
                    function = "HarmonyResponseProcessingStage::execute",
                    request_type = %request_type,
                    "{request_type} request type not supported in Harmony pipeline"
                );
                Err(error::internal_error(
                    "not_supported_in_harmony",
                    format!("{request_type} requests not supported in Harmony pipeline"),
                ))
            }
        }
    }

    fn name(&self) -> &'static str {
        "HarmonyResponseProcessing"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_response_processing_stage_creation() {
        let stage = HarmonyResponseProcessingStage::new();
        assert_eq!(stage.name(), "HarmonyResponseProcessing");
    }
}
