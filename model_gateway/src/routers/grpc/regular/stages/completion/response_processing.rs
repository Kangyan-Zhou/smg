//! Completion response processing stage: Handles both streaming and non-streaming responses
//!
//! Produces CompletionResponse (OpenAI /v1/completions format) from the gRPC execution result.

use std::{sync::Arc, time::Instant};

use async_trait::async_trait;
use axum::response::Response;
use llm_tokenizer::stop::SequenceDecoderOutput;
use openai_protocol::completion::{CompletionChoice, CompletionResponse};
use tracing::error;

use crate::{
    core::AttachedBody,
    observability::metrics::{metrics_labels, Metrics, RequestMetricsParams},
    routers::{
        error,
        grpc::{
            common::{response_collection, response_formatting, stages::PipelineStage},
            context::{FinalResponse, RequestContext},
            regular::streaming,
        },
    },
};

/// Completion response processing stage
///
/// Processes the gRPC execution result and produces a CompletionResponse.
/// For streaming, delegates to the StreamingProcessor's completion-specific method.
pub(crate) struct CompletionResponseProcessingStage {
    streaming_processor: Arc<streaming::StreamingProcessor>,
}

impl CompletionResponseProcessingStage {
    pub fn new(streaming_processor: Arc<streaming::StreamingProcessor>) -> Self {
        Self {
            streaming_processor,
        }
    }
}

#[async_trait]
impl PipelineStage for CompletionResponseProcessingStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        self.process_completion_response(ctx).await
    }

    fn name(&self) -> &'static str {
        "CompletionResponseProcessing"
    }
}

impl CompletionResponseProcessingStage {
    async fn process_completion_response(
        &self,
        ctx: &mut RequestContext,
    ) -> Result<Option<Response>, Response> {
        let start_time = Instant::now();
        let is_streaming = ctx.is_streaming();

        // Extract execution result
        let execution_result = ctx.state.response.execution_result.take().ok_or_else(|| {
            error!(
                function = "CompletionResponseProcessingStage::execute",
                "No execution result"
            );
            error::internal_error("no_execution_result", "No execution result")
        })?;

        // Get dispatch metadata
        let dispatch = ctx
            .state
            .dispatch
            .as_ref()
            .ok_or_else(|| {
                error!(
                    function = "CompletionResponseProcessingStage::execute",
                    "Dispatch metadata not set"
                );
                error::internal_error("dispatch_metadata_not_set", "Dispatch metadata not set")
            })?
            .clone();

        // Get cached tokenizer
        let tokenizer = ctx.tokenizer_arc().ok_or_else(|| {
            error!(
                function = "CompletionResponseProcessingStage::process_completion_response",
                "Tokenizer not cached in context"
            );
            error::internal_error(
                "tokenizer_not_cached",
                "Tokenizer not cached in context - preparation stage may have been skipped",
            )
        })?;

        if is_streaming {
            // Streaming: Use StreamingProcessor's completion-specific method
            // TODO: pass ctx.state.pipeline_start once grpc_backpressure lands
            let response = self
                .streaming_processor
                .clone()
                .process_streaming_completion(
                    execution_result,
                    ctx.completion_request_arc(),
                    dispatch,
                    tokenizer,
                    None,
                );

            // Attach load guards to response body for proper RAII lifecycle
            let response = match ctx.state.load_guards.take() {
                Some(guards) => AttachedBody::wrap_response(response, guards),
                None => response,
            };

            return Ok(Some(response));
        }

        // Non-streaming: Process all completions and build CompletionResponse
        let request_logprobs = ctx.completion_request().logprobs.is_some();

        let stop_decoder = ctx.state.response.stop_decoder.as_mut().ok_or_else(|| {
            error!(
                function = "CompletionResponseProcessingStage::execute",
                "Stop decoder not initialized"
            );
            error::internal_error(
                "stop_decoder_not_initialized",
                "Stop decoder not initialized",
            )
        })?;

        // Collect all responses from the execution result
        let all_responses =
            response_collection::collect_responses(execution_result, request_logprobs).await?;

        let mut choices = Vec::new();

        for (index, complete) in all_responses.iter().enumerate() {
            stop_decoder.reset();

            // Process tokens through stop decoder
            let outputs = match stop_decoder.process_tokens(complete.output_ids()) {
                Ok(outputs) => outputs,
                Err(e) => {
                    error!(
                        function = "CompletionResponseProcessingStage::process_completion_response",
                        "Failed to process tokens: {e}"
                    );
                    return Err(error::internal_error(
                        "process_tokens_failed",
                        format!("Failed to process tokens: {e}"),
                    ));
                }
            };

            // Accumulate text
            let mut decoded_text = String::new();
            for output in outputs {
                match output {
                    SequenceDecoderOutput::Text(t) => decoded_text.push_str(&t),
                    SequenceDecoderOutput::StoppedWithText(t) => {
                        decoded_text.push_str(&t);
                        break;
                    }
                    SequenceDecoderOutput::Stopped => break,
                    SequenceDecoderOutput::Held => {}
                }
            }

            // Flush remaining text
            if let SequenceDecoderOutput::Text(t) = stop_decoder.flush() {
                decoded_text.push_str(&t);
            }

            let finish_reason_str = complete.finish_reason();
            // CompletionResponse uses simple string finish reasons ("stop", "length", etc.)
            let finish_reason = finish_reason_str.to_string();
            let matched_stop = complete.matched_stop_json();

            choices.push(CompletionChoice {
                text: decoded_text,
                index: index as u32,
                logprobs: None, // TODO: map logprobs to OpenAI format if needed
                finish_reason: Some(finish_reason),
                matched_stop,
            });
        }

        let completion_response = CompletionResponse {
            id: dispatch.request_id.clone(),
            object: "text_completion".to_string(),
            created: dispatch.created,
            model: dispatch.model.clone(),
            choices,
            usage: Some(response_formatting::build_usage(&all_responses)),
            system_fingerprint: dispatch.weight_version.clone(),
        };

        // Record non-streaming request metrics
        let (input_tokens, output_tokens) = completion_response
            .usage
            .as_ref()
            .map(|u| (Some(u64::from(u.prompt_tokens)), u64::from(u.completion_tokens)))
            .unwrap_or((None, 0));

        Metrics::record_request_metrics(RequestMetricsParams {
            router_type: metrics_labels::ROUTER_GRPC,
            backend_type: self.streaming_processor.backend_type(),
            model_id: &dispatch.model,
            endpoint: metrics_labels::ENDPOINT_COMPLETIONS,
            ttft: None,
            generation_duration: start_time.elapsed(),
            input_tokens,
            output_tokens,
            itl_observations: &[],
            e2e_latency: ctx.state.pipeline_start.map(|ps| ps.elapsed()),
            queue_time: ctx.state.pipeline_start.map(|ps| start_time.saturating_duration_since(ps)),
        });

        // Store the final response
        ctx.state.response.final_response = Some(FinalResponse::Completion(completion_response));

        Ok(None)
    }
}
