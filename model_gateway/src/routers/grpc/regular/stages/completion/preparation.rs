//! Completion preparation stage: Resolve prompt input, tokenize, create stop decoder

use std::sync::Arc;

use async_trait::async_trait;
use axum::response::Response;
use llm_tokenizer::traits::Tokenizer;
use openai_protocol::{common::StringOrArray, completion::CompletionRequest};
use tracing::error;

use crate::routers::{
    error,
    grpc::{
        common::stages::PipelineStage,
        context::{PreparationOutput, RequestContext},
        utils,
    },
};

/// Completion preparation stage
///
/// Handles CompletionRequest-specific preparation: resolves the `prompt` field
/// (which can be a string or array), tokenizes it, and creates a stop decoder.
pub(crate) struct CompletionPreparationStage;

#[async_trait]
impl PipelineStage for CompletionPreparationStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        let request = ctx.completion_request_arc();
        self.prepare_completion(ctx, &request)?;
        Ok(None)
    }

    fn name(&self) -> &'static str {
        "CompletionPreparation"
    }
}

impl CompletionPreparationStage {
    #[expect(
        clippy::result_large_err,
        reason = "Response is the standard error type in the pipeline stage pattern"
    )]
    fn prepare_completion(
        &self,
        ctx: &mut RequestContext,
        request: &CompletionRequest,
    ) -> Result<(), Response> {
        // Resolve tokenizer from registry (cached for reuse in response processing)
        let tokenizer =
            utils::resolve_tokenizer(ctx, "CompletionPreparationStage::prepare_completion")
                .map_err(|e| *e)?;

        let (original_text, token_ids) = match self.resolve_completion_input(request, &tokenizer) {
            Ok(res) => res,
            Err(msg) => {
                error!(function = "CompletionPreparationStage::execute", error = %msg, "Failed to resolve completion input");
                return Err(error::bad_request("resolve_input_failed", msg));
            }
        };

        // Create stop sequence decoder for completion requests
        let stop_decoder = utils::create_stop_decoder(
            &tokenizer,
            request.stop.as_ref(),
            request.stop_token_ids.as_ref(),
            request.skip_special_tokens,
            request.no_stop_trim,
        );

        ctx.state.preparation = Some(PreparationOutput {
            original_text: Some(original_text),
            token_ids,
            processed_messages: None,
            tool_constraints: None,
            filtered_request: None,
            // Harmony fields (not used for completion requests)
            harmony_mode: false,
            selection_text: None,
            harmony_messages: None,
            harmony_stop_ids: None,
        });

        // Store stop decoder
        ctx.state.response.stop_decoder = Some(stop_decoder);

        Ok(())
    }

    #[expect(
        clippy::unused_self,
        reason = "method on stage struct for consistency with resolve_generate_input"
    )]
    fn resolve_completion_input(
        &self,
        request: &CompletionRequest,
        tokenizer: &Arc<dyn Tokenizer>,
    ) -> Result<(String, Vec<u32>), String> {
        match &request.prompt {
            StringOrArray::String(text) => {
                // Don't add special tokens - raw text completion uses text as-is
                let encoding = tokenizer
                    .encode(text, false)
                    .map_err(|e| format!("Tokenization failed: {e}"))?;
                Ok((text.clone(), encoding.token_ids().to_vec()))
            }
            StringOrArray::Array(texts) => {
                // For array prompts, concatenate (batch not supported over gRPC yet)
                if texts.is_empty() {
                    return Err("Prompt array must not be empty".to_string());
                }
                if texts.len() > 1 {
                    return Err(
                        "Batch prompts (array with multiple strings) are not supported over gRPC completion yet"
                            .to_string(),
                    );
                }
                let text = texts.first().cloned().unwrap_or_default();
                let encoding = tokenizer
                    .encode(&text, false)
                    .map_err(|e| format!("Tokenization failed: {e}"))?;
                Ok((text, encoding.token_ids().to_vec()))
            }
        }
    }
}
