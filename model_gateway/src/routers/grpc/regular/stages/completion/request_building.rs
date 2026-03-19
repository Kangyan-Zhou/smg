//! Completion request building stage: Build proto GenerateRequest for completion requests

use async_trait::async_trait;
use axum::response::Response;
use tracing::error;
use uuid::Uuid;

use crate::routers::{
    error,
    grpc::{
        common::stages::{helpers, PipelineStage},
        context::{ClientSelection, RequestContext},
        proto_wrapper::ProtoRequest,
    },
};

/// Completion request building stage
///
/// Builds a proto GenerateRequest from a CompletionRequest. Uses the dedicated
/// `build_completion_request` client method to properly map CompletionRequest
/// sampling parameters to the backend-specific proto format.
pub(crate) struct CompletionRequestBuildingStage {
    inject_pd_metadata: bool,
}

impl CompletionRequestBuildingStage {
    pub fn new(inject_pd_metadata: bool) -> Self {
        Self { inject_pd_metadata }
    }
}

#[async_trait]
impl PipelineStage for CompletionRequestBuildingStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        let prep = ctx.state.preparation.as_ref().ok_or_else(|| {
            error!(
                function = "CompletionRequestBuildingStage::execute",
                "Preparation not completed"
            );
            error::internal_error("preparation_not_completed", "Preparation not completed")
        })?;

        let clients = ctx.state.clients.as_ref().ok_or_else(|| {
            error!(
                function = "CompletionRequestBuildingStage::execute",
                "Client acquisition not completed"
            );
            error::internal_error(
                "client_acquisition_not_completed",
                "Client acquisition not completed",
            )
        })?;

        let completion_request = ctx.completion_request_arc();

        // Get client for building request (use prefill client if PD mode)
        let builder_client = match clients {
            ClientSelection::Single { client } => client,
            ClientSelection::Dual { prefill, .. } => prefill,
        };

        // Build request ID
        let request_id = format!("cmpl-{}", Uuid::now_v7());

        // Build proto request using the completion-specific builder
        let original_text = prep.original_text.clone().unwrap_or_default();
        let mut proto_request = builder_client
            .build_completion_request(
                request_id,
                &completion_request,
                original_text,
                prep.token_ids.clone(),
            )
            .map_err(|e| {
                error!(
                    function = "CompletionRequestBuildingStage::execute",
                    "Failed to build completion request: {e}"
                );
                error::internal_error("build_completion_request_failed", e)
            })?;

        if self.inject_pd_metadata {
            if let Some(workers) = ctx.state.workers.as_ref() {
                helpers::maybe_inject_pd_metadata(&mut proto_request, workers);
            }
        }

        ctx.state.proto_request = Some(ProtoRequest::Generate(proto_request));
        Ok(None)
    }

    fn name(&self) -> &'static str {
        "CompletionRequestBuilding"
    }
}
