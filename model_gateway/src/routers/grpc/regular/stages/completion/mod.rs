//! Completion endpoint pipeline stages (/v1/completions)
//!
//! These stages handle completion-specific preprocessing, request building, and response processing.
//! The completion endpoint is text-based (prompt in, text out), similar to generate but uses
//! OpenAI's CompletionRequest/CompletionResponse format.

mod preparation;
mod request_building;
mod response_processing;

pub(crate) use preparation::CompletionPreparationStage;
pub(crate) use request_building::CompletionRequestBuildingStage;
pub(crate) use response_processing::CompletionResponseProcessingStage;
