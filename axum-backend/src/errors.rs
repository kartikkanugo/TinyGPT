use thiserror::Error;

#[derive(Error, Debug)]
pub enum BackendErrors {
    #[error("Invalid Path: {0}")]
    ErrInvalidPath(String),

    #[error("Failed to bind to address: {0}")]
    ErrBindFailed(String),

    #[error("Server runtime error: {0}")]
    ErrServerRuntime(String),
}
