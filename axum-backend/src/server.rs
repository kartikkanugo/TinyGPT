use axum::{routing::get, Router};
use std::net::SocketAddr;
use tokio::net::TcpListener;

use crate::errors::BackendErrors;

pub async fn run_server() -> Result<(), BackendErrors> {
    let app = Router::new().route("/", get(|| async { "Hello, World!" }));

    let address = SocketAddr::from(([127, 0, 0, 1], 3000));

    let listener = TcpListener::bind(address)
        .await
        .map_err(|e| BackendErrors::ErrBindFailed(e.to_string()))?;

    println!("Server running at http://{}", address);

    axum::serve(listener, app)
        .await
        .map_err(|e| BackendErrors::ErrServerRuntime(e.to_string()))?;

    Ok(())
}
