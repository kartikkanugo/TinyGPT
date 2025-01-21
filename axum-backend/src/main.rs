mod errors;
mod server;
use std::process;

#[tokio::main]
async fn main() {
    if let Err(e) = server::run_server().await {
        eprintln!("Server failed: {}", e);
        process::exit(1);
    }
}
