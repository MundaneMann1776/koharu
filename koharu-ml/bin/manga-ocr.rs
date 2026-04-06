use clap::Parser;
use koharu_ml::manga_ocr::MangaOcr;

#[path = "common.rs"]
mod common;

#[derive(Parser)]
struct Cli {
    #[arg(short, long, value_name = "FILE")]
    input: String,

    #[arg(long, default_value_t = false)]
    cpu: bool,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    common::init_tracing();

    let cli = Cli::parse();
    let image = image::open(&cli.input)?;
    let images = vec![image];

    let runtime = common::prepare_runtime(cli.cpu).await?;
    let cpu = common::effective_cpu(&runtime, cli.cpu, "manga-ocr")?;
    let model = MangaOcr::load(&runtime, cpu).await?;
    let output = model
        .inference(&images)?
        .into_iter()
        .next()
        .unwrap_or_default();

    println!("{output}");

    Ok(())
}
