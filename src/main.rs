use clap::{Parser, Subcommand};
use std::path::PathBuf;

mod lexer;
mod parser;
mod ast;
mod types;
mod interpreter;

#[derive(Parser)]
#[command(name = "charl")]
#[command(author = "Charl Team")]
#[command(version = "0.1.0")]
#[command(about = "A revolutionary programming language for AI and Deep Learning", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Run a Charl program
    Run {
        /// Path to the .charl or .ch file
        #[arg(value_name = "FILE")]
        file: PathBuf,
    },
    /// Compile a Charl program to native binary
    Build {
        /// Path to the .charl or .ch file
        #[arg(value_name = "FILE")]
        file: PathBuf,

        /// Output binary path
        #[arg(short, long, value_name = "OUTPUT")]
        output: Option<PathBuf>,
    },
    /// Start Charl REPL (interactive mode)
    Repl,
    /// Show version information
    Version,
}

fn main() {
    let cli = Cli::parse();

    match &cli.command {
        Some(Commands::Run { file }) => {
            println!("🚀 Running Charl program: {}", file.display());
            // TODO: Implement interpreter
            println!("❌ Interpreter not yet implemented");
        }
        Some(Commands::Build { file, output }) => {
            println!("🔨 Building Charl program: {}", file.display());
            if let Some(out) = output {
                println!("📦 Output: {}", out.display());
            }
            // TODO: Implement compiler
            println!("❌ Compiler not yet implemented");
        }
        Some(Commands::Repl) => {
            println!("🎯 Charl REPL v0.1.0");
            println!("Type 'exit' to quit\n");
            // TODO: Implement REPL
            println!("❌ REPL not yet implemented");
        }
        Some(Commands::Version) => {
            println!("Charl v0.1.0");
            println!("A revolutionary programming language for AI and Deep Learning");
        }
        None => {
            println!("⚡ Charl Programming Language v0.1.0\n");
            println!("Usage: charl <COMMAND>\n");
            println!("Commands:");
            println!("  run <FILE>      Run a Charl program");
            println!("  build <FILE>    Compile to native binary");
            println!("  repl            Start interactive mode");
            println!("  version         Show version info");
            println!("\nUse 'charl --help' for more information");
        }
    }
}
