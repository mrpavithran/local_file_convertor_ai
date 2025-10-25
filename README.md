AI File System

A modular, extensible AI-powered file system that supports document processing, file conversion, image and text analysis, web scraping, and system utilities with a unified interface. Designed to run locally or integrate with AI models like Ollama, OpenAI, or Anthropic.

Table of Contents

Features

Architecture & Flow

Installation

Directory Structure

Usage

MCP Tools

RAG & Vector Store

Examples

Contributing

License

Features

File Operations: Auto-detect file types, scan directories, maintain or flatten structures.

Conversion Tools: Convert DOCX ↔ PDF, CSV ↔ XLSX, batch file conversions.

Image Tools: OCR, enhancement, metadata extraction, upscaling.

Text Tools: Summarization, translation, cleaning, and analysis.

Web Tools: Scrape websites, validate URLs, analyze web content.

System Tools: Disk usage, network monitoring, process checking, system info.

RAG (Retrieval-Augmented Generation): Build context from PDFs, DOCX, CSV, and TXT using ChromaDB vector store.

CLI & Interactive Mode: Access all features via command-line interface or interactive prompts.

Extensible: Add new MCP tools in structured categories.

Architecture & Flow

The system is modular. Core components and flow:

CLI / Interactive Interface (cli/)
Users interact via main.py or interactive prompts. Commands are parsed and dispatched.

Core Engine (core/)
Orchestrates file operations, tracks progress, validates input, and handles errors.

File Operations (file_operations/)
Handles file detection, batch processing, output management, and error handling.

MCP (Model Context Protocol) (ai_infrastructure/mcp/)

Tools are organized by category (file_tools, conversion_tools, image_tools, text_tools, web_tools, system_tools).

tool_registry.py automatically discovers and registers tools.

mcp_server.py exposes APIs to execute tools programmatically.

RAG Engine (ai_infrastructure/rag/)

Ingests documents and stores embeddings in ChromaDB.

Supports context retrieval for AI models.

AI Models / Execution (ai_infrastructure/ollama/)

Manage LLMs like Mistral for local AI execution.

Execute prompts via prompt_executor.py using templates in prompt_system/.

Installation

Clone the repository:

git clone https://github.com/yourusername/prompt-ai-file-system.git
cd prompt-ai-file-system


Install dependencies (recommended in a virtual environment):

python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install --upgrade pip
pip install -r requirements.txt


Optional: Set up Ollama models locally (if using ollama-python):

python scripts/deploy_ollama.py


Initialize RAG / vector store (optional):

python scripts/setup_rag.py

Directory Structure
prompt-ai-file-system/
├── ai_infrastructure/   # AI, RAG, and MCP tools
├── prompt_system/       # Prompt templates & workflows
├── file_operations/     # File handling & batch processing
├── core/                # Orchestration, config, utils
├── cli/                 # CLI entry points & commands
├── data/                # Output files, ChromaDB, logs
├── tests/               # Unit tests
├── docs/                # Documentation
├── examples/            # Example scripts
├── config/              # YAML configuration files
├── scripts/             # Setup, registration, and health checks
├── main.py              # Alternative entry point
├── requirements.txt
└── README.md

Usage
CLI
python cli/main.py convert --input "docs/sample.docx" --output "output/sample.pdf"
python cli/main.py enhance --image "images/sample.jpg"
python cli/main.py batch --directory "docs/batch_files"

Interactive Mode
python cli/main.py --interactive


Follow prompts to select files, tools, and operations.

MCP Tools

Tools are categorized in ai_infrastructure/mcp/tools/.

Categories:

file_tools, conversion_tools, image_tools, text_tools, web_tools, system_tools

To register tools:

python scripts/register_tools.py --all

RAG & Vector Store

ChromaDB stores embeddings for documents.

Add new documents via document_ingestor.py.

Retrieve context using context_retriever.py.

Use with prompts for LLM-powered retrieval.

Examples

examples/basic_conversion.py → Convert a single file.

examples/batch_processing.py → Process multiple files.

examples/ai_enhancement.py → Apply AI-based image/text enhancement.

examples/sample_prompts.txt → Ready-to-use prompt templates.

Contributing

Fork the repository

Create a new branch for your feature

Add tests for new functionality in tests/

Submit a pull request

Refer to docs/tool_development.md for adding new MCP tools.

License

MIT License. Free for personal, educational, and commercial use.