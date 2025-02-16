# PDF Table Extraction Benchmark

This project validates the findings from the article ["Ingesting Millions of PDFs and why Gemini 2.0 Changes Everything"](https://www.sergey.fyi/articles/gemini-flash-2). It builds upon the [rd-tablebench](https://github.com/Filimoa/rd-tablebench) repository to evaluate different LLM providers' capabilities in extracting tabular data from PDFs.

## Overview

This codebase provides tools to:
- Benchmark different LLM providers (OpenAI, Anthropic, Gemini) for PDF table extraction
- Process PDFs in parallel using multiple workers
- Track token usage , api latency and costs
- Grade extraction accuracy

## Dependencies

Install required dependencies using:

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file with the following configurations:

```env
# Directory containing the huggingface dataset
INPUT_DIR=

# Directory for saving output results
OUTPUT_DIR=

# API Keys (only needed for providers you plan to use)
OPENAI_API_KEY=
GEMINI_API_KEY=
ANTHROPIC_API_KEY=
```

## Usage

### Parsing PDFs

Run the parsing process with specified model and worker count:

```bash
python -m providers.llm --model gemini-2.0-flash-001 --num-workers 10
```

Available models:
- gemini-2.0-flash-001
- gpt-4
- claude-3

### Grading Results

Grade the extraction results:

```bash
python -m grade_cli --model gemini-2.0-flash-001
```

## Insights

- **Total PDFs processed:** 945
- **Total input tokens:** 294840
- **Total output tokens:** 670605
- **Average API latency:** 12.73 seconds
- **Total cost:** $0.297726

## Project Structure

```
├── providers/          # LLM provider implementations
├── results/           # Output directory for results
├── requirements.txt   # Project dependencies
└── README.md         # This file
```

## Acknowledgments

- Original implementation: [rd-tablebench](https://github.com/Filimoa/rd-tablebench)
- Research article: ["Ingesting Millions of PDFs and why Gemini 2.0 Changes Everything"](https://www.sergey.fyi/articles/gemini-flash-2)

## License

This project follows the same licensing as the original rd-tablebench repository.
