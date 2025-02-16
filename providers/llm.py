import argparse
import base64
import glob
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from typing import Any, Literal
import time

import backoff
import openai
from openai import OpenAI
from pdf2image import convert_from_path
from tqdm import tqdm
from dotenv import load_dotenv
from providers.config import settings
import pandas as pd

# Input path for ingestion of pdfs
base_path = os.path.expanduser(settings.input_dir)
pdfs = glob.glob(os.path.join(base_path, "**", "*.pdf"), recursive=True)


def convert_pdf_to_base64_image(pdf_path):
    images = convert_from_path(pdf_path, first_page=1, last_page=1)
    img_buffer = BytesIO()
    images[0].save(img_buffer, format="PNG")
    return base64.b64encode(img_buffer.getvalue()).decode("utf-8")


@backoff.on_exception(backoff.expo, (openai.RateLimitError), max_tries=5)
def analyze_document_openai_sdk(base64_image, model: str):
    start_time = time.time()

    # Define the prompt template within the function
    TABLE_CONVERSION_PROMPT = {
        "type": "text",
        "text": "Convert the image to an HTML table. The output should begin with <table> and end with </table>. Specify rowspan and colspan attributes when they are greater than 1. Do not specify any other attributes. Only use table related HTML tags, no additional formatting is required.",
    }
    TIMEOUT_SECONDS = 300

    if "gemini" in model:
        import google.generativeai as genai
        # Configure the Gemini API
        genai.configure(api_key=settings.gemini_api_key)

        # Create the model instance
        gemini_model = genai.GenerativeModel(model)

        # Create image part for Gemini
        image_part = {"mime_type": "image/png", "data": base64_image}

        # Count input tokens
        input_tokens = gemini_model.count_tokens(
            [TABLE_CONVERSION_PROMPT["text"], image_part]).total_tokens

        # Generate response
        response = gemini_model.generate_content(
            [TABLE_CONVERSION_PROMPT["text"], image_part],
        )

        api_latency = time.time() - start_time

        return {
            'content': response.text,
            'input_tokens': input_tokens,
            'output_tokens': response.usage_metadata.candidates_token_count,
            'api_latency': api_latency
        }
    elif "gpt" in model:
        assert settings.openai_api_key

        client = OpenAI(api_key=settings.openai_api_key)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": TABLE_CONVERSION_PROMPT["text"]},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"}}
                ]}
            ],
            max_tokens=4096,
            timeout=300  # Increase timeout to 300 seconds
        )

        api_latency = time.time() - start_time
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens

        return {
            'content': response.choices[0].message.content,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'api_latency': api_latency
        }
    else:
        raise ValueError(f"Unknown model: {model}")


@backoff.on_exception(backoff.expo, (openai.RateLimitError), max_tries=5)
def analyze_document_anthropic(base64_image, model: str):
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Convert the image to an HTML table. The output should begin with <table> and end with </table>. Specify rowspan and colspan attributes when they are greater than 1. Do not specify any other attributes. Only use table related HTML tags, no additional formatting is required.",
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": base64_image,
                        },
                    },
                ],
            }
        ],
    )
    return response.content[0].text


def parse_gemini_response(content: str) -> tuple[str | None, Any]:
    # Extract just the table portion between <table> and </table>
    start = content.find("<table>")
    end = content.find("</table>") + 8
    if start != -1 and end != -1:
        return content[start:end], None
    return None, None


def check_pdf_type(pdf_path: str) -> str:
    """
    Check if a PDF contains scanned images or is purely digital.
    Returns 'scanned' or 'digital'.
    """
    import fitz  # PyMuPDF

    doc = fitz.open(pdf_path)
    page = doc[0]  # Check first page

    # Get images from the page
    image_list = page.get_images()

    # Get text from the page
    text = page.get_text()

    doc.close()

    # If there are images and very little text, likely a scanned document
    # Adjust threshold as needed
    if len(image_list) > 0 and len(text.strip()) < 100:
        return 'scanned'
    return 'digital'


def process_pdf(pdf_path: str, model: str):
    output_path = os.path.join(
        os.getcwd(),
        'results',
        "outputs",
        f"{model}-raw",
        os.path.basename(pdf_path).replace(".pdf", ".html")
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        pdf_type = check_pdf_type(pdf_path)
        base64_image = convert_pdf_to_base64_image(pdf_path)
        tokens = {'input_tokens': 0, 'output_tokens': 0, 'api_latency': 0}

        if "gemini" in model or "gpt" in model:
            result = analyze_document_openai_sdk(base64_image, model)
            html_table = result['content']
            tokens['input_tokens'] = result['input_tokens']
            tokens['output_tokens'] = result['output_tokens']
            tokens['api_latency'] = result['api_latency']
        elif "claude" in model:
            html_table = analyze_document_anthropic(base64_image, model)
        else:
            raise ValueError(f"Unknown model: {model}")

        html, _ = parse_gemini_response(html_table)

        if not html:
            print(f"Skipping (no HTML found): {pdf_path}")
            return pdf_path, None, tokens, pdf_type

        with open(output_path, "w") as f:
            f.write(html_table)

        return pdf_path, None, tokens, pdf_type
    except Exception as e:
        return pdf_path, str(e), tokens, pdf_type


def process_all_pdfs(
    pdfs: list[str],
    model: Literal[
        "gemini-2.0-flash-exp",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "gpt-4o-mini",
        "gpt-4o",
        "claude-3-5-sonnet-latest",
    ],
    max_workers: int,
    batch_size: int = 200,
):
    # Process PDFs in batches
    for i in range(0, len(pdfs), batch_size):
        batch_pdfs = pdfs[i:i + batch_size]
        print(
            f"\nProcessing batch {i//batch_size + 1} of {(len(pdfs) + batch_size - 1)//batch_size}")

        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(
                process_pdf, pdf, model): pdf for pdf in batch_pdfs}

            progress_bar = tqdm(total=len(batch_pdfs), desc="Processing PDFs")

            for future in as_completed(futures):
                pdf_path, error, tokens, pdf_type = future.result()
                if error:
                    print(f"Error processing {pdf_path}: {error}")
                    continue

                pdf_name = os.path.basename(pdf_path)

                results.append({
                    'pdf_name': pdf_name,
                    'pdf_path': pdf_path,
                    'pdf_type': pdf_type,
                    'input_tokens': tokens['input_tokens'],
                    'output_tokens': tokens['output_tokens'],
                    'api_latency': tokens['api_latency'],
                    'total_cost': (tokens['input_tokens'] * 0.1e-6) + (tokens['output_tokens'] * 0.4e-6)
                })
                progress_bar.update(1)

            progress_bar.close()

        df = pd.DataFrame(results)

        batch_csv = os.path.join(os.getcwd(), 'results',
                                 f'{model}_token_usage_batch_{i//batch_size + 1}.csv')
        os.makedirs(os.path.dirname(batch_csv), exist_ok=True)
        df.to_csv(batch_csv, index=False)

        print(f"Processed {len(batch_pdfs)} PDFs")
        print(f"Token usage details saved to: {batch_csv}")

    # After all batches are processed, merge results
    batch_files = glob.glob(os.path.join(
        os.getcwd(), 'results', f'{model}_token_usage_batch_*.csv'))
    all_results = pd.concat([pd.read_csv(f) for f in batch_files])
    scores_csv = os.path.join(
        os.getcwd(), 'results', 'scores', f'{model}_final_results.csv')
    all_results.to_csv(scores_csv, index=False)

    # Calculate final totals
    total_input_tokens = all_results['input_tokens'].sum()
    total_output_tokens = all_results['output_tokens'].sum()
    total_api_latency = all_results['api_latency'].sum()
    avg_api_latency = all_results['api_latency'].mean()
    total_cost = all_results['total_cost'].sum()
    pdf_type_counts = all_results['pdf_type'].value_counts()

    print("\nFinal Processing Results:")
    print(f"Total PDFs processed: {len(all_results)}")
    print(f"PDF Types: {dict(pdf_type_counts)}")
    print(f"Total input tokens: {total_input_tokens}")
    print(f"Total output tokens: {total_output_tokens}")
    print(f"Total API latency: {total_api_latency:.2f} seconds")
    print(f"Average API latency: {avg_api_latency:.2f} seconds")
    print(f"Total cost: ${total_cost:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=200)
    args = parser.parse_args()
    print(settings.input_dir)
    print(f"Current working directory: {os.getcwd()}")
    print(f'base path {base_path}')
    print(f'pdf paths {pdfs}')

    if "gemini" in args.model:
        assert settings.gemini_api_key

        # client = OpenAI(
        #     api_key=settings.gemini_api_key,
        #     base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        # )
    elif "gpt" in args.model:
        assert settings.openai_api_key

        client = OpenAI(api_key=settings.openai_api_key)
    elif "claude" in args.model:
        from anthropic import Anthropic

        assert settings.anthropic_api_key
        client = Anthropic(api_key=settings.anthropic_api_key)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    process_all_pdfs(pdfs, args.model, args.num_workers, args.batch_size)
