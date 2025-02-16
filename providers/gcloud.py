from typing import Optional, Tuple
from google.api_core.client_options import ClientOptions
from google.cloud import documentai
from google.cloud.documentai_toolbox import document
import os
import glob
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
import backoff

project_id = os.environ["GCP_PROJECT_ID"]
location = "us"
processor_id = os.environ["GCP_PROCESSOR_ID"]
mime_type = "application/pdf"

base_path = os.path.expanduser("~/data/human_table_benchmark")
pdfs = glob.glob(os.path.join(base_path, "**", "*.pdf"), recursive=True)

opts = ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com")
client = documentai.DocumentProcessorServiceClient(client_options=opts)
name = client.processor_path(project_id, location, processor_id)


@backoff.on_exception(
    backoff.expo, Exception, giveup=lambda e: "quota" not in str(e).lower(), max_tries=5
)
def process_document(file_path: str) -> Tuple[Optional[str], float]:
    start_time = time.time()

    # Read the file into memory
    with open(file_path, "rb") as pdf_file:
        pdf_content = pdf_file.read()

    # Load binary data
    raw_document = documentai.RawDocument(content=pdf_content, mime_type=mime_type)

    # Configure the process request
    request = documentai.ProcessRequest(
        name=name,
        raw_document=raw_document,
    )

    result = client.process_document(request=request)

    toolbox_document = document.Document.from_documentai_document(result.document)

    for page in toolbox_document.pages:
        for table in page.tables:
            html_table = table.to_dataframe().to_html()
            processing_time = time.time() - start_time
            return html_table, processing_time

    processing_time = time.time() - start_time
    return None, processing_time


def process_pdf(pdf_path: str):
    output_path = pdf_path.replace("pdfs", "gcloud").replace(".pdf", ".json")

    if os.path.exists(output_path):
        return pdf_path, None

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        html_table, processing_time = process_document(pdf_path)

        result = {"html_table": html_table, "processing_time": processing_time}

        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)

        return pdf_path, None
    except Exception as e:
        return pdf_path, str(e)


def process_all_pdfs(pdfs: list[str]):
    max_workers = 5

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_pdf, pdf): pdf for pdf in pdfs}

        progress_bar = tqdm(total=len(pdfs), desc="Processing PDFs")

        for future in as_completed(futures):
            pdf_path, error = future.result()
            if error:
                print(f"Error processing {pdf_path}: {error}")
            progress_bar.update(1)

        progress_bar.close()

    print(f"Processed {len(pdfs)} PDFs")


if __name__ == "__main__":
    process_all_pdfs(pdfs)
