# import libraries
import os
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
import glob
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from azure.core.exceptions import HttpResponseError
import backoff

endpoint = os.environ["AZURE_ENDPOINT"]
key = os.environ["AZURE_KEY"]

base_path = os.path.expanduser("~/data/human_table_benchmark")
pdfs = glob.glob(os.path.join(base_path, "**", "*.pdf"), recursive=True)

document_intelligence_client = DocumentIntelligenceClient(
    endpoint=endpoint, credential=AzureKeyCredential(key)
)


def is_rate_limit_error(exception):
    return isinstance(exception, HttpResponseError) and exception.status_code == 429


@backoff.on_exception(
    backoff.expo, HttpResponseError, giveup=lambda e: not is_rate_limit_error(e)
)
def analyze_document(file_content):
    return document_intelligence_client.begin_analyze_document(
        "prebuilt-layout", AnalyzeDocumentRequest(bytes_source=file_content)
    ).result()


def process_pdf(pdf_path: str):
    output_path = pdf_path.replace("pdfs", "azure").replace(".pdf", ".json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        with open(pdf_path, "rb") as file:
            result = analyze_document(file.read())

        with open(output_path, "w") as f:
            json.dump(result.as_dict(), f, indent=2)

        return pdf_path, None
    except HttpResponseError as e:
        if e.status_code == 429:  # Rate limit exceeded
            return pdf_path, "Rate limit"
        else:
            return pdf_path, str(e)
    except Exception as e:
        return pdf_path, str(e)


def process_all_pdfs(pdfs: list[str]):
    max_workers = 200

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
