# Before calling the API, replace filename and ensure sdk is installed: "pip install unstructured-client"
# See https://docs.unstructured.io/api-reference/api-services/sdk for more details

import os
import glob
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import unstructured_client
from unstructured_client.models import operations, shared
import backoff
import time

# Replace with your actual API key
API_KEY = os.environ["UNSTRUCTURED_API_KEY"]
SERVER_URL = "https://api.unstructuredapp.io"

client = unstructured_client.UnstructuredClient(
    api_key_auth=API_KEY,
    server_url=SERVER_URL,
)

base_path = os.path.expanduser("~/data/human_table_benchmark")
pdfs = glob.glob(os.path.join(base_path, "**", "*.pdf"), recursive=True)


@backoff.on_exception(backoff.expo, Exception, max_tries=3)
def process_single_file(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()

    start_time = time.time()
    req = operations.PartitionRequest(
        partition_parameters=shared.PartitionParameters(
            files=shared.Files(
                content=data,
                file_name=os.path.basename(file_path),
            ),
            strategy=shared.Strategy.HI_RES,
        ),
    )

    res = client.general.partition(request=req)
    processing_time = time.time() - start_time
    return {"elements": res.elements, "processing_time": processing_time}


def process_pdf(pdf_path: str):
    output_path = pdf_path.replace("pdfs", "unstructured").replace(".pdf", ".json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        result = process_single_file(pdf_path)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
    except Exception as e:
        print(f"Error processing {pdf_path}: {str(e)}")


def process_all_pdfs(pdfs: list[str]):
    max_workers = 10  # Adjust this based on API rate limits

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_pdf, pdf): pdf for pdf in pdfs}

        progress_bar = tqdm(total=len(pdfs), desc="Processing PDFs")

        for _ in as_completed(futures):
            progress_bar.update(1)

        progress_bar.close()

    print(f"Processed {len(pdfs)} PDFs")


def test_single_file(file_path: str):
    try:
        result = process_single_file(file_path)
        print(result)
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")


if __name__ == "__main__":
    process_all_pdfs(pdfs)
