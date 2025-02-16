from pdf2image import convert_from_path
from textractor import Textractor
from textractor.data.constants import TextractFeatures
import re
import os
import glob
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
import backoff

extractor = Textractor(region_name="us-west-2")

base_path = os.path.expanduser("~/data/human_table_benchmark")
pdfs = glob.glob(os.path.join(base_path, "**", "*.pdf"), recursive=True)


@backoff.on_exception(
    backoff.expo, Exception, giveup=lambda e: "limit" not in str(e).lower(), max_tries=5
)
def extract_tables(pdf_path: str):
    start_time = time.time()

    image = convert_from_path(pdf_path)[0]

    document = extractor.analyze_document(
        file_source=image,
        features=[TextractFeatures.TABLES],
    )

    if len(document.tables) == 0:
        return None, time.time() - start_time

    html_table = document.tables[0].to_html()

    html_table = re.sub(r"<caption>.*?</caption>", "", html_table, flags=re.DOTALL)

    end_time = time.time()
    processing_time = end_time - start_time

    return html_table, processing_time


def process_pdf(pdf_path: str):
    output_path = pdf_path.replace("pdfs", "textract").replace(".pdf", ".json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        html_table, processing_time = extract_tables(pdf_path)

        if html_table is None:
            result = {"html_table": None, "processing_time": processing_time}
        else:
            result = {"html_table": html_table, "processing_time": processing_time}

        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)

        return pdf_path, None
    except Exception as e:
        return pdf_path, str(e)


def process_all_pdfs(pdfs: list[str]):
    max_workers = 20

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_pdf, pdf): pdf for pdf in pdfs}

        progress_bar = tqdm(total=len(pdfs), desc="Processing PDFs")

        for future in as_completed(futures):
            image_path, error = future.result()
            if error:
                print(f"Error processing {image_path}: {error}")
            progress_bar.update(1)

        progress_bar.close()

    print(f"Processed {len(pdfs)} PDFs")


if __name__ == "__main__":
    process_all_pdfs(pdfs)
