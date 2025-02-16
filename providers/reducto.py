import os
import glob
import json
import asyncio
import aiohttp
import time
from typing import Tuple
from tqdm import tqdm

base_path = os.path.expanduser("~/data/human_table_benchmark")
pdfs = glob.glob(os.path.join(base_path, "**", "*.pdf"), recursive=True)

API_KEY = os.environ.get("REDUCTO_API_KEY")
if not API_KEY:
    raise ValueError("REDUCTO_API_KEY environment variable is not set")

REDUCTO_URL = "https://platform.reducto.ai"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}


async def upload_pdf(
    pdf_path: str, session: aiohttp.ClientSession, semaphore: asyncio.Semaphore
) -> Tuple[str, str]:
    async with semaphore:
        try:
            upload_url = f"{REDUCTO_URL}/upload"
            with open(pdf_path, "rb") as file:
                data = aiohttp.FormData()
                data.add_field(
                    "file",
                    file,
                    filename=os.path.basename(pdf_path),
                    content_type="application/pdf",
                )

                async with session.post(
                    upload_url, headers=HEADERS, data=data
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return pdf_path, result["file_id"]
                    else:
                        return (
                            pdf_path,
                            f"Error: {response.status}, {await response.text()}",
                        )
        except Exception as e:
            return pdf_path, f"Error: {str(e)}"


async def parse_async(
    file_id: str,
    pdf_path: str,
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
) -> Tuple[str, str]:
    async with semaphore:
        try:
            parse_url = f"{REDUCTO_URL}/parse_async"
            payload = {
                "document_url": f"{file_id}",
                "advanced_options": {
                    "ocr_system": "combined",
                },
            }

            async with session.post(
                parse_url, headers=HEADERS, json=payload
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return pdf_path, result["job_id"]
                else:
                    return (
                        pdf_path,
                        f"Error: {response.status}, {await response.text()}",
                    )
        except Exception as e:
            return pdf_path, f"Error: {str(e)}"


async def poll_job(
    file_id: str, job_id: str, session: aiohttp.ClientSession
) -> Tuple[str, dict]:
    job_url = f"{REDUCTO_URL}/job/{job_id}"
    start_time = time.time()
    while time.time() - start_time < 60 * 10:  # 10 minutes timeout
        try:
            async with session.get(job_url, headers=HEADERS) as response:
                if response.status == 200:
                    job_info = await response.json()
                    if job_info["status"] == "Completed":
                        return file_id, job_info["result"]
                    elif job_info["status"] == "Failed":
                        return file_id, {"error": f"Job failed: {job_info}"}
                else:
                    return file_id, {
                        "error": f"Error polling job: {response.status}, {await response.text()}"
                    }
        except Exception as e:
            return file_id, {"error": str(e)}
        await asyncio.sleep(5)  # Wait 5 seconds before polling again
    return file_id, {"error": "Timeout: Job did not complete within 10 minutes"}


async def process_pdf(
    pdf_path: str,
    session: aiohttp.ClientSession,
    upload_semaphore: asyncio.Semaphore,
    parse_semaphore: asyncio.Semaphore,
) -> Tuple[str, dict]:
    pdf_path, file_id = await upload_pdf(pdf_path, session, upload_semaphore)
    if file_id.startswith("Error"):
        return pdf_path, {"error": file_id}

    file_id, job_id = await parse_async(file_id, pdf_path, session, parse_semaphore)
    if job_id.startswith("Error"):
        return pdf_path, {"error": job_id}

    return await poll_job(pdf_path, job_id, session)


async def process_all_pdfs(pdfs: list[str]):
    upload_semaphore = asyncio.Semaphore(10)
    parse_semaphore = asyncio.Semaphore(10)

    async with aiohttp.ClientSession() as session:
        tasks = [
            process_pdf(pdf, session, upload_semaphore, parse_semaphore) for pdf in pdfs
        ]
        progress_bar = tqdm(total=len(pdfs), desc="Processing PDFs")
        results = []
        for task in asyncio.as_completed(tasks):
            pdf_path, result = await task
            results.append((pdf_path, result))
            progress_bar.update(1)
        progress_bar.close()

    for pdf_path, result in results:
        output_path = pdf_path.replace("pdfs", "reducto_nov1").replace(".pdf", ".json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)

    print(f"Processed {len(results)} PDFs")


async def main():
    await process_all_pdfs(pdfs)


if __name__ == "__main__":
    asyncio.run(main())
