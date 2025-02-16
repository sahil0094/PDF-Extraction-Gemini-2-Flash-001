import os
import glob
import argparse

from convert import html_to_numpy
from grading import table_similarity
import polars as pl

from providers.config import settings


def main(model: str, folder: str, save_to_csv: bool):
    groundtruth = settings.input_dir / "groundtruth"
    scores = []

    html_files = glob.glob(os.path.join(folder, "*.html"))
    for pred_html_path in html_files:
        # The filename might be something like: 10035_png.rf.07e8e5bf2e9ad4e77a84fd38d1f53f38.html
        base_name = os.path.basename(pred_html_path)

        # Build the path to the corresponding ground-truth file
        gt_html_path = os.path.join(groundtruth, base_name)
        if not os.path.exists(gt_html_path):
            continue

        with open(pred_html_path, "r") as f:
            pred_html = f.read()

        with open(gt_html_path, "r") as f:
            gt_html = f.read()

        # Convert HTML -> NumPy arrays
        try:
            pred_array = html_to_numpy(pred_html)
            gt_array = html_to_numpy(gt_html)

            # Compute similarity (0.0 to 1.0)
            score = table_similarity(gt_array, pred_array)
        except Exception as e:
            print(f"Error converting {base_name}: {e}")
            continue

        scores.append((base_name, score))
        print(f"{base_name}: {score:.4f}")

    score_dicts = [{"filename": fname, "accuracy": scr}
                   for fname, scr in scores]
    scores_df = pl.DataFrame(score_dicts)

    # Clean up filename in scores_df by removing .html
    scores_df = scores_df.with_columns(
        pl.col("filename").str.replace(".html", "").alias("filename")
    )

    # Read the token usage CSV
    token_usage_path = os.path.join(
        os.getcwd(), 'results', f'{model}_token_usage.csv')
    token_df = pl.read_csv(token_usage_path)

    # Clean up pdf_name in token_df
    token_df = token_df.with_columns(
        pl.col("pdf_name").str.replace(".pdf", "").alias("filename")
    ).drop("pdf_name")  # drop the original pdf_name column

    # Merge the dataframes
    merged_df = scores_df.join(
        token_df,
        on="filename",
        how="left"
    )
    # Reorder columns to put accuracy at the end
    column_order = [
        "filename",
        "pdf_path",
        "pdf_type",
        "input_tokens",
        "output_tokens",
        "api_latency",
        "total_cost",
        "accuracy"
    ]
    merged_df = merged_df.select(column_order)

    print(
        f"Average accuracy for {model}: {merged_df['accuracy'].mean():.2f} with std {merged_df['accuracy'].std():.2f}"
    )

    if save_to_csv:
        # Create scores directory if it doesn't exists
        os.makedirs("scores", exist_ok=True)
        merged_df.write_csv(
            f"{settings.output_dir}/scores/{model}_accuracy_with_tokens.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--save-to-csv", type=bool, default=True)
    args = parser.parse_args()

    model_dir = settings.output_dir / "outputs"/f"{args.model}-raw"
    # cwd = os.getcwd()
    # print(f'curr {cwd}')
    # print(model_dir)
    assert model_dir.exists(), f"Model directory {model_dir} does not exist"
    main(args.model, model_dir, args.save_to_csv)
