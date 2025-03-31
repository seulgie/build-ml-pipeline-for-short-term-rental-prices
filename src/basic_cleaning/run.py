#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import wandb
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    logger.info(f"Downloading artifact: {args.input_artifact}")
    artifact_local_path = run.use_artifact(args.input_artifact).file()

    # Load data
    df = pd.read_csv(artifact_local_path)

    # Remove outliers (min_price ~ max_price)
    logger.info(f"Filtering prices between {args.min_price} and {args.max_price}")
    df = df[(df["price"] >= args.min_price) & (df["price"] <= args.max_price)].copy()

    # Remove null values
    logger.info("Dropping null values")
    df.dropna(inplace=True)

    # Save clean_sample.csv
    output_file = "clean_sample.csv"
    df.to_csv(output_file, index=False)
    logger.info(f"Saved cleaned data to {output_file}")

    # Upload to W&B
    logger.info("Uploading cleaned dataset to W&B")
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(output_file)
    run.log_artifact(artifact)

    logger.info("Data cleaning step completed successfully!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")

    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="Name of the input artifact to download",
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="Name of the output artifact to create",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help="Type of the output artifact",
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help="Description of the output artifact",
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=float,
        help="Minimum price threshold for filtering",
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=float,
        help="Maximum price threshold for filtering",
        required=True
    )

    args = parser.parse_args()

    go(args)
