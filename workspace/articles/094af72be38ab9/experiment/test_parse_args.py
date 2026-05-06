import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load completejourney parquet datasets via FileIOUtils.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Project root. Defaults to the nearest parent containing pyproject.toml.",
    )
    parser.add_argument(
        "--dataset-yaml",
        type=Path,
        default=None,
        help="Dataset definition YAML. Defaults to shared/py/myproj/conf/dataset/completejourney.yaml.",
    )
    parser.add_argument(
        "--entries",
        nargs="*",
        default=None,
        help="Logical dataset names to load. Defaults to all entries in the YAML.",
    )
    parser.add_argument(
        "--use-dask",
        action="store_true",
        help="Load parquet files as dask dataframes instead of pandas dataframes.",
    )
    return parser.parse_args()



def main(): 
    args = parse_args()

    print(args.project_root)
    print(args.dataset_yaml)
    print(args.entries)
    print(args.use_dask)


if __name__ == "__main__":
    main()
