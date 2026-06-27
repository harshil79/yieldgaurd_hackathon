"""One-time fetch of the UCI SECOM dataset into dataset/secom.csv.

Run once; train.py reads from the local CSV after that.
"""
from ucimlrepo import fetch_ucirepo

OUTPUT_PATH = "dataset/secom.csv"


def main():
    secom = fetch_ucirepo(id=179)
    secom.data.original.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {secom.data.original.shape} to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
