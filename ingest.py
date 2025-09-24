from db import ingest_from_data_dir
import argparse
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--persist_dir", default=None)
    args = parser.parse_args()
    ingest_from_data_dir(data_dir=args.data_dir, persist_dir=args.persist_dir)