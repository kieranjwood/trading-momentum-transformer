import argparse
import datetime as dt
import logging
import os

import quandl

from settings.default import ALL_QUANDL_CODES

DEPTH = 1


def main(api_key: str):
    logging.basicConfig(format="%(asctime)s %(levelname)-8s [%(module)s:%(lineno)d] %(message)s", level=logging.INFO)
    quandl.ApiConfig.api_key = api_key

    if not os.path.exists(os.path.join("data", "quandl")):
        os.mkdir(os.path.join("data", "quandl"))

    for i, t in enumerate(ALL_QUANDL_CODES):
        logging.info(f"Downloading ticker {t} {i} of {len(ALL_QUANDL_CODES)}")
        try:
            data = quandl.get(f"{t}{DEPTH}", start_date="1988-01-01")
            if ("Settle" in data.columns) and (data.index.min() <= dt.datetime(2015, 1, 1)):
                file_path = os.path.join("data", "quandl", f"{t.split('/')[-1]}.csv")
                logging.info(f"Writing ticker {t} to {file_path}")
                data[["Settle"]].to_csv(file_path)
        except BaseException as ex:
            logging.error(f"Failed to download ticker {t}: {ex}")


if __name__ == "__main__":
    def get_args():
        """Download the Quandl data"""

        parser = argparse.ArgumentParser(description="Download the Quandl data.")
        parser.add_argument(
            "api_key",
            metavar="k",
            type=str,
            nargs="?",
            help="quandl API key",
        )

        args = parser.parse_known_args()[0]

        return args.api_key,


    main(*get_args())
