import quandl
from settings.default import ALL_QUANDL_CODES
import datetime as dt
import argparse
import os

DEPTH = 1

def download_quandl_data():
    ALL_QUANDL_CODES

def main(api_key: str):
    quandl.ApiConfig.api_key = api_key

    if not os.path.exists("quandl"):
        os.mkdir("quandl")

    for t in ALL_QUANDL_CODES:
        print(t)
        try:
            data = quandl.get(f'{t}{DEPTH}', start_date='1988-01-01',)
        except BaseException as ex:
            print(ex)
        if ("Settle" in data.columns) and (data.index.min() <= dt.datetime(2015,1,1)):
            data[["Settle"]].to_csv(f"quandl/{t.split('/')[-1]}.csv")

if __name__ == "__main__":

    def get_args():
        """Download the quandl data"""

        parser = argparse.ArgumentParser(description="Run changepoint detection module")
        parser.add_argument(
            "api_key",
            metavar="k",
            type=str,
            nargs="?",
            help="quandl API key",
        )

        args = parser.parse_known_args()[0]

        return (
            args.api_key,
        )