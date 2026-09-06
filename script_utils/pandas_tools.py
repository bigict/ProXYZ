import click
import pandas as pd

from proxyz.data.utils import opener
from proxyz.utils import dict2object


@click.command(context_settings={"show_default": True})
@click.argument("csv_files", type=click.Path(), nargs=-1)
@click.option("-v", "--verbose", is_flag=True, help="verbose output.")
def main(**args):
    args = dict2object(**args)

    df_list = []
    for csv_file in args.csv_files:
        if args.verbose:
            print(f"[VERB] Read pd from csv: {csv_file}")
        with opener(csv_file) as f:
            df_list.append(pd.read_csv(f))
    df = pd.concat(df_list, ignore_index=True)
    print("Summpary:")
    print(df.describe())


if __name__ == "__main__":
    main()
