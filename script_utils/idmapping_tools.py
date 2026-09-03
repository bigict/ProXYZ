import os
import pickle

import click
import lmdb

from proxyz.data.utils import lines, opener
from proxyz.utils import dict2object


@click.command(context_settings={"show_default": True})
@click.option(
    "--output_dir", type=str, default=None, required=True, help="output dir",
)
@click.option(
    "--idmapping_file", type=str, default="-", help="idmapping file",
)
@click.option(
    "--idmapping_type",
    type=click.Choice(["UniRef90", "UniRef50"]),
    default="UniRef50",
    help="filter IDs by ID_type.",
)
@click.option(
    "--key_format", type=str, default=None, help="reformat key.",
)
@click.option("-v", "--verbose", is_flag=True, help="verbose output.")
def main(**args):
    args = dict2object(**args)

    if args.verbose:
        print(f"Write idmapping to lmdb: {args.output_dir}")

    if os.path.exists(args.output_dir):
        print(f"Error: {args.output_dir} exits.")
        return -1

    with lmdb.open(args.output_dir, map_size=1<<36, max_dbs=1) as env:
        db = env.open_db(b"cluster_map", dupsort=True)
        with env.begin(write=True, db=db) as txn:
            with opener(args.idmapping_file) as f:
                for key, id_type, value in map(lambda x: x.split("\t"), lines(f)):
                    if id_type == args.idmapping_type:
                        if args.key_format is not None:
                            key = args.key_format % (key)
                        txn.put(pickle.dumps(value), pickle.dumps(key))

        with env.begin(write=False, db=db) as txn:
            cursor = txn.cursor(db=db)
            n = sum(1 for _ in cursor.iternext_nodup(values=False))

        if args.verbose:
            print(f"Write idmapping to lmdb: {n}")


if __name__ == "__main__":
    main()
