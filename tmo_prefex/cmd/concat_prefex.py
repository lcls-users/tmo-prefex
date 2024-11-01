from typing import List
from typing_extensions import Annotated

from pathlib import Path

import h5py
import typer

from ..combine import Batch

def concat_prefex(files: Annotated[List[Path],
                                   typer.Argument(help="h5 files to concat")],
                  out: Annotated[Path, typer.Option("--out", "-o")]
                 ) -> None:

    assert len(files) > 0, "Need at least one file to concat."
    with h5py.File(files[0]) as f:
        batch = Batch.from_h5(f)
    for fname in files[1:]:
        with h5py.File(fname) as f:
            batch.extend( Batch.from_h5(f) )

    with h5py.File(out, "w") as f:
        batch.write_h5(f)

def run():
    typer.run(concat_prefex)
