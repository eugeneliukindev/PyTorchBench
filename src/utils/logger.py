import logging
import sys
from os import PathLike
from pathlib import Path
from typing import Literal


def configure_logger(
    level: (
        Literal["INFO", "DEBUG", "WARNING", "WARN", "ERROR", "CRITICAL", "FATAL", "NOTSET"]
        | Literal[0, 10, 20, 30, 40, 50]
    ),
    filename: str | PathLike[str],
) -> None:
    filename = Path(filename)
    filename.write_text("-" * 80 + "\n")

    logging.basicConfig(
        level=level,
        datefmt="%Y-%m-%d %H:%M:%S",
        format="[%(asctime)s.%(msecs)03d] %(module)s:%(lineno)d %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stderr), logging.FileHandler(filename)],
    )

    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
