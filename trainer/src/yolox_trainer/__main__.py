from __future__ import annotations

import logging

from .config import TrainerConfig
from .server import serve


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    serve(TrainerConfig.from_env())


if __name__ == "__main__":
    main()
