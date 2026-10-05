import json
import sys

from .main import app


def main() -> None:
    json.dump(app.openapi(), sys.stdout, indent=2)
    sys.stdout.write("\n")
