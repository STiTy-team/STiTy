import argparse


def parse(arguments: list[dict], *, prog: str,
          description: str | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog=prog, description=description)
    for spec in arguments:
        options = dict(spec)
        name = options.pop("name")
        options.setdefault("required", "default" not in options)
        parser.add_argument(f"--{name}", **options)
    return parser.parse_args()
