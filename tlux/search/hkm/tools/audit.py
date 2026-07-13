"""Validate an HKM index and print a machine-readable result."""

from __future__ import annotations

import argparse
import json

from ..search.searcher import audit_index


# Validate one index and print a concise result.
#
# Arguments:
#   None.
#
# Returns:
#   (None): Prints the resolved index root.
#
def main() -> None:
    parser = argparse.ArgumentParser(description="Audit HKM index artifacts.")
    parser.add_argument("index_root")
    args = parser.parse_args()
    root = audit_index(args.index_root)
    print(json.dumps({"audit": "PASS", "index_root": str(root)}))


if __name__ == "__main__":
    main()
