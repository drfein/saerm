import os
import sys
import random

import numpy as np


# Ensure src/ is on sys.path for imports like `from saerm...`
_THIS_DIR = os.path.dirname(__file__)
_SRC_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "src"))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


def pytest_configure():
    # Load environment variables from a .env file at repo root if present
    try:
        from dotenv import load_dotenv  # type: ignore[import-not-found]

        load_dotenv(os.path.abspath(os.path.join(_THIS_DIR, "..", ".env")))
    except Exception:
        env_path = os.path.abspath(os.path.join(_THIS_DIR, "..", ".env"))
        if os.path.exists(env_path):
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if "=" not in line:
                            continue
                        k, v = line.split("=", 1)
                        k = k.strip(); v = v.strip().strip('"').strip("'")
                        os.environ.setdefault(k, v)
            except Exception:
                pass

    random.seed(0)
    np.random.seed(0)
    try:
        import torch

        torch.manual_seed(0)
    except Exception:
        pass


