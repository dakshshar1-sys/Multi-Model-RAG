"""requirements-ci.txt must stay = requirements.txt minus the ML runtime (it is generated, never edited)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import make_ci_requirements as m


def test_ci_requirements_are_in_sync_with_requirements():
    wanted = m.render(open(m.SRC, encoding="utf-8").read())
    assert open(m.OUT, encoding="utf-8").read() == wanted, "stale: run `python scripts/make_ci_requirements.py`"


def test_heavy_packages_and_pip_options_are_left_out_and_pins_are_kept():
    out = m.render("--extra-index-url https://download.pytorch.org/whl/cpu\ntorch==2.14.0+cpu\ntorchvision==0.29.0+cpu\n"
                   "fastapi==0.110.0\npypdfium2==5.13.0   # renders PDF pages\nsentence-transformers==2.5.1\nfaiss-cpu==1.8.0\neasyocr==1.7.1\n\n# a comment\npytest==8.1.1\n")
    body = [l for l in out.splitlines() if l and not l.startswith("#")]
    assert body == ["fastapi==0.110.0", "pypdfium2==5.13.0", "pytest==8.1.1"]


def test_package_name_parsing():
    assert m.package_name("Sentence_Transformers==2.5.1") == "sentence-transformers"
    assert m.package_name("uvicorn[standard]>=0.29 ; python_version>'3.8'") == "uvicorn"
