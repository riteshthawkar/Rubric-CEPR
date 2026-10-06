PYTHON ?= python3

.PHONY: check test package
check:
	$(PYTHON) scripts/accv_v1.py check
	$(PYTHON) tools/verify_release.py

test:
	$(PYTHON) -m pytest -q

package:
	$(PYTHON) tools/package_release.py --output dist/accv-self-evolving-image-editing-v1.0.0-rc1.tar.gz
