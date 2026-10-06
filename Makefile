PYTHON ?= python3

.PHONY: check test package
check:
	$(PYTHON) scripts/rubric_cepr.py check
	$(PYTHON) tools/verify_release.py

test:
	$(PYTHON) -m pytest -q

package:
	$(PYTHON) tools/package_release.py --output dist/rubric-cepr-v1.0.0-rc2.tar.gz
