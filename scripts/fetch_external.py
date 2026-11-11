"""Acquire optional research sources at exact commits and verify their license files."""

import hashlib
import json
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
for item in json.loads((root / "third_party/sources.json").read_text())["sources"]:
    destination = root / "external" / item["name"]
    if not destination.exists():
        subprocess.run(["git", "clone", "--no-checkout", item["url"], str(destination)], check=True)
        subprocess.run(["git", "-C", str(destination), "checkout", "--detach", item["commit"]], check=True)
    actual = subprocess.check_output(["git", "-C", str(destination), "rev-parse", "HEAD"], text=True).strip()
    if actual != item["commit"]:
        raise ValueError(f"Existing {item['name']} has different commit; left unchanged")
    if subprocess.check_output(["git", "-C", str(destination), "diff", "HEAD", "--"], text=True):
        raise ValueError(f"Existing {item['name']} has tracked modifications; left unchanged")
    license_file = destination / "LICENSE"
    if hashlib.sha256(license_file.read_bytes()).hexdigest() != item["license_sha256"]:
        raise ValueError(f"License hash mismatch for {item['name']}")
    print(item["name"], actual, "verified")
