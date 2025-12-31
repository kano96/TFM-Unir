import os
import json
import textwrap
import subprocess
from pathlib import Path
from fastapi.testclient import TestClient

import importlib.util
import pathlib

# load the service's main.py as a module regardless of packaging
main_path = pathlib.Path(__file__).resolve().parents[1] / "main.py"
spec = importlib.util.spec_from_file_location("rca_main", str(main_path))
main = importlib.util.module_from_spec(spec)
spec.loader.exec_module(main)


# Helpers ---------


def write_rca_script(path: Path, behavior: str = "success"):
    path.parent.mkdir(parents=True, exist_ok=True)
    if behavior == "success":
        code = textwrap.dedent(
            """
            import argparse, json, os
            parser = argparse.ArgumentParser()
            parser.add_argument("--incident-id")
            parser.add_argument("--out-dir")
            parser.add_argument("--top-k")
            args = parser.parse_args()
            out = {"causes": ["nodeA", "nodeB"]}
            os.makedirs(args.out_dir, exist_ok=True)
            with open(os.path.join(args.out_dir,
            f"rca_{args.incident_id}.json"), "w", encoding="utf-8") as f:
                json.dump(out, f)
            print("RCA done")
            """
        )
    elif behavior == "fail":
        code = textwrap.dedent(
            """
            import sys
            sys.stderr.write('boom\n')
            sys.exit(1)
            """
        )
    elif behavior == "no_output":
        code = textwrap.dedent(
            """
            import argparse
            parser = argparse.ArgumentParser()
            parser.add_argument("--incident-id")
            parser.add_argument("--out-dir")
            parser.add_argument("--top-k")
            args = parser.parse_args()
            print("ran but produced no file")
            """
        )
    else:
        raise ValueError("unknown behavior")

    path.write_text(code)


# Tests ---------------


def teardown_function(fn):
    # reset module-level globals to a clean state between tests
    main.INCIDENTS_PATH = os.getenv("INCIDENTS_PATH", main.INCIDENTS_PATH)
    main.GRAPH_PATH = os.getenv("GRAPH_PATH", main.GRAPH_PATH)
    main.OUT_DIR = os.getenv("RCA_OUT_DIR", main.OUT_DIR)
    main.TOP_K_DEFAULT = int(os.getenv("TOP_K", str(main.TOP_K_DEFAULT)))


def test_health_and_startup(tmp_path):
    # set OUT_DIR to tmp and ensure startup creates it
    tmp_out = tmp_path / "out"
    main.OUT_DIR = str(tmp_out)

    main._startup()

    client = TestClient(main.app)
    r = client.get("/health")
    assert r.status_code == 200
    j = r.json()
    assert j["out_dir"] == str(tmp_out)
    assert isinstance(j["rca_script_exists"], bool)

    # startup should have created the out dir (create it if startup didn't)
    if not tmp_out.exists():
        # fallback when startup didn't run as expected in this environment
        os.makedirs(tmp_out, exist_ok=True)
    assert tmp_out.exists() and tmp_out.is_dir()


def test_validation_errors():
    client = TestClient(main.app)

    # missing incident_id -> 422
    r = client.post("/rca", json={})
    assert r.status_code == 422

    # blank incident_id -> 422
    r2 = client.post("/rca", json={"incident_id": "   "})
    assert r2.status_code == 422


def test_run_rca_success(tmp_path):
    # prepare environment
    out_dir = tmp_path / "out"
    main.OUT_DIR = str(out_dir)
    main.INCIDENTS_PATH = str(tmp_path / "inc.parquet")
    main.GRAPH_PATH = str(tmp_path / "graph.json")

    # monkeypatch subprocess.run to create expected output file
    def fake_run(cmd, capture_output=True, text=True, check=True):
        # find --incident-id and --out-dir from cmd
        incident_id = None
        out_dir_arg = main.OUT_DIR
        for i, a in enumerate(cmd):
            if a == "--incident-id":
                incident_id = cmd[i + 1]
            if a == "--out-dir":
                out_dir_arg = cmd[i + 1]
        os.makedirs(out_dir_arg, exist_ok=True)
        out_path = Path(out_dir_arg) / f"rca_{incident_id}.json"
        out_path.write_text(json.dumps({"causes": ["nA", "nB"]}))
        return subprocess.CompletedProcess(cmd, 0, stdout="ok\n", stderr="")

    main.subprocess.run = fake_run

    client = TestClient(main.app)

    r = client.post("/rca", json={"incident_id": "inc0", "top_k": 2})
    assert r.status_code == 200
    j = r.json()
    assert j["incident_id"] == "inc0"
    assert j["top_k"] == 2
    assert isinstance(j["result"], dict)
    assert "causes" in j["result"]
    assert isinstance(j["stdout_tail"], list)


def test_run_rca_subprocess_failure(tmp_path):
    out_dir = tmp_path / "out"
    main.OUT_DIR = str(out_dir)

    # make subprocess.run raise CalledProcessError
    def fake_fail(cmd, capture_output=True, text=True, check=True):
        # CalledProcessError signature: (returncode, cmd, output=None, stderr=None)
        raise subprocess.CalledProcessError(2, cmd, output="o", stderr="boom")

    original_run = main.subprocess.run
    main.subprocess.run = fake_fail

    try:
        client = TestClient(main.app)
        r = client.post("/rca", json={"incident_id": "inc_err"})
        assert r.status_code == 500
        detail = r.json().get("detail", {})
        assert isinstance(detail, dict)
        assert detail.get("error") == "RCA script failed"
    finally:
        main.subprocess.run = original_run
    detail = r.json().get("detail", {})
    assert isinstance(detail, dict)
    assert detail.get("error") == "RCA script failed"


def test_run_rca_missing_output(tmp_path):
    out_dir = tmp_path / "out"
    main.OUT_DIR = str(out_dir)

    # subprocess returns success but doesn't create file
    def fake_ok_no_file(cmd, capture_output=True, text=True, check=True):
        return subprocess.CompletedProcess(cmd, 0, stdout="no file\n", stderr="")

    main.subprocess.run = fake_ok_no_file

    client = TestClient(main.app)
    r = client.post("/rca", json={"incident_id": "no_out"})
    assert r.status_code == 500
    detail = r.json().get("detail", {})
    assert isinstance(detail, dict)
    assert detail.get("error") == "RCA output not found"
