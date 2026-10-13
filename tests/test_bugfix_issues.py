import io
import json
import logging
from pathlib import Path
import tempfile
from typing import Any, cast, get_args

import pytest
from typer.testing import CliRunner

from azure_functions_doctor import logging_config
from azure_functions_doctor.cli import cli as app
from azure_functions_doctor.handlers import (
    Rule,
    _discover_functionapp_aliases,
    _parse_requirements_names,
    _source_contains_ast,
    generic_handler,
)


def _make_rule(rule_type: str, condition: dict[str, str]) -> Rule:
    return cast(Rule, {"type": rule_type, "condition": condition})


def _reset_main_logger() -> logging.Logger:
    logger = logging.getLogger(logging_config.DEFAULT_LOGGER_NAME)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    logger.setLevel(logging.NOTSET)
    logger.propagate = True
    return logger


def test_rule_typed_literal_includes_package_declared() -> None:
    """Issue #100: Rule type Literal includes package_declared."""
    rule_type_literal = Rule.__annotations__["type"]
    assert "package_declared" in get_args(rule_type_literal)


def test_summary_json_writes_counts_file(tmp_path: Path) -> None:
    """Issue #95: --summary-json writes passed/warned/failed/skipped keys."""
    runner = CliRunner()
    summary_path = tmp_path / "summary.json"

    result = runner.invoke(
        app,
        ["doctor", "--path", str(tmp_path), "--summary-json", str(summary_path)],
    )

    assert result.exit_code in (0, 1)
    assert summary_path.exists()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert set(payload.keys()) == {"passed", "warned", "failed", "skipped"}


def test_summary_json_sidecar_created_with_json_format(tmp_path: Path) -> None:
    """Issue #95: sidecar summary file is still created with --format json."""
    runner = CliRunner()
    summary_path = tmp_path / "counts.json"

    result = runner.invoke(
        app,
        [
            "doctor",
            "--path",
            str(tmp_path),
            "--format",
            "json",
            "--summary-json",
            str(summary_path),
        ],
    )

    assert result.exit_code in (0, 1)
    assert summary_path.exists()


def test_summary_json_counts_are_integers(tmp_path: Path) -> None:
    """Issue #95: summary count values are integers."""
    runner = CliRunner()
    summary_path = tmp_path / "summary-counts.json"

    result = runner.invoke(
        app,
        ["doctor", "--path", str(tmp_path), "--summary-json", str(summary_path)],
    )

    assert result.exit_code in (0, 1)
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert isinstance(payload["passed"], int)
    assert isinstance(payload["warned"], int)
    assert isinstance(payload["failed"], int)


_V2_FUNCTION_APP = """import azure.functions as func

app = func.FunctionApp()


@app.route(route="hello")
def hello(req: func.HttpRequest) -> str:
    return "ok"
"""

# Minimal custom ruleset: one deterministic file-existence check. Together with
# the v2 function_app.py fixture this pins the finding-driven exit status
# (host.json present -> pass/exit 0, absent -> fail/exit 1) without depending
# on the machine's Python environment.
_HOST_JSON_RULE = [
    {
        "id": "check_host_json_exists",
        "category": "project_structure",
        "section": "project_structure",
        "label": "host.json",
        "description": "Checks if host.json exists at the project root.",
        "type": "file_exists",
        "required": True,
        "condition": {"target": "host.json"},
        "check_order": 0,
        "hint": "Add host.json to configure the Azure Functions host runtime.",
    }
]


def _invoke_doctor_with_summary(project: Path, rules_path: Path, summary_path: Path) -> Any:
    runner = CliRunner()
    return runner.invoke(
        app,
        [
            "doctor",
            "--path",
            str(project),
            "--format",
            "json",
            "--rules",
            str(rules_path),
            "--summary-json",
            str(summary_path),
        ],
    )


@pytest.mark.parametrize("exc_type", [OSError, PermissionError])
@pytest.mark.parametrize("host_json_present", [True, False])
@pytest.mark.parametrize("failing_op", ["parent_mkdir", "sidecar_write"])
def test_summary_sidecar_failure_stays_non_fatal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failing_op: str,
    host_json_present: bool,
    exc_type: type[OSError],
) -> None:
    """Issue #449: a failing --summary-json sidecar never disturbs the report.

    The sidecar is optional: its mkdir/write failures are caught and logged
    while the primary report continues with the same finding-driven exit
    status as the success path. Only the sidecar operation is mocked to fail;
    no real filesystem permissions are changed.
    """
    project = tmp_path / "project"
    project.mkdir()
    (project / "function_app.py").write_text(_V2_FUNCTION_APP, encoding="utf-8")
    if host_json_present:
        (project / "host.json").write_text("{}", encoding="utf-8")
    rules_path = tmp_path / "rules.json"
    rules_path.write_text(json.dumps(_HOST_JSON_RULE), encoding="utf-8")
    summary_path = project / "nested" / "summary.json"

    # Baseline run with a working sidecar pins the expected outcome.
    baseline = _invoke_doctor_with_summary(project, rules_path, summary_path)
    expected_exit = 0 if host_json_present else 1
    assert baseline.exit_code == expected_exit
    assert summary_path.is_file()
    baseline_results = json.loads(baseline.stdout)["results"]
    summary_path.unlink()

    # Fail only the sidecar operation, and only for the sidecar's own path.
    if failing_op == "parent_mkdir":
        original_mkdir = Path.mkdir

        def fake_mkdir(self: Path, *args: Any, **kwargs: Any) -> None:
            if self == summary_path.parent:
                raise exc_type("sidecar parent denied")
            original_mkdir(self, *args, **kwargs)

        monkeypatch.setattr(Path, "mkdir", fake_mkdir)
    else:
        original_write_text = Path.write_text

        def fake_write_text(self: Path, *args: Any, **kwargs: Any) -> int:
            if self == summary_path:
                raise exc_type("sidecar write denied")
            return original_write_text(self, *args, **kwargs)

        monkeypatch.setattr(Path, "write_text", fake_write_text)

    # Capture logging separately from the primary report stdout: the app
    # logger sets propagate=False, so attach a dedicated stream handler.
    logger = _reset_main_logger()
    log_stream = io.StringIO()
    logger.addHandler(logging.StreamHandler(log_stream))
    result = _invoke_doctor_with_summary(project, rules_path, summary_path)

    # Primary stdout JSON stays parseable and carries the same findings.
    payload = json.loads(result.stdout)
    assert payload["results"] == baseline_results
    # Same finding-driven exit status as the success path.
    assert result.exit_code == expected_exit == baseline.exit_code
    # No summary file was created by the failed sidecar.
    assert not summary_path.exists()
    # The warning identifies the summary-write failure and reaches logging
    # only — it never leaks into the primary report stdout.
    logged = log_stream.getvalue()
    assert "Failed to write summary JSON" in logged
    assert str(summary_path) in logged
    assert "Failed to write summary JSON" not in result.stdout


def test_callable_detection_no_false_positive_for_functionapp_only() -> None:
    """Issue #97: bare FunctionApp assignment does not trigger callable detection."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        (tmp_path / "function_app.py").write_text(
            "import azure.functions as func\napp = func.FunctionApp()\n",
            encoding="utf-8",
        )
        rule: Rule = {"type": "callable_detection"}

        result = generic_handler(rule, tmp_path)

        assert result["status"] == "skip"


def test_callable_detection_asgi_middleware_passes(tmp_path: Path) -> None:
    """Issue #97: AsgiMiddleware usage triggers callable detection pass."""
    (tmp_path / "main.py").write_text("app = AsgiMiddleware(app)\n", encoding="utf-8")
    rule: Rule = {"type": "callable_detection"}

    result = generic_handler(rule, tmp_path)

    assert result["status"] == "pass"


def test_callable_detection_wsgi_middleware_passes(tmp_path: Path) -> None:
    """Issue #97: WsgiMiddleware usage triggers callable detection pass."""
    (tmp_path / "main.py").write_text("app = WsgiMiddleware(app)\n", encoding="utf-8")
    rule: Rule = {"type": "callable_detection"}

    result = generic_handler(rule, tmp_path)

    assert result["status"] == "pass"


def test_callable_detection_asgi_function_app_passes(tmp_path: Path) -> None:
    """Issue #97: AsgiFunctionApp usage triggers callable detection pass."""
    (tmp_path / "main.py").write_text("app = AsgiFunctionApp()\n", encoding="utf-8")
    rule: Rule = {"type": "callable_detection"}

    result = generic_handler(rule, tmp_path)

    assert result["status"] == "pass"


def test_callable_detection_fastapi_unexposed_fails(tmp_path: Path) -> None:
    """#307: FastAPI present but no exposure => warn (fail)."""
    (tmp_path / "main.py").write_text(
        "from fastapi import FastAPI\napp = FastAPI()\n",
        encoding="utf-8",
    )
    rule: Rule = {"type": "callable_detection"}

    result = generic_handler(rule, tmp_path)

    assert result["status"] == "fail"


def test_callable_detection_flask_unexposed_fails(tmp_path: Path) -> None:
    """#307: Flask present but no exposure => warn (fail)."""
    (tmp_path / "main.py").write_text(
        "from flask import Flask\napp = Flask(__name__)\n",
        encoding="utf-8",
    )
    rule: Rule = {"type": "callable_detection"}

    result = generic_handler(rule, tmp_path)

    assert result["status"] == "fail"


def test_setup_logging_reinvocation_updates_log_level() -> None:
    """Issue #99: second setup_logging call updates logger and handler levels."""
    logger = _reset_main_logger()
    first = logging_config.setup_logging(level="INFO", format_style="simple")
    second = logging_config.setup_logging(level="DEBUG", format_style="simple")

    assert first is logger
    assert second is logger
    assert logger.level == logging.DEBUG
    assert all(handler.level == logging.DEBUG for handler in logger.handlers)


def test_setup_logging_reinvocation_updates_formatter() -> None:
    """Issue #99: second setup_logging call updates formatter style."""
    logger = _reset_main_logger()
    logging_config.setup_logging(level="INFO", format_style="simple")
    initial_format = logger.handlers[0].formatter._fmt if logger.handlers[0].formatter else ""

    logging_config.setup_logging(level="INFO", format_style="structured")
    updated_format = logger.handlers[0].formatter._fmt if logger.handlers[0].formatter else ""

    assert initial_format == "%(levelname)s: %(message)s"
    assert updated_format == "%(asctime)s [%(levelname)8s] %(name)s: %(message)s"


def test_parse_requirements_names_handles_extras() -> None:
    """Issue #98: parser extracts package name from extras syntax."""
    names = _parse_requirements_names("requests[security]>=2.0")
    assert names == {"requests"}


def test_parse_requirements_names_handles_environment_markers() -> None:
    """Issue #98: parser handles environment markers."""
    names = _parse_requirements_names("azure-functions;python_version>='3.8'")
    assert names == {"azure-functions"}


def test_parse_requirements_names_handles_direct_url() -> None:
    """Issue #98: parser handles package URL requirements."""
    names = _parse_requirements_names("mypackage @ https://example.com/mypackage.tar.gz")
    assert names == {"mypackage"}


def test_parse_requirements_names_skips_requirements_include() -> None:
    """Issue #98: parser skips -r include directives."""
    names = _parse_requirements_names("-r other.txt\nazure-functions")
    assert names == {"azure-functions"}


def test_parse_requirements_names_handles_editable_egg() -> None:
    """Issue #98: parser extracts package from editable #egg syntax."""
    names = _parse_requirements_names("-e git+https://github.com/user/repo#egg=mylib")
    assert names == {"mylib"}


def test_parse_requirements_names_skips_comments_and_blanks() -> None:
    """Issue #98: parser skips comments and blank lines."""
    names = _parse_requirements_names("# comment\nazure-functions\n\n")
    assert names == {"azure-functions"}


def test_parse_requirements_names_normalizes_lowercase() -> None:
    """Issue #98: parser normalizes package names to lowercase."""
    names = _parse_requirements_names("Azure-Functions==1.0\nazure-functions>=2.0")
    assert names == {"azure-functions"}


def test_parse_requirements_names_handles_inline_comments() -> None:
    """Issue #98: parser ignores inline comments after package specifiers."""
    names = _parse_requirements_names("package1\npackage2  # inline comment")
    assert names == {"package1", "package2"}


def test_parse_requirements_names_handles_utf8_bom_and_crlf() -> None:
    names = _parse_requirements_names("\ufeffazure-functions==1.25.0\r\n")

    assert names == {"azure-functions"}


def test_package_declared_rule_supports_extras_syntax(tmp_path: Path) -> None:
    """Issue #98: package_declared rule matches package declared with extras."""
    (tmp_path / "requirements.txt").write_text(
        "azure-functions[durable]>=1.0\n",
        encoding="utf-8",
    )
    rule = _make_rule(
        "package_declared",
        {"package": "azure-functions", "file": "requirements.txt"},
    )

    result = generic_handler(rule, tmp_path)

    assert result["status"] == "pass"


def test_package_declared_resolves_nested_relative_requirement_includes(tmp_path: Path) -> None:
    requirements = tmp_path / "requirements"
    requirements.mkdir()
    (tmp_path / "requirements.txt").write_text("-r requirements/base.txt\n", encoding="utf-8")
    (requirements / "base.txt").write_text("--requirement=shared.txt\n", encoding="utf-8")
    (requirements / "shared.txt").write_text(
        "azure-functions==1.25.0\n-r base.txt\n",
        encoding="utf-8",
    )
    rule = _make_rule(
        "package_declared",
        {"package": "azure-functions", "file": "requirements.txt"},
    )

    result = generic_handler(rule, tmp_path)

    assert result["status"] == "pass"


def test_package_declared_ignores_remote_and_constraint_includes(tmp_path: Path) -> None:
    (tmp_path / "requirements.txt").write_text(
        "-r https://example.com/requirements.txt\n-c constraints.txt\n",
        encoding="utf-8",
    )
    (tmp_path / "constraints.txt").write_text("azure-functions==1.25.0\n", encoding="utf-8")
    rule = _make_rule(
        "package_declared",
        {"package": "azure-functions", "file": "requirements.txt"},
    )

    result = generic_handler(rule, tmp_path)

    assert result["status"] == "fail"


def test_package_declared_stops_at_requirement_include_depth_limit(tmp_path: Path) -> None:
    for depth in range(12):
        content = f"-r {depth + 1}.txt\n" if depth < 11 else "azure-functions==1.25.0\n"
        (tmp_path / f"{depth}.txt").write_text(content, encoding="utf-8")
    rule = _make_rule(
        "package_declared",
        {"package": "azure-functions", "file": "0.txt"},
    )

    result = generic_handler(rule, tmp_path)

    assert result["status"] == "fail"


def test_package_forbidden_rule_supports_extras_syntax(tmp_path: Path) -> None:
    """Issue #98: package_forbidden rule detects forbidden package with extras."""
    (tmp_path / "requirements.txt").write_text(
        "azure-functions-worker[grpc]==1.0\n",
        encoding="utf-8",
    )
    rule = _make_rule(
        "package_forbidden",
        {"package": "azure-functions-worker", "file": "requirements.txt"},
    )

    result = generic_handler(rule, tmp_path)

    assert result["status"] == "fail"


def test_discover_functionapp_aliases_detects_app_alias() -> None:
    """Issue #96: alias discovery detects standard app assignment."""
    aliases = _discover_functionapp_aliases("app = func.FunctionApp()")
    assert aliases == {"app"}


def test_discover_functionapp_aliases_detects_custom_alias() -> None:
    """Issue #96: alias discovery detects custom FunctionApp alias."""
    aliases = _discover_functionapp_aliases("fa = func.FunctionApp()")
    assert aliases == {"fa"}


def test_discover_functionapp_aliases_detects_blueprint_alias() -> None:
    """Issue #96: alias discovery detects Blueprint alias."""
    aliases = _discover_functionapp_aliases("bp = Blueprint()")
    assert aliases == {"bp"}


def test_discover_functionapp_aliases_falls_back_when_no_assignment() -> None:
    """Issue #96: alias discovery falls back to default when no aliases exist."""
    aliases = _discover_functionapp_aliases("x = 1")
    assert aliases == {"app"}


def test_discover_functionapp_aliases_falls_back_on_syntax_error() -> None:
    """Issue #96: alias discovery falls back to default on syntax error."""
    aliases = _discover_functionapp_aliases("def (broken")
    assert aliases == {"app"}


def test_source_contains_ast_autodiscovers_custom_alias() -> None:
    """Issue #96: AST source detection auto-discovers FunctionApp custom alias."""
    source = "fa = func.FunctionApp()\n@fa.route()\ndef main(req):\n    return 'ok'\n"
    assert _source_contains_ast(source, "app") is True


def test_source_contains_ast_detects_standard_app_decorator() -> None:
    """Issue #96: AST source detection matches @app.route decorators."""
    source = "@app.route()\ndef main(req):\n    return 'ok'\n"
    assert _source_contains_ast(source, "app") is True


def test_source_contains_ast_returns_false_without_decorators() -> None:
    """Issue #96: AST source detection returns False without decorators."""
    source = "x = 1\n"
    assert _source_contains_ast(source, "app") is False
