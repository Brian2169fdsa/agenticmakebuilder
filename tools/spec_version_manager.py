"""
Spec Version Manager

Handles versioned storage of canonical specs and build artifacts
in the output directory.

Structure:
  /output/
    index.json                    # Global scenario index
    <scenario_slug>/
      index.json                  # Per-scenario version index
      v1/
        canonical_spec.json
        validation_report.json
        confidence.json
        build_log.md
      v2/
        ...

Input: slug, canonical_spec, validation_report, confidence, base_output_dir
Output: dict with version, output_path, files_written

Deterministic. No network calls. No conversation context.
"""

import json
import os
from datetime import datetime


def save_versioned_spec(slug, canonical_spec, validation_report, confidence,
                        build_log_md=None, base_output_dir=None):
    """Save a canonical spec and its artifacts to a versioned output directory.

    Args:
        slug: Scenario slug (kebab-case).
        canonical_spec: The canonical spec dict.
        validation_report: Validation results dict.
        confidence: Confidence score dict.
        build_log_md: Optional build log markdown string.
        base_output_dir: Base output directory. Defaults to /output relative to project root.

    Returns:
        dict with:
            - version: int — the version number assigned
            - output_path: str — full path to the version directory
            - files_written: list[str] — filenames written
            - index_updated: bool
    """
    if base_output_dir is None:
        base_output_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "output"
        )

    scenario_dir = os.path.join(base_output_dir, slug)

    # Determine next version
    scenario_index = _load_scenario_index(scenario_dir)
    if scenario_index:
        next_version = scenario_index.get("latest_version", 0) + 1
    else:
        next_version = 1

    version_dir = os.path.join(scenario_dir, f"v{next_version}")
    os.makedirs(version_dir, exist_ok=True)

    files_written = []

    # Write canonical_spec.json
    _write_json(os.path.join(version_dir, "canonical_spec.json"), canonical_spec)
    files_written.append("canonical_spec.json")

    # Write validation_report.json
    _write_json(os.path.join(version_dir, "validation_report.json"), validation_report)
    files_written.append("validation_report.json")

    # Write confidence.json
    _write_json(os.path.join(version_dir, "confidence.json"), confidence)
    files_written.append("confidence.json")

    # Write build_log.md
    if build_log_md is None:
        build_log_md = _generate_build_log(
            slug, next_version, canonical_spec, validation_report, confidence
        )
    with open(os.path.join(version_dir, "build_log.md"), "w") as f:
        f.write(build_log_md)
    files_written.append("build_log.md")

    # Update per-scenario index
    _update_scenario_index(scenario_dir, slug, canonical_spec, confidence, next_version)

    # Update global index
    _update_global_index(base_output_dir, slug, canonical_spec, next_version)

    return {
        "version": next_version,
        "output_path": version_dir,
        "files_written": files_written,
        "index_updated": True
    }


def get_latest_version(slug, base_output_dir=None):
    """Get the latest version number for a scenario.

    Args:
        slug: Scenario slug.
        base_output_dir: Base output directory.

    Returns:
        int or None if scenario doesn't exist.
    """
    if base_output_dir is None:
        base_output_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "output"
        )
    scenario_dir = os.path.join(base_output_dir, slug)
    idx = _load_scenario_index(scenario_dir)
    if idx:
        return idx.get("latest_version")
    return None


def load_version_artifacts(slug, version, base_output_dir=None):
    """Load all artifacts for a specific version.

    Args:
        slug: Scenario slug.
        version: Version number.
        base_output_dir: Base output directory.

    Returns:
        dict with canonical_spec, validation_report, confidence, build_log
        or None if version doesn't exist.
    """
    if base_output_dir is None:
        base_output_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "output"
        )
    version_dir = os.path.join(base_output_dir, slug, f"v{version}")

    if not os.path.isdir(version_dir):
        return None

    result = {}
    spec_path = os.path.join(version_dir, "canonical_spec.json")
    if os.path.exists(spec_path):
        with open(spec_path, "r") as f:
            result["canonical_spec"] = json.load(f)

    report_path = os.path.join(version_dir, "validation_report.json")
    if os.path.exists(report_path):
        with open(report_path, "r") as f:
            result["validation_report"] = json.load(f)

    conf_path = os.path.join(version_dir, "confidence.json")
    if os.path.exists(conf_path):
        with open(conf_path, "r") as f:
            result["confidence"] = json.load(f)

    log_path = os.path.join(version_dir, "build_log.md")
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            result["build_log"] = f.read()

    return result


def _write_json(path, data):
    """Write a dict as formatted JSON."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def _load_scenario_index(scenario_dir):
    """Load the per-scenario index.json if it exists."""
    idx_path = os.path.join(scenario_dir, "index.json")
    if os.path.exists(idx_path):
        with open(idx_path, "r") as f:
            return json.load(f)
    return None


def _update_scenario_index(scenario_dir, slug, canonical_spec, confidence, version):
    """Create or update the per-scenario index.json."""
    idx_path = os.path.join(scenario_dir, "index.json")
    now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    if os.path.exists(idx_path):
        with open(idx_path, "r") as f:
            idx = json.load(f)
    else:
        idx = {
            "slug": slug,
            "name": canonical_spec.get("scenario", {}).get("name", slug),
            "versions": [],
            "latest_version": 0
        }

    idx["versions"].append({
        "version": version,
        "created_at": now,
        "confidence_score": confidence.get("score", 0.0),
        "validation_passed": confidence.get("factors", {}).get("validation_error_penalty", 0) == 0.0,
        "original_request": canonical_spec.get("metadata", {}).get("original_request", "")
    })
    idx["latest_version"] = version

    _write_json(idx_path, idx)


def _update_global_index(base_output_dir, slug, canonical_spec, version):
    """Create or update the global output/index.json."""
    idx_path = os.path.join(base_output_dir, "index.json")
    now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    if os.path.exists(idx_path):
        with open(idx_path, "r") as f:
            idx = json.load(f)
    else:
        idx = {"scenarios": {}, "total_scenarios": 0, "total_versions": 0}

    if slug in idx["scenarios"]:
        entry = idx["scenarios"][slug]
        entry["latest_version"] = version
        entry["versions"].append(version)
        entry["updated_at"] = now
    else:
        idx["scenarios"][slug] = {
            "name": canonical_spec.get("scenario", {}).get("name", slug),
            "latest_version": version,
            "versions": [version],
            "created_at": now,
            "updated_at": now
        }

    idx["total_scenarios"] = len(idx["scenarios"])
    idx["total_versions"] = sum(len(s["versions"]) for s in idx["scenarios"].values())

    _write_json(idx_path, idx)


def _generate_build_log(slug, version, canonical_spec, validation_report, confidence):
    """Generate a human-readable build log in Markdown."""
    scenario = canonical_spec.get("scenario", {})
    metadata = canonical_spec.get("metadata", {})
    modules = canonical_spec.get("modules", [])
    trigger = canonical_spec.get("trigger", {})

    lines = [
        f"# Build Log: {scenario.get('name', slug)}",
        f"",
        f"**Slug:** {slug}",
        f"**Version:** {version}",
        f"**Built:** {datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')}",
        f"",
        f"## Original Request",
        f"",
        f"> {metadata.get('original_request', 'N/A')}",
        f"",
        f"## Scenario Structure",
        f"",
        f"- **Trigger:** {trigger.get('label', 'N/A')} ({trigger.get('module', 'N/A')})",
        f"- **Modules:** {len(modules)}",
        f"- **Connections:** {len(canonical_spec.get('connections', []))}",
        f"",
    ]

    if modules:
        lines.append("### Modules")
        lines.append("")
        for mod in modules:
            lines.append(f"- [{mod.get('id')}] {mod.get('label')} ({mod.get('module')})")
        lines.append("")

    lines.extend([
        f"## Validation",
        f"",
        f"- **Checks Run:** {validation_report.get('checks_run', 0)}",
        f"- **Checks Passed:** {validation_report.get('checks_passed', 0)}",
        f"- **Errors:** {len(validation_report.get('errors', []))}",
        f"- **Warnings:** {len(validation_report.get('warnings', []))}",
        f"",
        f"## Confidence",
        f"",
        f"- **Score:** {confidence.get('score', 0.0)}",
        f"- **Grade:** {confidence.get('grade', 'N/A')}",
        f"- **{confidence.get('explanation', '')}**",
        f"",
    ])

    notes = metadata.get("agent_notes", [])
    if notes:
        lines.append("## Agent Notes")
        lines.append("")
        for note in notes:
            lines.append(f"- {note}")
        lines.append("")

    return "\n".join(lines)


# --- Self-check ---
if __name__ == "__main__":
    import tempfile
    import shutil

    print("=== Spec Version Manager Self-Check ===\n")

    # Use a temporary directory for testing
    test_dir = tempfile.mkdtemp(prefix="amb_test_output_")

    try:
        test_spec = {
            "scenario": {"name": "Test Scenario", "slug": "test-scenario", "description": "Test."},
            "trigger": {"id": 1, "label": "Webhook", "module": "gateway:CustomWebHook"},
            "modules": [
                {"id": 2, "label": "Parse", "module": "json:ParseJSON"},
                {"id": 3, "label": "Post", "module": "slack:PostMessage"}
            ],
            "connections": [
                {"from": "trigger", "to": 2},
                {"from": 2, "to": 3}
            ],
            "metadata": {"original_request": "Test request", "agent_notes": ["Test note"]}
        }
        test_report = {"errors": [], "warnings": [], "checks_run": 30, "checks_passed": 30}
        test_confidence = {"score": 0.95, "grade": "A", "explanation": "Good.", "factors": {"validation_error_penalty": 0.0}}

        # Test 1: Save version 1
        print("Test 1: Save v1")
        result = save_versioned_spec(
            slug="test-scenario",
            canonical_spec=test_spec,
            validation_report=test_report,
            confidence=test_confidence,
            base_output_dir=test_dir
        )
        assert result["version"] == 1
        assert len(result["files_written"]) == 4
        assert os.path.isdir(result["output_path"])
        print(f"  [OK] Version {result['version']} saved to {result['output_path']}")
        print(f"       Files: {result['files_written']}")

        # Test 2: Save version 2
        print("\nTest 2: Save v2")
        result2 = save_versioned_spec(
            slug="test-scenario",
            canonical_spec=test_spec,
            validation_report=test_report,
            confidence=test_confidence,
            base_output_dir=test_dir
        )
        assert result2["version"] == 2
        print(f"  [OK] Version {result2['version']} saved")

        # Test 3: Check latest version
        print("\nTest 3: Get latest version")
        latest = get_latest_version("test-scenario", base_output_dir=test_dir)
        assert latest == 2, f"Expected 2, got {latest}"
        print(f"  [OK] Latest version: {latest}")

        # Test 4: Load artifacts
        print("\nTest 4: Load version artifacts")
        artifacts = load_version_artifacts("test-scenario", 1, base_output_dir=test_dir)
        assert artifacts is not None
        assert "canonical_spec" in artifacts
        assert "validation_report" in artifacts
        assert "confidence" in artifacts
        assert "build_log" in artifacts
        print(f"  [OK] Loaded {len(artifacts)} artifact types")

        # Test 5: Verify global index
        print("\nTest 5: Verify global index")
        with open(os.path.join(test_dir, "index.json"), "r") as f:
            global_idx = json.load(f)
        assert global_idx["total_scenarios"] == 1
        assert global_idx["total_versions"] == 2
        assert "test-scenario" in global_idx["scenarios"]
        print(f"  [OK] Global index: {global_idx['total_scenarios']} scenarios, {global_idx['total_versions']} versions")

        # Test 6: Verify scenario index
        print("\nTest 6: Verify scenario index")
        with open(os.path.join(test_dir, "test-scenario", "index.json"), "r") as f:
            scenario_idx = json.load(f)
        assert scenario_idx["latest_version"] == 2
        assert len(scenario_idx["versions"]) == 2
        print(f"  [OK] Scenario index: {len(scenario_idx['versions'])} versions")

        # Test 7: Nonexistent scenario
        print("\nTest 7: Nonexistent scenario")
        assert get_latest_version("nonexistent", base_output_dir=test_dir) is None
        assert load_version_artifacts("nonexistent", 1, base_output_dir=test_dir) is None
        print("  [OK] Returns None for nonexistent")

    finally:
        shutil.rmtree(test_dir)

    print("\n=== All spec version manager checks passed ===")
