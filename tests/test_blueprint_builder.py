"""
Tests for blueprint_builder — plan_dict to Make.com blueprint conversion.
"""

import pytest

from tools.blueprint_builder import (
    build_blueprint,
    build_webhook_trigger,
    build_http_request_module,
    build_router_module,
    build_set_variable_module,
    build_json_parse_module,
    build_slack_module,
    build_email_module,
    build_google_sheets_module,
    extract_modules_from_plan,
    validate_blueprint,
)


class TestModuleBuilders:
    def test_build_webhook_trigger(self):
        """Webhook trigger has correct module type and structure."""
        mod = build_webhook_trigger(label="My Webhook")
        assert mod["module"] == "gateway:CustomWebHook"
        assert mod["id"] == 1
        assert mod["metadata"]["designer"]["name"] == "My Webhook"

    def test_build_webhook_trigger_with_hook_id(self):
        """Webhook trigger includes hook_id in parameters."""
        mod = build_webhook_trigger(label="Hook", hook_id=42)
        assert mod["parameters"]["hook"] == 42

    def test_build_http_request_module(self):
        """HTTP module has correct structure and mapper."""
        mod = build_http_request_module(3, "Fetch Data", url="https://api.test.com", method="POST")
        assert mod["module"] == "http:ActionSendData"
        assert mod["id"] == 3
        assert mod["mapper"]["url"] == "https://api.test.com"
        assert mod["mapper"]["method"] == "POST"

    def test_build_router_module(self):
        """Router module has correct type and routes list."""
        mod = build_router_module(5, "Branch")
        assert mod["module"] == "builtin:BasicRouter"
        assert mod["id"] == 5
        assert "routes" in mod

    def test_build_set_variable_module(self):
        """Set variable module correctly maps variable list."""
        mod = build_set_variable_module(2, "Set Vars", variables=[
            {"name": "api_key", "value": "123"},
        ])
        assert mod["module"] == "util:SetVariable2"
        assert mod["mapper"]["variables"][0]["name"] == "api_key"

    def test_build_json_parse_module(self):
        """JSON parse module includes json mapper field."""
        mod = build_json_parse_module(4, "Parse Response", json_string_var="{{1.body}}")
        assert mod["module"] == "json:ParseJSON"
        assert mod["mapper"]["json"] == "{{1.body}}"

    def test_build_slack_module(self):
        """Slack module has channel and text in mapper."""
        mod = build_slack_module(6, "Notify", channel="#general", message="Hello")
        assert mod["module"] == "slack:ActionCreateMessage"
        assert mod["mapper"]["channel"] == "#general"
        assert mod["mapper"]["text"] == "Hello"

    def test_build_email_module(self):
        """Email module has to, subject, html in mapper."""
        mod = build_email_module(7, "Send Email", to="a@b.com", subject="Hi", body="<p>Hi</p>")
        assert mod["module"] == "email:ActionSendEmail"
        assert mod["mapper"]["to"] == "a@b.com"

    def test_build_google_sheets_module(self):
        """Google Sheets module uses correct action."""
        mod = build_google_sheets_module(8, "Add Row", action="addRow")
        assert mod["module"] == "google-sheets:addRow"


class TestExtractModules:
    def test_extract_webhook_step(self):
        """Step with 'webhook' keyword becomes webhook trigger."""
        plan = {"processing_steps": [
            {"description": "Receive webhook trigger from CRM"},
        ]}
        modules = extract_modules_from_plan(plan)
        assert len(modules) == 1
        assert "gateway" in modules[0]["module"] or "WebHook" in modules[0]["module"]

    def test_extract_http_step(self):
        """Step with 'http' or 'api' keyword becomes HTTP module."""
        plan = {"processing_steps": [
            {"description": "Make HTTP request to external API", "inputs": "https://api.test.com"},
        ]}
        modules = extract_modules_from_plan(plan)
        assert modules[0]["module"] == "http:ActionSendData"

    def test_extract_slack_step(self):
        """Step with 'slack' keyword becomes Slack module."""
        plan = {"processing_steps": [
            {"description": "Send slack notification to channel"},
        ]}
        modules = extract_modules_from_plan(plan)
        assert "slack" in modules[0]["module"]

    def test_extract_email_step(self):
        """Step with 'email' keyword becomes email module."""
        plan = {"processing_steps": [
            {"description": "Send email notification to user"},
        ]}
        modules = extract_modules_from_plan(plan)
        assert "email" in modules[0]["module"]

    def test_extract_mixed_steps(self):
        """Multiple step types map correctly."""
        plan = {"processing_steps": [
            {"description": "Webhook trigger"},
            {"description": "Fetch data from API"},
            {"description": "Parse JSON response"},
            {"description": "Send slack message"},
        ]}
        modules = extract_modules_from_plan(plan)
        assert len(modules) == 4
        assert "gateway" in modules[0]["module"] or "WebHook" in modules[0]["module"]
        assert modules[1]["module"] == "http:ActionSendData"
        assert modules[2]["module"] == "json:ParseJSON"
        assert "slack" in modules[3]["module"]

    def test_extract_empty_steps(self):
        """Empty processing_steps returns empty list."""
        assert extract_modules_from_plan({}) == []
        assert extract_modules_from_plan({"processing_steps": []}) == []

    def test_sequential_ids(self):
        """Module IDs are sequential starting from 1."""
        plan = {"processing_steps": [
            {"description": "Step A"},
            {"description": "Step B"},
            {"description": "Step C"},
        ]}
        modules = extract_modules_from_plan(plan)
        assert [m["id"] for m in modules] == [1, 2, 3]


class TestBuildBlueprint:
    def test_build_blueprint_basic(self):
        """build_blueprint produces valid structure."""
        plan = {"processing_steps": [
            {"description": "Receive webhook"},
            {"description": "Fetch from API"},
        ]}
        bp = build_blueprint(plan, "Test Scenario")
        assert bp["name"] == "Test Scenario"
        assert "flow" in bp
        assert len(bp["flow"]) >= 2
        assert "metadata" in bp

    def test_build_blueprint_empty_plan(self):
        """Empty plan gets a placeholder webhook trigger."""
        bp = build_blueprint({}, "Empty Scenario")
        assert len(bp["flow"]) >= 1
        first = bp["flow"][0]
        assert "gateway" in first["module"] or "WebHook" in first["module"]

    def test_build_blueprint_prepends_trigger(self):
        """If first step isn't a trigger, a webhook trigger is prepended."""
        plan = {"processing_steps": [
            {"description": "Send HTTP request to API"},
        ]}
        bp = build_blueprint(plan, "No Trigger Plan")
        # First module should be the prepended webhook trigger
        assert "gateway" in bp["flow"][0]["module"] or "WebHook" in bp["flow"][0]["module"]
        assert len(bp["flow"]) == 2

    def test_build_blueprint_metadata(self):
        """Blueprint has proper metadata structure."""
        bp = build_blueprint({}, "Meta Test")
        assert bp["metadata"]["version"] == 1
        assert bp["metadata"]["scenario"]["maxErrors"] == 3


class TestValidateBlueprint:
    def test_validate_valid_blueprint(self):
        """Valid blueprint passes validation."""
        bp = {
            "name": "Test",
            "flow": [{"id": 1, "module": "gateway:CustomWebHook"}],
        }
        is_valid, errors = validate_blueprint(bp)
        assert is_valid is True
        assert errors == []

    def test_validate_missing_name(self):
        """Blueprint without name fails validation."""
        bp = {"flow": [{"id": 1, "module": "test"}]}
        is_valid, errors = validate_blueprint(bp)
        assert is_valid is False
        assert any("name" in e for e in errors)

    def test_validate_missing_flow(self):
        """Blueprint without flow fails validation."""
        bp = {"name": "Test"}
        is_valid, errors = validate_blueprint(bp)
        assert is_valid is False
        assert any("flow" in e for e in errors)

    def test_validate_empty_flow(self):
        """Blueprint with empty flow fails validation."""
        bp = {"name": "Test", "flow": []}
        is_valid, errors = validate_blueprint(bp)
        assert is_valid is False

    def test_validate_module_missing_type(self):
        """Module without 'module' field fails validation."""
        bp = {"name": "Test", "flow": [{"id": 1}]}
        is_valid, errors = validate_blueprint(bp)
        assert is_valid is False
        assert any("module" in e.lower() for e in errors)
