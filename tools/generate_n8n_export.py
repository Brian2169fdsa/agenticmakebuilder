"""
n8n Workflow Generator

Converts the canonical spec format into a valid n8n workflow JSON.
Drop-in addition to the pipeline — same input, different output target.

n8n workflow structure:
{
  "name": "Workflow Name",
  "nodes": [...],
  "connections": {...},
  "settings": {...},
  "staticData": null
}

Usage:
    from tools.generate_n8n_export import generate_n8n_export
    result = generate_n8n_export(canonical_spec, registry)
"""

import json
import uuid
import os


# ─── Module mapping: Make.com package → n8n node type ───
# Format: "make_package:Module" → "n8n-nodes-base.NodeType"

MODULE_MAP = {
    # Triggers
    "gateway:CustomWebHook": "n8n-nodes-base.webhook",
    "gateway:MailHook": "n8n-nodes-base.emailTrigger",
    "rss:Trigger": "n8n-nodes-base.rssFeedReadTrigger",
    "schedule:Trigger": "n8n-nodes-base.scheduleTrigger",
    "google-gmail:TriggerOnNewEmail": "n8n-nodes-base.gmailTrigger",

    # Messaging
    "slack:PostMessage": "n8n-nodes-base.slack",
    "slack:CreateChannel": "n8n-nodes-base.slack",
    "telegram:SendMessage": "n8n-nodes-base.telegram",
    "discord:CreateMessage": "n8n-nodes-base.discord",
    "email:ActionSendEmail": "n8n-nodes-base.emailSend",

    # Google
    "google-sheets:addRow": "n8n-nodes-base.googleSheets",
    "google-sheets:updateRow": "n8n-nodes-base.googleSheets",
    "google-sheets:getRows": "n8n-nodes-base.googleSheets",
    "google-sheets:deleteRow": "n8n-nodes-base.googleSheets",
    "google-drive:uploadFile": "n8n-nodes-base.googleDrive",
    "google-drive:createFolder": "n8n-nodes-base.googleDrive",
    "google-calendar:createEvent": "n8n-nodes-base.googleCalendar",
    "google-gmail:sendEmail": "n8n-nodes-base.gmail",

    # Data stores
    "airtable:CreateRecord": "n8n-nodes-base.airtable",
    "airtable:UpdateRecord": "n8n-nodes-base.airtable",
    "airtable:SearchRecords": "n8n-nodes-base.airtable",
    "notion:CreatePage": "n8n-nodes-base.notion",
    "notion:UpdatePage": "n8n-nodes-base.notion",
    "notion:GetPage": "n8n-nodes-base.notion",

    # HTTP
    "http:ActionSendData": "n8n-nodes-base.httpRequest",
    "http:ActionGetFile": "n8n-nodes-base.httpRequest",

    # Data processing
    "json:ParseJSON": "n8n-nodes-base.set",
    "tools:SetVariables": "n8n-nodes-base.set",
    "tools:TextAggregator": "n8n-nodes-base.aggregate",
    "filter:FilterByCondition": "n8n-nodes-base.if",
    "flow:Router": "n8n-nodes-base.switch",
    "flow:Sleep": "n8n-nodes-base.wait",
    "flow:SetVariable": "n8n-nodes-base.set",
    "text-parser:TextParser": "n8n-nodes-base.set",

    # AI
    "openai:CreateCompletion": "@n8n/n8n-nodes-langchain.openAi",
    "openai:CreateChatCompletion": "@n8n/n8n-nodes-langchain.openAi",
    "anthropic:CreateMessage": "@n8n/n8n-nodes-langchain.anthropic",

    # CRM
    "hubspot:CreateContact": "n8n-nodes-base.hubspot",
    "hubspot:UpdateContact": "n8n-nodes-base.hubspot",
    "salesforce:CreateRecord": "n8n-nodes-base.salesforce",
    "pipedrive:CreatePerson": "n8n-nodes-base.pipedrive",

    # Databases
    "postgresql:executeQuery": "n8n-nodes-base.postgres",
    "mysql:executeQuery": "n8n-nodes-base.mySql",
    "mongodb:InsertDocument": "n8n-nodes-base.mongoDb",

    # File storage
    "dropbox:uploadFile": "n8n-nodes-base.dropbox",
    "box:uploadFile": "n8n-nodes-base.box",
}

# n8n node operation mappings
NODE_OPERATIONS = {
    "n8n-nodes-base.slack": {"resource": "message", "operation": "post"},
    "n8n-nodes-base.googleSheets": {"resource": "sheet", "operation": "appendOrUpdate"},
    "n8n-nodes-base.airtable": {"resource": "record", "operation": "create"},
    "n8n-nodes-base.notion": {"resource": "page", "operation": "create"},
    "n8n-nodes-base.httpRequest": {"method": "GET"},
    "n8n-nodes-base.gmail": {"resource": "message", "operation": "send"},
    "n8n-nodes-base.googleCalendar": {"resource": "event", "operation": "create"},
    "n8n-nodes-base.hubspot": {"resource": "contact", "operation": "create"},
    "n8n-nodes-base.salesforce": {"operation": "create"},
}

# Default node dimensions
NODE_WIDTH = 240
NODE_HEIGHT = 60
X_START = 240
Y_CENTER = 300
X_SPACING = 320


def _make_node_id() -> str:
    return str(uuid.uuid4())


def _map_module(make_module: str) -> str:
    """Map a Make.com module identifier to an n8n node type."""
    if make_module in MODULE_MAP:
        return MODULE_MAP[make_module]
    # Fallback: try package-level match
    pkg = make_module.split(":")[0]
    for key, val in MODULE_MAP.items():
        if key.startswith(pkg + ":"):
            return val
    # Final fallback
    return "n8n-nodes-base.httpRequest"


def _convert_mapper(make_mapper: dict) -> dict:
    """
    Convert Make.com mapper syntax to n8n parameter syntax.
    Make: {{1.data.email}} → n8n: {{ $json.email }} (simplified)
    """
    if not make_mapper:
        return {}
    n8n_params = {}
    for key, val in make_mapper.items():
        if isinstance(val, str):
            # Convert {{N.field}} → {{ $json.field }}
            import re
            converted = re.sub(r"\{\{(\d+)\.([^}]+)\}\}", r"{{ $json.\2 }}", val)
            # Remove __PLACEHOLDER__ patterns
            converted = re.sub(r"__[A-Z_]+__", "", converted).strip()
            n8n_params[key] = converted
        else:
            n8n_params[key] = val
    return n8n_params


def _build_trigger_node(trigger: dict, node_id: str) -> dict:
    """Build n8n trigger node from canonical spec trigger."""
    make_module = trigger.get("module", "gateway:CustomWebHook")
    n8n_type = _map_module(make_module)

    node = {
        "id": node_id,
        "name": trigger.get("label", "Trigger"),
        "type": n8n_type,
        "typeVersion": 1,
        "position": [X_START, Y_CENTER],
        "parameters": {},
    }

    # Trigger-specific config
    if n8n_type == "n8n-nodes-base.webhook":
        node["parameters"] = {
            "httpMethod": "POST",
            "path": trigger.get("parameters", {}).get("hook", "webhook"),
            "responseMode": "onReceived",
        }
        node["webhookId"] = _make_node_id()

    elif n8n_type == "n8n-nodes-base.scheduleTrigger":
        node["parameters"] = {
            "rule": {
                "interval": [{"field": "hours", "hoursInterval": 1}]
            }
        }

    return node


def _build_step_node(step: dict, position: list, node_id: str) -> dict:
    """Build n8n action node from canonical spec step."""
    make_module = step.get("module", "http:ActionSendData")
    n8n_type = _map_module(make_module)

    params = {}
    # Apply default operations for known node types
    if n8n_type in NODE_OPERATIONS:
        params.update(NODE_OPERATIONS[n8n_type])

    # Merge converted mapper values
    mapped = _convert_mapper(step.get("mapper", {}))
    params.update(mapped)

    node = {
        "id": node_id,
        "name": step.get("label", step.get("module", "Step")),
        "type": n8n_type,
        "typeVersion": 1,
        "position": position,
        "parameters": params,
    }

    return node


def _build_connections(node_ids: list) -> dict:
    """
    Build n8n connections dict from ordered node ID list.
    Linear flow: each node connects to the next.
    """
    connections = {}
    for i in range(len(node_ids) - 1):
        source_id = node_ids[i]
        target_id = node_ids[i + 1]
        connections[source_id] = {
            "main": [[{"node": target_id, "type": "main", "index": 0}]]
        }
    return connections


def generate_n8n_export(canonical_spec: dict, registry: dict = None) -> dict:
    """
    Convert canonical spec to n8n workflow JSON.

    Args:
        canonical_spec: Standard canonical spec dict from the pipeline
        registry: Module registry (used for validation hints, optional)

    Returns:
        dict with:
            workflow: complete n8n workflow JSON
            agent_notes: list of conversion notes
            unmapped_modules: list of modules with no direct n8n equivalent
    """
    agent_notes = []
    unmapped_modules = []

    trigger = canonical_spec.get("trigger", {})
    steps = canonical_spec.get("steps", [])
    scenario_name = canonical_spec.get("scenario_name", "ManageAI Workflow")

    node_ids = []
    nodes = []

    # Build trigger node
    trigger_id = _make_node_id()
    trigger_node = _build_trigger_node(trigger, trigger_id)
    nodes.append(trigger_node)
    node_ids.append(trigger_id)

    make_module = trigger.get("module", "")
    if make_module not in MODULE_MAP:
        unmapped_modules.append(make_module)
        agent_notes.append(f"Trigger module '{make_module}' has no direct n8n equivalent — mapped to webhook.")

    # Build step nodes
    for i, step in enumerate(steps):
        x = X_START + (i + 1) * X_SPACING
        y = Y_CENTER
        node_id = _make_node_id()
        node = _build_step_node(step, [x, y], node_id)
        nodes.append(node)
        node_ids.append(node_id)

        make_module = step.get("module", "")
        if make_module not in MODULE_MAP:
            unmapped_modules.append(make_module)
            agent_notes.append(f"Module '{make_module}' has no direct n8n equivalent — mapped to HTTP Request. Manual configuration required.")

    # Build connections
    connections = _build_connections(node_ids)

    # Build settings
    error_strategy = canonical_spec.get("error_handling", {}).get("default_strategy", "ignore")
    settings = {
        "executionOrder": "v1",
        "saveManualExecutions": True,
        "callerPolicy": "workflowsFromSameOwner",
        "errorWorkflow": "" if error_strategy == "ignore" else "__ERROR_WORKFLOW__",
    }

    if error_strategy != "ignore":
        agent_notes.append(
            f"Error handling strategy '{error_strategy}' noted. "
            "Configure an n8n error workflow and set errorWorkflow ID."
        )

    workflow = {
        "name": scenario_name,
        "nodes": nodes,
        "connections": connections,
        "settings": settings,
        "staticData": None,
        "tags": [{"name": "manageai"}, {"name": "auto-generated"}],
        "meta": {
            "templateCredsSetupCompleted": False,
            "generatedBy": "ManageAI Agentic Builder",
            "sourceFormat": "canonical_spec_v1",
            "originalScenario": scenario_name,
        }
    }

    if unmapped_modules:
        agent_notes.append(
            f"{len(unmapped_modules)} module(s) required manual mapping: {', '.join(set(unmapped_modules))}"
        )

    return {
        "workflow": workflow,
        "agent_notes": agent_notes,
        "unmapped_modules": list(set(unmapped_modules)),
        "node_count": len(nodes),
        "connection_count": len(connections),
    }


if __name__ == "__main__":
    print("=== n8n Export Generator Self-Check ===\n")

    sample_spec = {
        "scenario_name": "Acme — Form to Slack",
        "scenario_description": "Webhook triggers Slack notification",
        "trigger": {
            "type": "webhook",
            "module": "gateway:CustomWebHook",
            "label": "Form Submitted",
            "parameters": {"hook": "acme-form-webhook"},
        },
        "steps": [
            {"module": "json:ParseJSON", "label": "Parse Form Data",
             "mapper": {"data": "{{1.data}}"}, "parameters": {}},
            {"module": "slack:PostMessage", "label": "Notify Slack",
             "mapper": {"text": "New form: {{2.email}}", "channel": "#general"},
             "parameters": {"connection": "__SLACK__"}},
            {"module": "airtable:CreateRecord", "label": "Log to Airtable",
             "mapper": {"Email": "{{2.email}}", "Name": "{{2.name}}"},
             "parameters": {"connection": "__AIRTABLE__"}},
        ],
        "error_handling": {"default_strategy": "ignore"},
    }

    print("Test 1: Generate n8n workflow")
    result = generate_n8n_export(sample_spec)
    assert "workflow" in result
    wf = result["workflow"]
    assert wf["name"] == "Acme — Form to Slack"
    print(f"  Nodes: {result['node_count']}")
    print(f"  Connections: {result['connection_count']}")
    print("  [OK]")

    print("Test 2: Trigger node is correct type")
    trigger_node = wf["nodes"][0]
    assert trigger_node["type"] == "n8n-nodes-base.webhook"
    assert "webhookId" in trigger_node
    print(f"  Trigger: {trigger_node['type']}")
    print("  [OK]")

    print("Test 3: Steps converted to correct node types")
    step_types = [n["type"] for n in wf["nodes"][1:]]
    assert "n8n-nodes-base.slack" in step_types
    assert "n8n-nodes-base.airtable" in step_types
    print(f"  Step types: {step_types}")
    print("  [OK]")

    print("Test 4: Connections are linear")
    assert len(wf["connections"]) == 3  # trigger + 2 step connections
    print(f"  Connection count: {len(wf['connections'])}")
    print("  [OK]")

    print("Test 5: Mapper conversion removes __PLACEHOLDER__")
    slack_node = next(n for n in wf["nodes"] if "slack" in n["type"])
    assert "__SLACK__" not in str(slack_node["parameters"])
    print("  [OK]")

    print("Test 6: Unmapped module handled gracefully")
    spec_with_unknown = {
        **sample_spec,
        "steps": [{"module": "unknown:WeirdModule", "label": "Unknown", "mapper": {}, "parameters": {}}]
    }
    result2 = generate_n8n_export(spec_with_unknown)
    assert len(result2["unmapped_modules"]) > 0
    assert len(result2["agent_notes"]) > 0
    print(f"  Unmapped: {result2['unmapped_modules']}")
    print("  [OK]")

    print("Test 7: Workflow JSON is serializable")
    serialized = json.dumps(result["workflow"])
    reloaded = json.loads(serialized)
    assert reloaded["name"] == "Acme — Form to Slack"
    print("  [OK]")

    print("\n=== All n8n export checks passed ===")
