"""
Module Registry Loader

Loads and indexes the curated module registry from disk.
Returns a dict keyed by module_id for O(1) lookups.

Input: registry_path (str) — path to module_registry.json
Output: dict keyed by module_id with full module metadata

Deterministic. No network calls. No conversation context.
"""

import json
import os


def load_module_registry(registry_path=None):
    """Load the module registry and return an indexed dict.

    Args:
        registry_path: Path to module_registry.json.
                       Defaults to module_registry.json in the same directory.

    Returns:
        dict with keys:
            - registry_version: str
            - modules: dict keyed by module_id
            - module_count: int

    Raises:
        FileNotFoundError: If registry file does not exist.
        ValueError: If registry is malformed or missing required fields.
    """
    if registry_path is None:
        registry_path = os.path.join(os.path.dirname(__file__), "module_registry.json")

    if not os.path.exists(registry_path):
        raise FileNotFoundError(f"Module registry not found at: {registry_path}")

    with open(registry_path, "r") as f:
        raw = json.load(f)

    # Validate top-level structure
    if "registry_version" not in raw:
        raise ValueError("Registry missing 'registry_version' field")
    if "modules" not in raw or not isinstance(raw["modules"], dict):
        raise ValueError("Registry missing 'modules' dict")

    required_module_fields = [
        "module_id", "app", "label", "category", "version",
        "requires_credential", "credential_placeholder",
        "required_parameters", "required_mapper_fields", "description"
    ]

    valid_categories = {
        "trigger", "action", "search", "flow_control",
        "transformer", "aggregator", "iterator", "responder"
    }

    modules = {}
    for module_id, meta in raw["modules"].items():
        # Validate each module entry
        missing = [f for f in required_module_fields if f not in meta]
        if missing:
            raise ValueError(
                f"Module '{module_id}' missing required fields: {missing}"
            )

        if meta["module_id"] != module_id:
            raise ValueError(
                f"Module key '{module_id}' does not match module_id '{meta['module_id']}'"
            )

        if meta["category"] not in valid_categories:
            raise ValueError(
                f"Module '{module_id}' has invalid category: '{meta['category']}'"
            )

        modules[module_id] = meta

    return {
        "registry_version": raw["registry_version"],
        "modules": modules,
        "module_count": len(modules)
    }


def get_module(registry, module_id):
    """Look up a single module by ID.

    Args:
        registry: The loaded registry dict from load_module_registry().
        module_id: The module identifier string (e.g., 'slack:PostMessage').

    Returns:
        Module metadata dict, or None if not found.
    """
    return registry["modules"].get(module_id)


def list_modules_by_category(registry, category):
    """List all modules in a given category.

    Args:
        registry: The loaded registry dict.
        category: Category string (trigger, action, search, etc.)

    Returns:
        List of module metadata dicts.
    """
    return [
        meta for meta in registry["modules"].values()
        if meta["category"] == category
    ]


# --- Self-check ---
if __name__ == "__main__":
    print("=== Module Registry Loader Self-Check ===\n")

    reg = load_module_registry()
    print(f"Registry version: {reg['registry_version']}")
    print(f"Module count: {reg['module_count']}")
    assert reg["module_count"] == 16, f"Expected 16 modules, got {reg['module_count']}"

    # Verify all expected modules exist
    expected = [
        "gateway:CustomWebHook", "builtin:BasicRouter",
        "builtin:BasicIterator", "builtin:BasicAggregator",
        "http:ActionSendData", "http:ActionGetFile",
        "json:ParseJSON", "json:TransformToJSON", "json:AggregateToJSON",
        "util:SetVariable", "util:SetMultipleVariables",
        "google-sheets:addRow", "google-sheets:getRow", "google-sheets:updateRow",
        "slack:PostMessage", "gateway:WebhookResponse"
    ]
    for mid in expected:
        m = get_module(reg, mid)
        assert m is not None, f"Module '{mid}' not found in registry"
        print(f"  [OK] {mid} ({m['category']})")

    # Verify category lookup
    triggers = list_modules_by_category(reg, "trigger")
    assert len(triggers) == 1, f"Expected 1 trigger, got {len(triggers)}"

    actions = list_modules_by_category(reg, "action")
    assert len(actions) >= 5, f"Expected >= 5 actions, got {len(actions)}"

    # Verify credential modules
    cred_modules = [
        m for m in reg["modules"].values() if m["requires_credential"]
    ]
    assert len(cred_modules) == 4, f"Expected 4 credential modules, got {len(cred_modules)}"
    for m in cred_modules:
        assert m["credential_placeholder"] is not None, \
            f"Module '{m['module_id']}' requires credential but placeholder is None"

    print(f"\n=== All checks passed ({reg['module_count']} modules validated) ===")
