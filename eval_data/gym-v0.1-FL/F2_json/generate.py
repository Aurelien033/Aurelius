#!/usr/bin/env python3
"""F2 JSON/schema repair generator for gym-v0.1-FL.

Generates synthetic JSON configs and applies mutations (bracket drop, key deletion,
type confusion, comma/quote corruption). Verifier: json.loads + jsonschema.
"""

import hashlib
import json
import random
import uuid
from pathlib import Path

SEED = 20260612
DEV_POOL_SIZE = 300

# JSON schema templates (synthetic, post-cutoff by construction)
SCHEMA_TEMPLATES = [
    {
        "name": "database_config",
        "schema": {
            "type": "object",
            "properties": {
                "host": {"type": "string"},
                "port": {"type": "integer", "minimum": 1, "maximum": 65535},
                "username": {"type": "string"},
                "password": {"type": "string"},
                "database": {"type": "string"},
                "pool_size": {"type": "integer", "minimum": 1}
            },
            "required": ["host", "port", "username", "database"]
        },
        "valid_instance": {
            "host": "localhost",
            "port": 5432,
            "username": "admin",
            "password": "secret",
            "database": "mydb",
            "pool_size": 10
        }
    },
    {
        "name": "api_endpoint",
        "schema": {
            "type": "object",
            "properties": {
                "method": {"type": "string", "enum": ["GET", "POST", "PUT", "DELETE"]},
                "path": {"type": "string", "pattern": "^/"},
                "timeout_ms": {"type": "integer", "minimum": 0},
                "retry_count": {"type": "integer", "minimum": 0},
                "headers": {"type": "object"}
            },
            "required": ["method", "path"]
        },
        "valid_instance": {
            "method": "POST",
            "path": "/api/v1/users",
            "timeout_ms": 5000,
            "retry_count": 3,
            "headers": {"Content-Type": "application/json"}
        }
    },
    {
        "name": "user_profile",
        "schema": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string"},
                "email": {"type": "string"},
                "age": {"type": "integer", "minimum": 0, "maximum": 150},
                "active": {"type": "boolean"},
                "tags": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["user_id", "email", "active"]
        },
        "valid_instance": {
            "user_id": "u12345",
            "email": "user@example.com",
            "age": 30,
            "active": True,
            "tags": ["admin", "verified"]
        }
    },
    {
        "name": "logging_config",
        "schema": {
            "type": "object",
            "properties": {
                "level": {"type": "string", "enum": ["DEBUG", "INFO", "WARNING", "ERROR"]},
                "file": {"type": "string"},
                "max_size_mb": {"type": "integer", "minimum": 1},
                "rotate_count": {"type": "integer", "minimum": 0},
                "format": {"type": "string"}
            },
            "required": ["level", "file"]
        },
        "valid_instance": {
            "level": "INFO",
            "file": "/var/log/app.log",
            "max_size_mb": 100,
            "rotate_count": 5,
            "format": "%(asctime)s - %(levelname)s - %(message)s"
        }
    }
]

MUTATION_OPS = [
    "bracket_drop",        # Remove opening/closing brace
    "key_deletion",        # Delete a required key
    "type_confusion",      # Change type (string -> int, etc.)
    "comma_corruption",    # Add/remove comma
    "quote_corruption",    # Remove quotes from string value
    "trailing_comma",      # Add trailing comma (invalid JSON)
]


def sha256_short(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def apply_json_mutation(json_str: str, op: str, rng: random.Random) -> tuple[str, str]:
    """Apply mutation to JSON string. Returns (mutated, description)."""
    
    if op == "bracket_drop":
        # Remove first { or }
        if "{" in json_str:
            mutated = json_str.replace("{", "", 1)
            return mutated, "removed opening brace"
        elif "}" in json_str:
            mutated = json_str.replace("}", "", 1)
            return mutated, "removed closing brace"
    
    elif op == "key_deletion":
        # Find and remove a key-value pair
        lines = json_str.split("\n")
        for i, line in enumerate(lines):
            if '":' in line and i > 0:  # Skip first line
                mutated = "\n".join(lines[:i] + lines[i+1:])
                return mutated, f"deleted key at line {i}"
    
    elif op == "type_confusion":
        # Change a numeric value to string or vice versa
        # Find a number and quote it
        import re
        nums = re.findall(r':\s*(\d+)', json_str)
        if nums:
            old = nums[0]
            mutated = json_str.replace(f": {old}", f': "{old}"', 1)
            return mutated, f"type confusion: {old} -> \"{old}\""
    
    elif op == "comma_corruption":
        # Remove a comma
        if "," in json_str:
            mutated = json_str.replace(",", "", 1)
            return mutated, "removed comma"
    
    elif op == "quote_corruption":
        # Remove quotes from a string value
        import re
        strings = re.findall(r'":\s*"([^"]+)"', json_str)
        if strings:
            old = strings[0]
            mutated = json_str.replace(f'"{old}"', old, 1)
            return mutated, f"removed quotes from \"{old}\""
    
    elif op == "trailing_comma":
        # Add trailing comma before closing brace
        if "}" in json_str:
            mutated = json_str.replace("}", ",}", 1)
            return mutated, "added trailing comma"
    
    return json_str, "no_mutation"


def generate_f2_instances():
    """Generate F2 JSON/schema repair instances."""
    print("Generating F2 JSON/schema instances...")
    rng = random.Random(SEED)
    
    instances = []
    
    for template_idx, template in enumerate(SCHEMA_TEMPLATES):
        for i in range(DEV_POOL_SIZE // len(SCHEMA_TEMPLATES)):
            # Start with valid instance
            valid_json = json.dumps(template["valid_instance"], indent=2)
            
            # Apply mutation
            op = rng.choice(MUTATION_OPS)
            mutated_json, mutation_desc = apply_json_mutation(valid_json, op, rng)
            
            if mutated_json == valid_json:
                continue
            
            canary_guid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"gym-v0.1-FL-F2-{template_idx}-{i}"))
            artifact_hash = sha256_short(mutated_json)
            unique_suffix = sha256_short(f"{template_idx}-{i}-{op}")
            instance_id = f"gym-v0.1-FL/F2/{artifact_hash}/{unique_suffix}"
            
            prompt_context = f"""Repair the following broken JSON configuration.

Schema name: {template['name']}
Required fields: {', '.join(template['schema']['required'])}

Broken JSON:
# Canary GUID: {canary_guid}
{mutated_json}

The JSON must be valid and conform to the schema constraints.
"""
            
            instance = {
                "instance_id": instance_id,
                "broken_artifact": mutated_json,
                "prompt_context": prompt_context,
                "verifier": "json.loads + jsonschema",
                "hidden_tests": [
                    f"json.loads(repaired) must not raise",
                    f"jsonschema.validate(repaired, {json.dumps(template['schema'])}) must pass"
                ],
                "metadata": {
                    "family": "F2_json",
                    "mutation_operators": [op],
                    "mutation_description": mutation_desc,
                    "schema_name": template["name"],
                    "contamination_flag": "CLEAN_BY_CONSTRUCTION",
                    "canary_guid": canary_guid,
                    "license": "MIT",
                    "provenance": f"synthetic schema {template['name']}",
                }
            }
            
            instances.append(instance)
    
    print(f"Generated {len(instances)} F2 instances")
    return instances


if __name__ == "__main__":
    instances = generate_f2_instances()
    
    dev_dir = Path(__file__).parent / "dev"
    dev_dir.mkdir(exist_ok=True)
    
    for inst in instances:
        out_path = dev_dir / f"{inst['instance_id'].replace('/', '_')}.json"
        with open(out_path, "w") as f:
            json.dump(inst, f, indent=2)
    
    print(f"Wrote {len(instances)} instances to {dev_dir}")
