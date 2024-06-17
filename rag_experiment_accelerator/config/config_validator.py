import json
import os
from jsonschema import ValidationError, validate
import requests

schema_cache = {}


def fetch_json_schema_from_url(schema_url):
    """Fetch the JSON schema from a URL."""
    response = requests.get(schema_url, timeout=5)
    response.raise_for_status()
    return response.json()


def fetch_json_schema_from_file(schema_file_path, source_file_path):
    """Fetch the JSON schema from a local file path."""
    normalised_schema_path = get_normalised_schema_path(
        schema_file_path, source_file_path
    )

    if not os.path.isfile(normalised_schema_path):
        raise ValueError(f"Local schema file not found: {normalised_schema_path}")

    with open(normalised_schema_path, "r", encoding="utf8") as schema_file:
        return json.load(schema_file)


def get_normalised_schema_path(schema_file_path, source_file_path):
    source_dir = os.path.dirname(source_file_path)
    new_schema_file_path = os.path.join(source_dir, schema_file_path)
    return os.path.normpath(new_schema_file_path)


def fetch_json_schema(schema_reference, source_file_path):
    """Fetch the JSON schema from a URL or local file path, with caching."""
    if schema_reference in schema_cache:
        return schema_cache[schema_reference]

    schema = (
        fetch_json_schema_from_url(schema_reference)
        if schema_reference.startswith(("http://", "https://"))
        else fetch_json_schema_from_file(schema_reference, source_file_path)
    )

    schema_cache[schema_reference] = schema
    return schema


def validate_json_with_schema(
    json_data, source_file_path
) -> tuple[bool, ValidationError | None]:
    """Validate a JSON object using the schema specified in its $schema property."""
    try:
        schema_reference = json_data.get("$schema")
        if not schema_reference:
            return True, None

        schema = fetch_json_schema(schema_reference, source_file_path)

        validate(instance=json_data, schema=schema)
        return True, None
    except ValidationError as ve:
        return False, ve
