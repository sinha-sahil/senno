use indexmap::IndexMap;
use serde_json::{Map, Value};

use crate::types::{ParameterSchema, SchemaType};

const UNION_KEYS: [&str; 3] = ["anyOf", "oneOf", "allOf"];

pub(super) fn to_parameter_schema(node: &Value) -> Option<ParameterSchema> {
    let map = node.as_object()?;
    match schema_type(map) {
        Some(schema_type) => Some(build(map, schema_type)),
        None => first_union_branch(map),
    }
}

fn build(map: &Map<String, Value>, schema_type: SchemaType) -> ParameterSchema {
    let properties = converted_properties(map);
    let required = declared_required(map, properties.as_ref());

    ParameterSchema {
        schema_type,
        description: map
            .get("description")
            .and_then(Value::as_str)
            .map(String::from),
        properties,
        items: map.get("items").and_then(to_parameter_schema).map(Box::new),
        required,
        enum_values: string_enum(map),
    }
}

fn schema_type(map: &Map<String, Value>) -> Option<SchemaType> {
    match map.get("type") {
        Some(Value::String(name)) => named_type(name),
        Some(Value::Array(names)) => names
            .iter()
            .filter_map(Value::as_str)
            .find(|name| *name != "null")
            .and_then(named_type),
        _ => inferred_type(map),
    }
}

fn named_type(name: &str) -> Option<SchemaType> {
    match name {
        "string" => Some(SchemaType::String),
        "integer" => Some(SchemaType::Integer),
        "number" => Some(SchemaType::Number),
        "boolean" => Some(SchemaType::Boolean),
        "array" => Some(SchemaType::Array),
        "object" => Some(SchemaType::Object),
        _ => None,
    }
}

fn inferred_type(map: &Map<String, Value>) -> Option<SchemaType> {
    if map.contains_key("properties") {
        return Some(SchemaType::Object);
    }
    if map.contains_key("items") {
        return Some(SchemaType::Array);
    }
    string_enum(map).map(|_| SchemaType::String)
}

fn first_union_branch(map: &Map<String, Value>) -> Option<ParameterSchema> {
    UNION_KEYS
        .iter()
        .filter_map(|key| map.get(*key))
        .filter_map(Value::as_array)
        .flatten()
        .find_map(to_parameter_schema)
}

fn converted_properties(map: &Map<String, Value>) -> Option<IndexMap<String, ParameterSchema>> {
    let source = map.get("properties")?.as_object()?;
    let mut converted = IndexMap::with_capacity(source.len());

    for (name, node) in source {
        match to_parameter_schema(node) {
            Some(schema) => {
                converted.insert(name.clone(), schema);
            }
            None => tracing::debug!(property = %name, "mcp: property is not expressible"),
        }
    }

    (!converted.is_empty()).then_some(converted)
}

fn declared_required(
    map: &Map<String, Value>,
    properties: Option<&IndexMap<String, ParameterSchema>>,
) -> Option<Vec<String>> {
    let listed = map.get("required")?.as_array()?;
    let kept: Vec<String> = listed
        .iter()
        .filter_map(Value::as_str)
        .filter(|name| properties.is_some_and(|schemas| schemas.contains_key(*name)))
        .map(String::from)
        .collect();

    (!kept.is_empty()).then_some(kept)
}

fn string_enum(map: &Map<String, Value>) -> Option<Vec<String>> {
    let values: Vec<String> = map
        .get("enum")?
        .as_array()?
        .iter()
        .map(|value| value.as_str().map(String::from))
        .collect::<Option<_>>()?;

    (!values.is_empty()).then_some(values)
}

pub(super) fn mismatch(schema: &ParameterSchema, value: &Value) -> Option<String> {
    check(schema, value, "$")
}

fn check(schema: &ParameterSchema, value: &Value, path: &str) -> Option<String> {
    if value.is_null() {
        return None;
    }
    if !matches_type(&schema.schema_type, value) {
        return Some(format!("{path} should be {}", label(&schema.schema_type)));
    }

    match value {
        Value::Object(fields) => check_fields(schema, fields, path),
        Value::Array(items) => check_items(schema, items, path),
        Value::String(text) => check_enum(schema, text, path),
        _ => None,
    }
}

fn matches_type(schema_type: &SchemaType, value: &Value) -> bool {
    match schema_type {
        SchemaType::String => value.is_string(),
        SchemaType::Integer => {
            value.is_i64() || value.is_u64() || value.as_f64().is_some_and(|n| n.fract() == 0.0)
        }
        SchemaType::Number => value.is_number(),
        SchemaType::Boolean => value.is_boolean(),
        SchemaType::Array => value.is_array(),
        SchemaType::Object => value.is_object(),
    }
}

fn label(schema_type: &SchemaType) -> &'static str {
    match schema_type {
        SchemaType::String => "a string",
        SchemaType::Integer => "an integer",
        SchemaType::Number => "a number",
        SchemaType::Boolean => "a boolean",
        SchemaType::Array => "an array",
        SchemaType::Object => "an object",
    }
}

fn check_fields(
    schema: &ParameterSchema,
    fields: &Map<String, Value>,
    path: &str,
) -> Option<String> {
    if let Some(required) = schema.required.as_ref()
        && let Some(missing) = required
            .iter()
            .find(|name| !fields.contains_key(name.as_str()))
    {
        return Some(format!("{path} is missing {missing}"));
    }

    let properties = schema.properties.as_ref()?;
    fields.iter().find_map(|(name, value)| {
        properties
            .get(name)
            .and_then(|property| check(property, value, &format!("{path}.{name}")))
    })
}

fn check_items(schema: &ParameterSchema, items: &[Value], path: &str) -> Option<String> {
    let item_schema = schema.items.as_ref()?;
    items
        .iter()
        .enumerate()
        .find_map(|(index, value)| check(item_schema, value, &format!("{path}[{index}]")))
}

fn check_enum(schema: &ParameterSchema, text: &str, path: &str) -> Option<String> {
    let allowed = schema.enum_values.as_ref()?;
    if allowed.iter().any(|value| value == text) {
        return None;
    }
    Some(format!("{path} is not one of {}", allowed.join(", ")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn convert(value: Value) -> Option<ParameterSchema> {
        to_parameter_schema(&value)
    }

    #[test]
    fn nullable_union_takes_the_first_real_type() {
        let schema = convert(json!({ "type": ["string", "null"] })).expect("converts");
        assert!(matches!(schema.schema_type, SchemaType::String));
    }

    #[test]
    fn a_missing_type_is_inferred_from_properties() {
        let schema =
            convert(json!({ "properties": { "id": { "type": "string" } } })).expect("converts");
        assert!(matches!(schema.schema_type, SchemaType::Object));
        assert_eq!(schema.properties.expect("properties").len(), 1);
    }

    #[test]
    fn a_missing_type_is_inferred_from_items() {
        let schema = convert(json!({ "items": { "type": "string" } })).expect("converts");
        assert!(matches!(schema.schema_type, SchemaType::Array));
    }

    #[test]
    fn an_untyped_union_falls_back_to_its_first_usable_branch() {
        let schema = convert(json!({
            "anyOf": [{ "$ref": "#/$defs/Missing" }, { "type": "integer" }]
        }))
        .expect("converts");
        assert!(matches!(schema.schema_type, SchemaType::Integer));
    }

    #[test]
    fn a_node_with_nothing_to_go_on_is_dropped() {
        assert!(convert(json!({ "$ref": "#/$defs/Missing" })).is_none());
        assert!(convert(json!({ "type": "chimera" })).is_none());
        assert!(convert(json!("not an object")).is_none());
    }

    #[test]
    fn required_drops_names_whose_property_was_dropped() {
        let schema = convert(json!({
            "type": "object",
            "properties": {
                "hotelId": { "type": "string" },
                "shape": { "$ref": "#/$defs/Missing" }
            },
            "required": ["hotelId", "shape"]
        }))
        .expect("converts");

        assert_eq!(schema.required.expect("required"), vec!["hotelId"]);
        assert!(!schema.properties.expect("properties").contains_key("shape"));
    }

    #[test]
    fn only_a_wholly_string_enum_survives() {
        let kept = convert(json!({ "type": "string", "enum": ["PAY_LATER", "FULL_PAYMENT"] }))
            .expect("converts");
        assert_eq!(kept.enum_values.expect("enum").len(), 2);

        let dropped = convert(json!({ "type": "integer", "enum": [1, 2] })).expect("converts");
        assert!(dropped.enum_values.is_none());
    }

    #[test]
    fn an_empty_enum_is_not_an_enum() {
        let typed = convert(json!({ "type": "string", "enum": [] })).expect("converts");
        assert!(typed.enum_values.is_none());
        assert!(convert(json!({ "enum": [] })).is_none());
    }

    #[test]
    fn nested_objects_inside_arrays_survive() {
        let schema = convert(json!({
            "type": "array",
            "items": {
                "type": "object",
                "properties": { "guestId": { "type": "string" } },
                "required": ["guestId"]
            }
        }))
        .expect("converts");

        let items = schema.items.expect("items");
        assert!(matches!(items.schema_type, SchemaType::Object));
        assert_eq!(items.required.expect("required"), vec!["guestId"]);
    }

    #[test]
    fn unrepresentable_keywords_are_dropped_without_failing() {
        let schema = convert(json!({
            "type": "string",
            "format": "date",
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "minLength": 4,
            "description": "Arrival date"
        }))
        .expect("converts");

        assert!(matches!(schema.schema_type, SchemaType::String));
        assert_eq!(schema.description.as_deref(), Some("Arrival date"));
    }
    fn out(schema: Value, value: Value) -> Option<String> {
        let schema = to_parameter_schema(&schema).expect("schema converts");
        mismatch(&schema, &value)
    }

    #[test]
    fn a_conforming_result_reports_nothing() {
        assert!(
            out(
                json!({
                    "type": "object",
                    "properties": {
                        "total": { "type": "integer" },
                        "currency": { "type": "string", "enum": ["INR", "SEK"] },
                        "lines": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": { "sku": { "type": "string" } },
                                "required": ["sku"]
                            }
                        }
                    },
                    "required": ["total"]
                }),
                json!({
                    "total": 4200,
                    "currency": "SEK",
                    "lines": [{ "sku": "room-1" }],
                    "extra": "ignored"
                })
            )
            .is_none()
        );
    }

    #[test]
    fn a_missing_required_field_is_named() {
        let problem = out(
            json!({ "type": "object", "properties": { "total": { "type": "integer" } }, "required": ["total"] }),
            json!({ "currency": "SEK" }),
        )
        .expect("a problem");
        assert!(problem.contains("missing total"), "{problem}");
    }

    #[test]
    fn a_wrong_type_names_its_path() {
        let problem = out(
            json!({ "type": "object", "properties": { "total": { "type": "integer" } } }),
            json!({ "total": "free" }),
        )
        .expect("a problem");
        assert!(problem.contains("$.total"), "{problem}");
        assert!(problem.contains("integer"), "{problem}");
    }

    #[test]
    fn a_bad_array_element_names_its_index() {
        let problem = out(
            json!({
                "type": "object",
                "properties": { "lines": { "type": "array", "items": { "type": "object", "properties": { "sku": { "type": "string" } } } } }
            }),
            json!({ "lines": [{ "sku": "ok" }, { "sku": 7 }] }),
        )
        .expect("a problem");
        assert!(problem.contains("$.lines[1].sku"), "{problem}");
    }

    #[test]
    fn a_value_outside_the_enum_is_reported() {
        let problem = out(
            json!({ "type": "object", "properties": { "currency": { "type": "string", "enum": ["INR"] } } }),
            json!({ "currency": "SEK" }),
        )
        .expect("a problem");
        assert!(problem.contains("not one of INR"), "{problem}");
    }

    #[test]
    fn null_is_never_a_mismatch_because_nullability_was_dropped() {
        assert!(
            out(
                json!({ "type": "object", "properties": { "total": { "type": "integer" } } }),
                json!({ "total": null })
            )
            .is_none()
        );
    }

    #[test]
    fn a_whole_number_float_still_counts_as_an_integer() {
        assert!(
            out(
                json!({ "type": "object", "properties": { "nights": { "type": "integer" } } }),
                json!({ "nights": 3.0 })
            )
            .is_none()
        );
    }
}
