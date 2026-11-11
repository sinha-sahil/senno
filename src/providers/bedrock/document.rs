//! `serde_json::Value` ⇄ `aws_smithy_types::Document`: tool schemas and tool
//! arguments cross the SDK boundary as Smithy documents.

use std::collections::HashMap;

use aws_smithy_types::{Document, Number};
use serde_json::Value;

pub(super) fn to_document(value: &Value) -> Document {
    match value {
        Value::Null => Document::Null,
        Value::Bool(b) => Document::Bool(*b),
        Value::Number(n) => Document::Number(if let Some(u) = n.as_u64() {
            Number::PosInt(u)
        } else if let Some(i) = n.as_i64() {
            Number::NegInt(i)
        } else {
            Number::Float(n.as_f64().unwrap_or(0.0))
        }),
        Value::String(s) => Document::String(s.clone()),
        Value::Array(items) => Document::Array(items.iter().map(to_document).collect()),
        Value::Object(map) => Document::Object(
            map.iter()
                .map(|(k, v)| (k.clone(), to_document(v)))
                .collect::<HashMap<_, _>>(),
        ),
    }
}

pub(super) fn to_value(document: &Document) -> Value {
    match document {
        Document::Null => Value::Null,
        Document::Bool(b) => Value::Bool(*b),
        Document::Number(Number::PosInt(u)) => Value::from(*u),
        Document::Number(Number::NegInt(i)) => Value::from(*i),
        Document::Number(Number::Float(f)) => {
            serde_json::Number::from_f64(*f).map_or(Value::Null, Value::Number)
        }
        Document::String(s) => Value::String(s.clone()),
        Document::Array(items) => Value::Array(items.iter().map(to_value).collect()),
        Document::Object(map) => {
            Value::Object(map.iter().map(|(k, v)| (k.clone(), to_value(v))).collect())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn round_trips_every_json_shape() {
        let value = json!({
            "s": "text",
            "n": 3,
            "neg": -7,
            "f": 1.5,
            "b": true,
            "null": null,
            "list": [1, "two", {"three": 3}],
        });
        assert_eq!(to_value(&to_document(&value)), value);
    }
}
