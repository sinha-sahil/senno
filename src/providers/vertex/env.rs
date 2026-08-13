use std::str::FromStr;

pub(crate) fn resolve_with_default<T>(param: Option<T>, env_var: &str, default: T) -> T
where
    T: FromStr,
{
    if let Some(value) = param {
        return value;
    }

    std::env::var(env_var)
        .ok()
        .and_then(|v| v.parse::<T>().ok())
        .unwrap_or(default)
}

pub(crate) fn resolve_optional<T>(param: Option<T>, env_var: &str) -> Option<T>
where
    T: FromStr,
{
    if param.is_some() {
        return param;
    }

    std::env::var(env_var)
        .ok()
        .and_then(|v| v.parse::<T>().ok())
}
