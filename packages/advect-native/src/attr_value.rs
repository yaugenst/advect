//! Python conversion for runtime-owned closed graph attributes.

use std::collections::HashSet;

use advect_runtime::ExactFloat;
pub(crate) use advect_runtime::{AttrMap, AttrValue};
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyOverflowError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};

/// Construction-only identity cache for common immutable Python mappings.
#[derive(Debug, Default)]
pub(crate) struct AttrMapCache {
    entries: Vec<(Py<PyDict>, AttrCacheKey, AttrMap)>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum AttrCacheKey {
    Empty,
    Backend(String),
}

impl AttrMapCache {
    pub(crate) fn resolve(
        &mut self,
        py: Python<'_>,
        attrs: &Bound<'_, PyDict>,
    ) -> PyResult<AttrMap> {
        let cache_key = cache_key(attrs)?;
        if let Some((_owner, _key, values)) = cache_key.as_ref().and_then(|key| {
            self.entries
                .iter()
                .find(|(owner, cached_key, _)| owner.bind(py).is(attrs) && cached_key == key)
        }) {
            return Ok(values.clone());
        }
        let values = attr_map_from_python(attrs)?;
        if let Some(cache_key) = cache_key {
            self.entries
                .push((attrs.clone().unbind(), cache_key, values.clone()));
        }
        Ok(values)
    }
}

fn cache_key(attrs: &Bound<'_, PyDict>) -> PyResult<Option<AttrCacheKey>> {
    if attrs.is_empty() {
        return Ok(Some(AttrCacheKey::Empty));
    }
    if attrs.len() != 1 {
        return Ok(None);
    }
    attrs
        .get_item("_advect_backend")?
        .map(|value| value.extract::<String>().map(AttrCacheKey::Backend))
        .transpose()
}

pub(crate) fn attr_map_from_python(attrs: &Bound<'_, PyDict>) -> PyResult<AttrMap> {
    let mut active_containers = HashSet::new();
    map_from_python(attrs, "attrs", &mut active_containers)
}

pub(crate) fn attr_map_to_python(py: Python<'_>, attrs: &AttrMap) -> PyResult<Py<PyDict>> {
    let result = PyDict::new(py);
    for (key, value) in attrs {
        result.set_item(key, attr_value_to_python(py, value)?)?;
    }
    Ok(result.unbind())
}

fn attr_value_from_python(
    value: &Bound<'_, PyAny>,
    path: &str,
    active_containers: &mut HashSet<usize>,
) -> PyResult<AttrValue> {
    if value.is_none() {
        return Ok(AttrValue::Null);
    }
    if let Ok(value) = value.cast_exact::<PyBool>() {
        return Ok(AttrValue::Bool(value.is_true()));
    }
    if let Ok(value) = value.cast_exact::<PyInt>() {
        return value.extract::<i64>().map(AttrValue::Integer).map_err(|_| {
            PyOverflowError::new_err(format!(
                "graph attribute integer at {path} must fit in a signed 64-bit value"
            ))
        });
    }
    if let Ok(value) = value.cast_exact::<PyFloat>() {
        return Ok(AttrValue::Float(ExactFloat::from_f64(value.value())));
    }
    if let Ok(value) = value.cast_exact::<PyString>() {
        return Ok(AttrValue::String(value.to_str()?.to_owned()));
    }
    if let Ok(value) = value.cast_exact::<PyBytes>() {
        return Ok(AttrValue::Bytes(value.as_bytes().to_vec()));
    }
    if let Ok(value) = value.cast_exact::<PyDict>() {
        return with_container(value.as_any(), path, active_containers, |active| {
            map_from_python(value, path, active).map(AttrValue::Map)
        });
    }
    if let Ok(value) = value.cast_exact::<PyList>() {
        return with_container(value.as_any(), path, active_containers, |active| {
            sequence_from_python(value.iter(), path, active).map(AttrValue::List)
        });
    }
    if let Ok(value) = value.cast_exact::<PyTuple>() {
        return with_container(value.as_any(), path, active_containers, |active| {
            sequence_from_python(value.iter(), path, active).map(AttrValue::Tuple)
        });
    }
    let type_name = value.get_type().qualname()?;
    Err(PyTypeError::new_err(format!(
        "unsupported graph attribute value at {path}: {type_name}; expected None, bool, \
         int64, float, str, bytes, list/tuple, or a string-keyed dict"
    )))
}

fn attr_value_to_python(py: Python<'_>, value: &AttrValue) -> PyResult<Py<PyAny>> {
    match value {
        AttrValue::Null => Ok(py.None()),
        AttrValue::Bool(value) => value.into_py_any(py),
        AttrValue::Integer(value) => value.into_py_any(py),
        AttrValue::Float(value) => value.to_f64().into_py_any(py),
        AttrValue::String(value) => value.into_py_any(py),
        AttrValue::Bytes(value) => Ok(PyBytes::new(py, value).into_any().unbind()),
        AttrValue::List(values) => {
            let items = values
                .iter()
                .map(|value| attr_value_to_python(py, value))
                .collect::<PyResult<Vec<_>>>()?;
            Ok(PyList::new(py, items)?.into_any().unbind())
        }
        AttrValue::Tuple(values) => {
            let items = values
                .iter()
                .map(|value| attr_value_to_python(py, value))
                .collect::<PyResult<Vec<_>>>()?;
            Ok(PyTuple::new(py, items)?.into_any().unbind())
        }
        AttrValue::Map(values) => Ok(attr_map_to_python(py, values)?.into_any()),
    }
}

fn map_from_python(
    attrs: &Bound<'_, PyDict>,
    path: &str,
    active_containers: &mut HashSet<usize>,
) -> PyResult<AttrMap> {
    let mut result = AttrMap::new();
    for (key, value) in attrs {
        let key = key.cast_exact::<PyString>().map_err(|_| {
            PyTypeError::new_err(format!(
                "graph attribute mapping at {path} requires string keys; got {}",
                python_type_name(&key)
            ))
        })?;
        let key = key.to_str()?.to_owned();
        let value_path = format!("{path}[{key:?}]");
        result.insert(
            key,
            attr_value_from_python(&value, &value_path, active_containers)?,
        );
    }
    Ok(result)
}

fn sequence_from_python<'py>(
    values: impl Iterator<Item = Bound<'py, PyAny>>,
    path: &str,
    active_containers: &mut HashSet<usize>,
) -> PyResult<Vec<AttrValue>> {
    values
        .enumerate()
        .map(|(index, value)| {
            attr_value_from_python(&value, &format!("{path}[{index}]"), active_containers)
        })
        .collect()
}

fn python_type_name(value: &Bound<'_, PyAny>) -> String {
    value
        .get_type()
        .qualname()
        .ok()
        .and_then(|name| name.to_str().ok().map(str::to_owned))
        .unwrap_or_else(|| "<unknown>".to_owned())
}

fn with_container<T>(
    value: &Bound<'_, PyAny>,
    path: &str,
    active_containers: &mut HashSet<usize>,
    convert: impl FnOnce(&mut HashSet<usize>) -> PyResult<T>,
) -> PyResult<T> {
    let identity = value.as_ptr() as usize;
    if !active_containers.insert(identity) {
        return Err(PyTypeError::new_err(format!(
            "recursive graph attribute container at {path} is not supported"
        )));
    }
    let result = convert(active_containers);
    active_containers.remove(&identity);
    result
}
