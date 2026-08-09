//! Python conversion for runtime-owned dtype descriptors.

pub(crate) use advect_runtime::DTypeDescriptor;
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyString, PyTuple, PyType};

const BUILTIN_DTYPE_NAMES: &[&str] = &["bool", "int", "float", "complex"];
const DTYPE_NAMESPACE_ROOTS: &[&str] = &["array_api_strict", "cupy", "numpy"];

#[derive(Debug, Default)]
pub(crate) struct DTypeCache {
    entries: Vec<(Py<PyAny>, DTypeDescriptor)>,
}

impl DTypeCache {
    pub(crate) fn resolve(
        &mut self,
        py: Python<'_>,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<DTypeDescriptor> {
        if let Some((_owner, descriptor)) = self
            .entries
            .iter()
            .find(|(owner, _)| owner.bind(py).is(value))
        {
            return Ok(descriptor.clone());
        }
        let descriptor = dtype_from_python(value)?;
        self.entries
            .push((value.clone().unbind(), descriptor.clone()));
        Ok(descriptor)
    }

    pub(crate) fn resolve_sequence(
        &mut self,
        py: Python<'_>,
        values: &Bound<'_, PyAny>,
    ) -> PyResult<Vec<DTypeDescriptor>> {
        if let Ok(values) = values.cast_exact::<PyList>() {
            return values
                .iter()
                .map(|value| self.resolve(py, &value))
                .collect();
        }
        if let Ok(values) = values.cast_exact::<PyTuple>() {
            return values
                .iter()
                .map(|value| self.resolve(py, &value))
                .collect();
        }
        Err(PyTypeError::new_err(format!(
            "output_dtypes must be a list or tuple of dtype descriptors; got {}",
            values.get_type().qualname()?
        )))
    }
}

pub(crate) fn dtype_from_python(value: &Bound<'_, PyAny>) -> PyResult<DTypeDescriptor> {
    if let Ok(value) = value.cast_exact::<PyString>() {
        return descriptor_from_name(value.to_str()?);
    }
    if let Ok(dtype_type) = value.cast::<PyType>() {
        let module = dtype_type
            .getattr("__module__")?
            .cast_into::<PyString>()?
            .to_str()?
            .to_owned();
        let name = dtype_type
            .getattr("__name__")?
            .cast_into::<PyString>()?
            .to_str()?
            .to_owned();
        if (module == "builtins" && BUILTIN_DTYPE_NAMES.contains(&name.as_str()))
            || is_dtype_namespace(&module)
        {
            return descriptor_from_name(&name);
        }
        return Err(unsupported_dtype(value, &module));
    }

    let module = value
        .get_type()
        .getattr("__module__")?
        .cast_into::<PyString>()?
        .to_str()?
        .to_owned();
    if !is_dtype_namespace(&module) {
        return Err(unsupported_dtype(value, &module));
    }
    if (module == "numpy" || module.starts_with("numpy."))
        && let Ok(name) = value.getattr("name")
        && let Ok(name) = name.cast_into::<PyString>()
    {
        let display = name.to_str()?;
        if let Ok(byte_order) = value.getattr("byteorder")
            && let Ok(byte_order) = byte_order.cast_into::<PyString>()
            && matches!(byte_order.to_str()?, ">" | "<")
            && !value.getattr("isnative")?.is_truthy()?
            && let Ok(raw) = value.getattr("str")
            && let Ok(raw) = raw.cast_into::<PyString>()
        {
            return descriptor_from_name(raw.to_str()?);
        }
        return descriptor_from_name(display);
    }
    if let Ok(name) = value.getattr("name")
        && let Ok(name) = name.cast_into::<PyString>()
    {
        return descriptor_from_name(name.to_str()?);
    }
    descriptor_from_name(value.str()?.to_str()?)
}

pub(crate) fn dtype_to_python(dtype: &DTypeDescriptor, py: Python<'_>) -> PyResult<Py<PyAny>> {
    dtype.name().into_py_any(py)
}

fn descriptor_from_name(name: &str) -> PyResult<DTypeDescriptor> {
    DTypeDescriptor::from_name(name).map_err(|error| PyValueError::new_err(error.into_message()))
}

fn is_dtype_namespace(module: &str) -> bool {
    DTYPE_NAMESPACE_ROOTS
        .iter()
        .any(|root| module == *root || module.starts_with(&format!("{root}.")))
}

fn unsupported_dtype(value: &Bound<'_, PyAny>, module: &str) -> PyErr {
    let type_name = value
        .get_type()
        .qualname()
        .map_or_else(|_| "<unknown>".to_owned(), |name| name.to_string());
    PyTypeError::new_err(format!(
        "unsupported dtype object {type_name} from module {module:?}; pass a dtype string, a \
         built-in numeric type, or a supported array-backend dtype"
    ))
}
