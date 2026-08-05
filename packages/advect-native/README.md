# advect-native

`advect-native` is Advect's required PyO3 adapter. Maturin builds it as the private
`advect._native_core` extension inside the `advect` wheel; it is not a
separately installable distribution.

The adapter owns Python-specific runtime state:

- invocation-local dynamic-tape values, literals, callbacks, and residual
  handles;
- reentrant JVP and VJP callback invocation;
- Python conversion and exception mapping;
- thin Python handles over runtime-owned graph builders and stores;
- `PythonHost`, which links and evaluates Python providers through the
  host-independent staged execution schedule and clones Python handles for
  repeated flat graph outputs.

Durable graph authority lives in the PyO3-free `advect-runtime` crate:
`RawArena`, graph metadata, portable constants, validation, canonical
serialization, fixed cleanup, topology, use counts, conservative alias-root
sets, and execution planning. `PythonHost` validates inputs, constants, and
every evaluated output leaf. `advect-native` does not maintain a second durable
graph model, optimizer, serializer, or staged execution loop.

Dynamic autodiff uses this extension for its invocation-local tape and direct
forward/reverse traversal. It does not convert that tape to `GraphStore` or run
staged cleanup before backward.
