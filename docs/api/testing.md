# Testing Utilities

`check_gradient` checks an ordinary unary composition against directional
finite differences and reverse-mode consistency. `check_primitive` validates
the narrower rule contract of one custom primitive authoring handle.

The primitive check defaults to `abstract`, `jvp`, and `transpose`: a useful
first-order smoke test, but not a staging or mutation claim. Request `stage`
explicitly to compile and restore the program, validate output metadata, and
check input preservation during both staged executions. Request `nested` and a
separate complex-valued `complex` case when those capabilities are supported.
Choose a check tuple that matches the rules the primitive promises. A
transpose-only primitive can request `transpose` without a JVP; the checker
compares its explicit transpose with a central finite difference. Forward-mode
`jvp`, `complex`, and `nested` checks require a JVP.

These utilities help extension authors; they do not replace application tests
or prove that the implemented function represents the intended mathematics.

::: advect.testing
