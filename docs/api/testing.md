# Testing Utilities

`check_gradient` checks one representative composition against directional
finite differences and reverse-mode consistency. `check_primitive` validates
the capabilities requested for one representative custom-primitive call.

They complement application tests; neither utility can prove that the
implemented function is the mathematics you intended.

::: advect.testing
