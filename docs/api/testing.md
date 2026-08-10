# Testing Utilities

[`check_gradient`](testing.md#advect.testing.check_gradient) checks one
representative composition against directional finite differences and
reverse-mode consistency.
[`check_primitive`](testing.md#advect.testing.check_primitive) validates the
capabilities requested for one representative custom-primitive call.

They complement application tests; neither utility can prove that the
implemented function is the mathematics you intended.

See [Troubleshooting](../tutorials/debugging.md#check-a-suspicious-gradient) for
a composed gradient check and [Custom primitives](../tutorials/primitives.md#check-the-promised-capabilities)
for the extension-author workflow.

::: advect.testing
