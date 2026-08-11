# Developer Guide

Advect is organized around one semantic core, explicit array frontends, and two execution lifetimes. Start with the part of the system you intend to change; the design archive is supporting rationale, not required reading for every contribution.

## Find the owning guide

| If you need to…                                                                                                    | Read                                                                                                |
| ------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------- |
| Locate a responsibility or check an import boundary                                                                | [Codebase map](https://yaugenst.github.io/advect/0.1.0/development/codebase/index.md)               |
| Add a primitive, frontend form, provider feature, staged-program envelope, runtime graph, or native adapter change | [Adding operations](https://yaugenst.github.io/advect/0.1.0/development/adding-operations/index.md) |
| Choose the owning test suite and verification commands                                                             | [Testing](https://yaugenst.github.io/advect/0.1.0/development/testing/index.md)                     |
| Write public API docs, module docstrings, or repository guidance                                                   | [Documentation](https://yaugenst.github.io/advect/0.1.0/development/documentation/index.md)         |

The [architecture overview](https://yaugenst.github.io/advect/0.1.0/architecture/index.md) explains the public execution model. The repository's [`design/` index](https://github.com/yaugenst/advect/blob/main/design/README.md) routes requirements, decisions, implementation status, and performance evidence when a change needs the deeper rationale.

## Working agreement

Keep each change at one owning boundary and preserve one source of truth for every contract. A support declaration must be executable. Python core owns the versioned `StagedProgram` envelope, while `advect-runtime` owns its enclosed canonical graph artifact. A frontend must emit canonical operations rather than grow a second semantic registry.

Work on a feature branch. While iterating, run the focused checks for the changed contract; before requesting review, run every required gate from the [testing guide](https://yaugenst.github.io/advect/0.1.0/development/testing/index.md) and name any unavailable hardware or remote lane.
