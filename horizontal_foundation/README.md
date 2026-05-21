# Horizontal Foundation Layer

The horizontal baseline that all layers of the OpenLogic Finance ecosystem sit on. It provides shared utilities, primitive base classes, and global configurations.

## Target Structure & Subdirectories

- **`config/`**: Global environment setups, credentials, API key managers, and core system parameters.
- **`utils/`**: Shared logging, advanced math utilities, data structure helpers, and general system helper functions.
- **`core/`**: Base primitives, abstract base classes, and standard interfaces used universally across other packages.

## Purpose & Architectural Rule

This layer represents a **strict dependency boundary**: No component inside `horizontal_foundation/` is allowed to import from any higher-level box (such as `model_library`, `strategy_testing`, or `agentic_workflows`). It serves purely as the horizontal infrastructure foundation.
