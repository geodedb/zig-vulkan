# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Benchmark regression tracking with configurable thresholds
- Example compute shaders for ML operations (ReLU, softmax, conv2d)
- Example compute shaders for algorithms (reduction, prefix sum)
- GPU runner support in GitLab CI
- API documentation generation

### Changed
- Updated compile.sh to handle both `.comp` and `.comp.glsl` shader files

## [0.6.0] - 2025-12-31

### Added

#### Core Features
- Multi-queue support in ComputeGraph with `TaskQueueHint`
- AsyncCoordinator integration with ComputeGraph (`submitAsync`, `submitAsyncAndWait`)
- HotReloadablePipeline for shader hot-reload during development
- ShaderCache for GLSL/HLSL compilation caching with disk persistence
- DOT graph visualization export for ComputeGraph debugging
- VkEvent type support for fine-grained GPU synchronization
- GPU timestamp profiling in ComputeGraph

#### Production Features
- Device lost recovery with automatic resource recreation
- Memory pressure handling with eviction callbacks
- Memory budget monitoring and alerts
- Comprehensive error context with diagnostic suggestions

#### Developer Experience
- HLSL to SPIR-V compilation via DXC
- Shader reflection for automatic descriptor layout
- Debug labels for RenderDoc/NSight integration

### Changed
- Updated to Zig 0.15.1 API compatibility
- Improved error messages with actionable suggestions

### Fixed
- CommandBufferResetFlags initialization for Zig 0.15
- File I/O API changes (readToEndAlloc, buffered writer)
- StdIo enum capitalization for process spawning

## [0.5.0] - 2025-12-30

### Added
- Pooled memory allocator with suballocation support
- Timeline semaphores for advanced synchronization
- Cross-queue synchronization utilities
- Multi-device context for multi-GPU systems
- Pipeline cache with disk persistence
- Pipeline statistics queries
- Indirect dispatch support
- Specialization constants
- Push constants

### Changed
- Improved buffer memory allocation strategies
- Better device selection heuristics

## [0.4.0] - 2025-12-29

### Added
- GLSL to SPIR-V runtime compilation
- Validation layer integration
- Timestamp queries for GPU profiling
- Descriptor update templates

### Fixed
- Memory alignment issues on some drivers
- Validation errors with certain descriptor combinations

## [0.3.0] - 2025-12-28

### Added
- Mock testing framework for CI without GPU
- Compute graph (DAG) for automatic dependency scheduling
- Extension capability queries (subgroups, descriptor indexing, etc.)

### Changed
- Refactored command buffer API for better ergonomics

## [0.2.0] - 2025-12-27

### Added
- Cross-platform support (Linux, Windows, macOS)
- Graceful degradation when Vulkan unavailable
- Buffer usage hints (storage, staging, readback, uniform)
- Fence and semaphore synchronization

### Changed
- Dynamic Vulkan loader instead of static linking

## [0.1.0] - 2025-12-26

### Added
- Initial release
- Basic Vulkan compute context
- Buffer creation and memory management
- SPIR-V shader module loading
- Compute pipeline creation
- Command buffer recording and submission
- Basic synchronization (fences)

[Unreleased]: https://gitlab.com/devnw/zig/vulkan/-/compare/v0.6.0...main
[0.6.0]: https://gitlab.com/devnw/zig/vulkan/-/compare/v0.5.0...v0.6.0
[0.5.0]: https://gitlab.com/devnw/zig/vulkan/-/compare/v0.4.0...v0.5.0
[0.4.0]: https://gitlab.com/devnw/zig/vulkan/-/compare/v0.3.0...v0.4.0
[0.3.0]: https://gitlab.com/devnw/zig/vulkan/-/compare/v0.2.0...v0.3.0
[0.2.0]: https://gitlab.com/devnw/zig/vulkan/-/compare/v0.1.0...v0.2.0
[0.1.0]: https://gitlab.com/devnw/zig/vulkan/-/releases/v0.1.0
