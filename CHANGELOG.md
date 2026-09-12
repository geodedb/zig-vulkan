# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.14] - 2026-09-12

### Changed

- Converge thread_pool v0.2.7, fips v0.1.1, protobuf v0.1.5, config v0.1.13 (GAP-0838 diamond sync).

## [0.1.13] - 2026-09-12

### Changed

- Converge thread_pool v0.2.7, fips v0.1.1, protobuf v0.1.5 (GAP-0838 diamond sync).

## [0.1.0] - 2026-01-24

### Added

#### Core Features
- Basic Vulkan compute context with cross-platform support (Linux, Windows, macOS)
- Dynamic Vulkan loader with graceful degradation when Vulkan unavailable
- Buffer creation and memory management with usage hints (storage, staging, readback, uniform)
- SPIR-V shader module loading
- Compute pipeline creation with specialization constants and push constants
- Command buffer recording and submission
- Fence and semaphore synchronization
- Timeline semaphores for advanced synchronization
- Cross-queue synchronization utilities

#### Compute Graph
- Compute graph (DAG) for automatic dependency scheduling
- Multi-queue support with `TaskQueueHint`
- AsyncCoordinator integration (`submitAsync`, `submitAsyncAndWait`)
- DOT graph visualization export for debugging
- GPU timestamp profiling

#### Memory Management
- Pooled memory allocator with suballocation support
- Multi-device context for multi-GPU systems
- Memory pressure handling with eviction callbacks
- Memory budget monitoring and alerts

#### Shader Tooling
- GLSL to SPIR-V runtime compilation
- HLSL to SPIR-V compilation via DXC
- Shader reflection for automatic descriptor layout
- HotReloadablePipeline for shader hot-reload during development
- ShaderCache for compilation caching with disk persistence

#### Pipeline Features
- Pipeline cache with disk persistence
- Pipeline statistics queries
- Indirect dispatch support
- Descriptor update templates

#### Production Features
- Device lost recovery with automatic resource recreation
- Comprehensive error context with diagnostic suggestions
- Validation layer integration
- VkEvent type support for fine-grained GPU synchronization
- Debug labels for RenderDoc/NSight integration

#### Testing and CI
- Mock testing framework for CI without GPU
- Extension capability queries (subgroups, descriptor indexing, etc.)
- Timestamp queries for GPU profiling
- Benchmark regression tracking with configurable thresholds
- GPU runner support in GitLab CI
- API documentation generation

#### Examples
- Example compute shaders for ML operations (ReLU, softmax, conv2d)
- Example compute shaders for algorithms (reduction, prefix sum)

[Unreleased]: https://gitlab.com/devnw/zig/vulkan/-/compare/v0.1.0...main
[0.1.0]: https://gitlab.com/devnw/zig/vulkan/-/releases/v0.1.0
