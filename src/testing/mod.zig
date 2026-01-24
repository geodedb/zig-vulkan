// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024-2025 CodePros

//! Testing Utilities
//!
//! Provides mock Vulkan implementations and test helpers for unit testing
//! without requiring actual GPU hardware.

pub const mock = @import("mock.zig");

// Re-export commonly used types
pub const MockContext = mock.MockContext;
pub const MockDeviceProperties = mock.MockDeviceProperties;
pub const MockOperation = mock.MockOperation;
pub const MockStats = mock.MockStats;
pub const ErrorInjection = mock.ErrorInjection;

// Test utilities
pub const createMinimalSpirv = mock.createMinimalSpirv;
pub const assertNoLeaks = mock.assertNoLeaks;

test {
    @import("std").testing.refAllDecls(@This());
}
