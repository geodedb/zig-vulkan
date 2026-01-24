.PHONY: build test fmt check clean

ZIG ?= zig

build:
	$(ZIG) build

test:
	$(ZIG) build test --summary all

fmt:
	$(ZIG) fmt src/ lib/ 2>/dev/null || $(ZIG) fmt src/

check:
	$(ZIG) fmt --check src/ lib/ 2>/dev/null || $(ZIG) fmt --check src/

clean:
	rm -rf zig-out zig-cache .zig-cache

.DEFAULT_GOAL := build
