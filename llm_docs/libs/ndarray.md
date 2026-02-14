# ndarray: How to Access and Explore Documentation

## Summary

This document explains HOW to find and navigate ndarray documentation, NOT the ndarray API itself. For actual ndarray docs, generate them locally or use online resources.

## Prerequisites

The project must already depend on ndarray. Verify in `Cargo.toml`:

```toml
[dependencies]
ndarray = "0.17.2"  # or current version
```

## Method 1: Generate Local Documentation (Recommended)

### Generate Docs for All Dependencies

```bash
cargo doc
```

This generates HTML documentation for:
- Your project (`bytecode2mlir`)
- All direct dependencies (including `ndarray`)
- Transitive dependencies

Output directory: `target/doc/`

### Generate Docs Without Dependencies

```bash
cargo doc --no-deps
```

Only generates docs for your project, but you can still access already-generated ndarray docs from previous builds.

### Document Private Items

```bash
cargo doc --document-private-items
```

Includes private methods, fields, and traits in documentation.

### Open in Browser

```bash
cargo doc --open
```

Generates docs and automatically opens them in your default browser.

## Method 2: Direct File Access

### ndarray Array Documentation

After running `cargo doc`, the main ndarray documentation is at:

```
target/doc/ndarray/index.html
```

Key files for ndarray:

| File | Description |
|------|-------------|
| `target/doc/ndarray/index.html` | Crate overview and modules |
| `target/doc/ndarray/struct.Array.html` | Main `Array` struct docs |
| `target/doc/ndarray/struct.ArrayBase.html` | `ArrayBase` - base implementation |
| `target/doc/ndarray/struct.ArrayView.html` | View types |
| `target/doc/ndarray/struct.CowArray.html` | Copy-on-write array |
| `target/doc/ndarray/enum.Order.html` | Memory order (C vs Fortran) |
| `target/doc/ndarray/trait.Dimension.html` | Dimension traits |

### Quick Command Line Access

View specific documentation sections without opening a browser:

```bash
# Extract is_standard_layout docs
sed -n '/id="method\.is_standard_layout"/,/\/section/p' \
  target/doc/ndarray/struct.ArrayBase.html | \
  sed 's/<[^>]*>//g' | \
  sed 's/&quot;/"/g' | \
  tr -s '[:space:]' ' '

# Extract as_standard_layout docs
sed -n '/id="method\.as_standard_layout"/,/\/section/p' \
  target/doc/ndarray/struct.ArrayBase.html | \
  sed 's/<[^>]*>//g' | \
  sed 's/&quot;/"/g' | \
  tr -s '[:space:]' ' '
```

## Method 3: Search Documentation

### Search for Specific Methods

```bash
# Search for method names in docs
grep -r "pub fn is_standard_layout" target/doc/ndarray/

# Find all methods mentioning "contiguous"
grep -r "contiguous" target/doc/ndarray/*.html | head -20
```

### Find Related Methods

```bash
# Find all layout-related methods
grep -r "layout\|strides\|order" target/doc/ndarray/*.html | \
  grep -E "fn |method " | \
  head -30
```

## Method 4: View Source Code Links

Documentation pages include "Source" links pointing to implementation:

1. Open `target/doc/ndarray/struct.ArrayBase.html`
2. Find method (e.g., `is_standard_layout`)
3. Click "Source" link (e.g., `src/ndarray/impl_methods.rs.html#1623-1626`)

This shows the actual implementation, which is often clearer than documentation.

## Method 5: Online Documentation (Alternative)

If local docs aren't available:

- Official docs: https://docs.rs/ndarray/
- Version-specific: https://docs.rs/ndarray/{version}/
- GitHub source: https://github.com/rust-ndarray/ndarray

**WARNING**: Always check the version matches your `Cargo.toml`.

## Common Documentation Tasks

### Task: Check if Array is C-Contiguous

1. Generate docs: `cargo doc`
2. Open file: `target/doc/ndarray/struct.ArrayBase.html`
3. Search for: "contiguous" or "standard layout"
4. Result: `is_standard_layout()` method

### Task: Make Array C-Contiguous

1. Same file: `target/doc/ndarray/struct.ArrayBase.html`
2. Search for: "standard_layout" or "as_standard"
3. Result: `as_standard_layout()` method

### Task: Find All Array Methods

```bash
# List all methods on ArrayBase
grep -o 'id="method\.[^"]*"' target/doc/ndarray/struct.ArrayBase.html | \
  sed 's/id="method\.//g' | \
  sed 's/"//g' | \
  sort
```

## Documentation Structure Tips

### Trait Implementations

Scroll to "Trait Implementations" section to see:
- `IntoIterator`
- `From<Vec>>`
- `Clone`
- `PartialEq`

### Related Types

Click type links in signatures to navigate:
- Click `CowArray` to see copy-on-write docs
- Click `Dimension` to understand dimension constraints
- Click `Order` to see C/Fortran layout options

### Example Code

Look for:
- Examples section under methods
- `[src]` links to see test code using the method
- Module-level examples at `target/doc/ndarray/index.html`

## Troubleshooting

### Docs Not Generated

```bash
# Force rebuild of docs
cargo clean && cargo doc
```

### Wrong Version Documented

```bash
# Update to match Cargo.lock
cargo update
cargo doc
```

### Private Items Missing

```bash
# Add --document-private-items flag
cargo doc --document-private-items
```

## Integration with Development Workflow

### Before Implementing ndarray Operations

1. Generate fresh docs: `cargo doc`
2. Identify relevant methods
3. Check method signatures and constraints
4. Review examples if available
5. Check source code for implementation details

### Quick Reference During Coding

```bash
# Terminal 1: Generate and serve docs
cargo doc --open

# Terminal 2: Quick search
grep -r "method_name" target/doc/ndarray/
```

### Documentation for Specific ndarray Version

```bash
# Check current version in Cargo.toml
grep ndarray Cargo.toml

# Download exact version docs
cargo doc --no-deps
# ndarray docs will be at correct version
```
