//! Tests for the safety invariants of the unsafe blocks in
//! rust_memory/src/checkpoint.rs and lib.rs.
//!
//! The pre-remediation code had multiple `unsafe { ... }` blocks
//! with no `// SAFETY:` documentation. The audit M1 finding
//! requires that the invariants be documented AND that they
//! be exercisable as tests where possible.

use crate::checkpoint::{
    CheckpointHeader, HEADER_MAGIC, HEADER_VERSION, serialize_checkpoint,
    HEADER_SIZE,
};

#[test]
fn unsafe_block_documented_in_checkpoint_rs() {
    // The audit requires that every `unsafe {` block be
    // preceded by a `// SAFETY:` comment. The check is
    // implemented as a string-level grep because the unsafe
    // blocks are tied to the lifetime of a buffer and can't
    // be safely exercised from a test.
    let src = include_str!("../src/checkpoint.rs");
    let mut search_start = 0;
    let mut found_documented = 0;
    let mut found_undocumented = 0;
    while let Some(idx) = src[search_start..].find("unsafe {") {
        let abs = search_start + idx;
        // Look back 500 bytes for a SAFETY: comment.
        let window_start = abs.saturating_sub(500);
        let window = &src[window_start..abs];
        if window.contains("SAFETY:") {
            found_documented += 1;
        } else {
            found_undocumented += 1;
        }
        search_start = abs + 1;
    }
    assert_eq!(
        found_undocumented, 0,
        "checkpoint.rs: {} unsafe block(s) missing SAFETY: comment",
        found_undocumented
    );
    assert!(
        found_documented >= 1,
        "checkpoint.rs must have at least one documented unsafe block; found {}",
        found_documented
    );
}

#[test]
fn unsafe_block_documented_in_lib_rs() {
    let src = include_str!("../src/lib.rs");
    let mut search_start = 0;
    let mut found_documented = 0;
    let mut found_undocumented = 0;
    while let Some(idx) = src[search_start..].find("unsafe {") {
        let abs = search_start + idx;
        let window_start = abs.saturating_sub(500);
        let window = &src[window_start..abs];
        if window.contains("SAFETY:") {
            found_documented += 1;
        } else {
            found_undocumented += 1;
        }
        search_start = abs + 1;
    }
    assert_eq!(
        found_undocumented, 0,
        "lib.rs: {} unsafe block(s) missing SAFETY: comment",
        found_undocumented
    );
    assert!(
        found_documented >= 1,
        "lib.rs must have at least one documented unsafe block; found {}",
        found_documented
    );
}

#[test]
fn header_layout_invariants() {
    // The header is read as a raw byte slice via
    // `std::slice::from_raw_parts`. The invariant: HEADER_SIZE
    // must match the actual size of the serialized
    // CheckpointHeader. If CheckpointHeader grows a new
    // field, HEADER_SIZE must be updated.
    let serialized = serialize_checkpoint(&CheckpointHeader {
        magic: HEADER_MAGIC,
        version: HEADER_VERSION,
        entry_count: 0,
        total_bytes: 0,
        created_at: 0,
    });
    assert_eq!(
        serialized.len(),
        HEADER_SIZE,
        "CheckpointHeader size ({}) must match HEADER_SIZE constant ({})",
        serialized.len(),
        HEADER_SIZE
    );
}

