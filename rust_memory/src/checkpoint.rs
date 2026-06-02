//! Checkpoint deserializer.
//!
//! The pre-remediation checkpoint module only had writers
//! (MmapCheckpointWriter, DifferentialCheckpointer) but
//! no reader. The audit M1 finding requires a reader that
//! validates malformed inputs: truncated payload, oversized
//! length, bad alignment, and header mismatch. This module
//! provides a minimal `deserialize_header` function plus a
//! `validate_payload` function that checks the byte length
//! against the header's `total_bytes` claim.


/// Magic bytes: "AURLCKPT" (Aurelius Checkpoint).
pub const MAGIC: [u8; 8] = *b"AURLCKPT";
pub const HEADER_SIZE: usize = 32;
/// Current supported version. Older or newer versions are
/// rejected to fail closed.
pub const SUPPORTED_VERSION: u32 = 1;
/// Maximum checkpoint size. A header claiming more than this
/// is rejected (defense in depth against resource exhaustion).
pub const MAX_CHECKPOINT_BYTES: u64 = 16 * 1024 * 1024 * 1024; // 16 GiB

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointHeader {
    pub magic: [u8; 8],
    pub version: u32,
    pub num_tensors: u32,
    pub total_bytes: u64,
    pub step: u64,
    pub timestamp: u64,
}

#[derive(Debug, PartialEq, Eq)]
pub enum CheckpointError {
    /// Input is shorter than the header.
    Truncated { expected: usize, got: usize },
    /// Header claims more bytes than the absolute maximum.
    Oversized { claimed: u64, max: u64 },
    /// Header magic bytes don't match.
    BadMagic,
    /// Header version is not supported.
    BadVersion { version: u32 },
    /// Number of tensors is implausible (sanity bound).
    ImplausibleTensorCount { count: u32, max: u32 },
    /// Alignment / layout mismatch (e.g. payload size is not
    /// a multiple of the tensor record size).
    BadAlignment { offset: u64, alignment: u64 },
    /// Header vs payload length mismatch.
    LengthMismatch { claimed: u64, actual: usize },
}

/// Maximum plausible tensor count. Defense in depth: any
/// header claiming more tensors than this is rejected.
const MAX_TENSOR_COUNT: u32 = 100_000;

/// Deserialize and validate a 32-byte checkpoint header.
///
/// The input is read as raw bytes via `from_raw_parts` inside
/// the MmapCheckpointWriter; the test surface here validates
/// the same byte layout. The header layout is:
///
/// offset  size  field
///   0      8   magic ("AURLCKPT")
///   8      4   version (little-endian u32)
///  12      4   num_tensors (little-endian u32)
///  16      8   total_bytes (little-endian u64)
///  24      8   step (little-endian u64)
///  --- 32 bytes total; the timestamp is not in the wire
///      format for the current version (it's an in-memory
///      field used by the writer for diagnostics).
pub fn deserialize_header(bytes: &[u8]) -> Result<CheckpointHeader, CheckpointError> {
    if bytes.len() < HEADER_SIZE {
        return Err(CheckpointError::Truncated {
            expected: HEADER_SIZE,
            got: bytes.len(),
        });
    }

    // Extract fields by index (the wire format is little-
    // endian by convention; the writer uses to_le_bytes
    // when writing). We parse the magic and version first
    // because those are the most common failure modes.
    let mut magic = [0u8; 8];
    magic.copy_from_slice(&bytes[0..8]);
    if magic != MAGIC {
        return Err(CheckpointError::BadMagic);
    }

    let version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    if version != SUPPORTED_VERSION {
        return Err(CheckpointError::BadVersion { version });
    }

    let num_tensors = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
    if num_tensors > MAX_TENSOR_COUNT {
        return Err(CheckpointError::ImplausibleTensorCount {
            count: num_tensors,
            max: MAX_TENSOR_COUNT,
        });
    }

    let total_bytes = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
    if total_bytes > MAX_CHECKPOINT_BYTES {
        return Err(CheckpointError::Oversized {
            claimed: total_bytes,
            max: MAX_CHECKPOINT_BYTES,
        });
    }

    let step = u64::from_le_bytes(bytes[24..32].try_into().unwrap());

    Ok(CheckpointHeader {
        magic,
        version,
        num_tensors,
        total_bytes,
        step,
        timestamp: 0,
    })
}

/// Validate a payload against a header. Returns the validated
/// payload length, or an error.
///
/// The validation rules:
/// 1. The payload must be at least `total_bytes` long.
/// 2. The payload's first `total_bytes` bytes must be aligned
///    to a 4-byte boundary (the tensor record size).
/// 3. The actual payload length must be a multiple of the
///    alignment.
pub fn validate_payload(header: &CheckpointHeader, payload: &[u8]) -> Result<usize, CheckpointError> {
    let claimed = header.total_bytes as usize;
    if payload.len() < claimed {
        return Err(CheckpointError::LengthMismatch {
            claimed: header.total_bytes,
            actual: payload.len(),
        });
    }
    // 4-byte alignment is the tensor record size for f32
    // tensors. A misaligned payload is malformed.
    if claimed % 4 != 0 {
        return Err(CheckpointError::BadAlignment {
            offset: claimed as u64,
            alignment: 4,
        });
    }
    if payload.len() % 4 != 0 {
        return Err(CheckpointError::BadAlignment {
            offset: payload.len() as u64,
            alignment: 4,
        });
    }
    Ok(claimed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_header_bytes(magic: [u8; 8], version: u32, num_tensors: u32, total_bytes: u64) -> Vec<u8> {
        let mut v = vec![0u8; HEADER_SIZE];
        v[0..8].copy_from_slice(&magic);
        v[8..12].copy_from_slice(&version.to_le_bytes());
        v[12..16].copy_from_slice(&num_tensors.to_le_bytes());
        v[16..24].copy_from_slice(&total_bytes.to_le_bytes());
        // step is 0
        v
    }

    #[test]
    fn truncated_input_rejected() {
        let bytes = vec![0u8; 16];
        let err = deserialize_header(&bytes).unwrap_err();
        assert!(matches!(err, CheckpointError::Truncated { .. }));
    }

    #[test]
    fn bad_magic_rejected() {
        let mut bytes = build_header_bytes(*b"WRONGMGC", 1, 0, 0);
        bytes.extend_from_slice(&[0u8; 32]);
        let err = deserialize_header(&bytes).unwrap_err();
        assert_eq!(err, CheckpointError::BadMagic);
    }

    #[test]
    fn bad_version_rejected() {
        let bytes = build_header_bytes(MAGIC, 99, 0, 0);
        let err = deserialize_header(&bytes).unwrap_err();
        assert!(matches!(err, CheckpointError::BadVersion { version: 99 }));
    }

    #[test]
    fn oversized_rejected() {
        let bytes = build_header_bytes(MAGIC, 1, 0, MAX_CHECKPOINT_BYTES + 1);
        let err = deserialize_header(&bytes).unwrap_err();
        assert!(matches!(err, CheckpointError::Oversized { .. }));
    }

    #[test]
    fn implausible_count_rejected() {
        let bytes = build_header_bytes(MAGIC, 1, MAX_TENSOR_COUNT + 1, 0);
        let err = deserialize_header(&bytes).unwrap_err();
        assert!(matches!(err, CheckpointError::ImplausibleTensorCount { .. }));
    }

    #[test]
    fn valid_header_accepted() {
        let bytes = build_header_bytes(MAGIC, 1, 3, 256);
        let h = deserialize_header(&bytes).unwrap();
        assert_eq!(h.num_tensors, 3);
        assert_eq!(h.total_bytes, 256);
    }

    #[test]
    fn validate_payload_length_mismatch() {
        let h = CheckpointHeader {
            magic: MAGIC,
            version: 1,
            num_tensors: 0,
            total_bytes: 1024,
            step: 0,
            timestamp: 0,
        };
        let payload = vec![0u8; 512];
        let err = validate_payload(&h, &payload).unwrap_err();
        assert!(matches!(err, CheckpointError::LengthMismatch { .. }));
    }

    #[test]
    fn validate_payload_misaligned_claimed() {
        let h = CheckpointHeader {
            magic: MAGIC,
            version: 1,
            num_tensors: 0,
            total_bytes: 7, // not a multiple of 4
            step: 0,
            timestamp: 0,
        };
        let payload = vec![0u8; 1024];
        let err = validate_payload(&h, &payload).unwrap_err();
        assert!(matches!(err, CheckpointError::BadAlignment { .. }));
    }
}
