//! Lowercase hexadecimal encoding shared by durable wire formats.

pub(crate) fn encode(bytes: &[u8]) -> String {
    let mut encoded = String::with_capacity(bytes.len().saturating_mul(2));
    for &byte in bytes {
        encoded.push(char::from(digit(byte >> 4)));
        encoded.push(char::from(digit(byte & 0x0f)));
    }
    encoded
}

pub(crate) fn decode(encoded: &str) -> Result<Vec<u8>, &'static str> {
    if !encoded.len().is_multiple_of(2) {
        return Err("must contain an even number of characters");
    }
    encoded
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            let Some(&high) = pair.first() else {
                return Err("is missing a high digit");
            };
            let Some(&low) = pair.get(1) else {
                return Err("is missing a low digit");
            };
            Ok((decode_digit(high)? << 4) | decode_digit(low)?)
        })
        .collect()
}

pub(crate) const fn digit(value: u8) -> u8 {
    if value < 10 {
        b'0' + value
    } else {
        b'a' + (value - 10)
    }
}

const fn decode_digit(value: u8) -> Result<u8, &'static str> {
    match value {
        b'0'..=b'9' => Ok(value - b'0'),
        b'a'..=b'f' => Ok(value - b'a' + 10),
        _ => Err("must use lowercase hexadecimal"),
    }
}
