// Quantization Types
// Defines quantized data types and their parameters

/// Quantization data type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantType {
    /// 8-bit signed integer (-128 to 127)
    /// Memory: 4x reduction from FP32
    INT8,

    /// 4-bit signed integer (-8 to 7)
    /// Memory: 8x reduction from FP32
    /// Note: Usually stored in u8, 2 values per byte
    INT4,

    /// 16-bit floating point
    /// Memory: 2x reduction from FP32
    /// Better dynamic range than INT8
    FP16,

    /// Brain Float 16 (BF16)
    /// Same range as FP32, less precision
    /// Good for training
    BF16,
}

impl QuantType {
    /// Get the number of bits for this type
    pub fn bits(&self) -> u8 {
        match self {
            QuantType::INT8 => 8,
            QuantType::INT4 => 4,
            QuantType::FP16 => 16,
            QuantType::BF16 => 16,
        }
    }

    /// Get the memory reduction factor vs FP32
    pub fn reduction_factor(&self) -> u8 {
        match self {
            QuantType::INT8 => 4,
            QuantType::INT4 => 8,
            QuantType::FP16 => 2,
            QuantType::BF16 => 2,
        }
    }

    /// Get min/max representable values
    pub fn range(&self) -> (i32, i32) {
        match self {
            QuantType::INT8 => (-128, 127),
            QuantType::INT4 => (-8, 7),
            // FP16/BF16 have dynamic range, return INT8 range as approximation
            QuantType::FP16 | QuantType::BF16 => (-128, 127),
        }
    }

    /// Check if this is an integer quantization type
    pub fn is_integer(&self) -> bool {
        matches!(self, QuantType::INT8 | QuantType::INT4)
    }

    /// Check if this is a floating point type
    pub fn is_float(&self) -> bool {
        matches!(self, QuantType::FP16 | QuantType::BF16)
    }
}

/// Quantization parameters
///
/// Formula: quantized = round(value / scale) + zero_point
/// Inverse: value = (quantized - zero_point) * scale
#[derive(Debug, Clone)]
pub struct QuantParams {
    /// Scale factor for quantization
    pub scale: f32,

    /// Zero point (offset) for quantization
    /// For symmetric quantization, this is 0
    pub zero_point: i32,

    /// Quantization type
    pub quant_type: QuantType,
}

impl QuantParams {
    /// Create quantization parameters from min/max values
    ///
    /// # Arguments
    /// * `min` - Minimum value in the data
    /// * `max` - Maximum value in the data
    /// * `quant_type` - Target quantization type
    /// * `symmetric` - Use symmetric quantization (zero_point = 0)
    pub fn from_min_max(min: f32, max: f32, quant_type: QuantType, symmetric: bool) -> Self {
        let (qmin, qmax) = quant_type.range();
        let qmin = qmin as f32;
        let qmax = qmax as f32;

        if symmetric {
            // Symmetric: zero_point = 0, scale based on max absolute value
            let abs_max = min.abs().max(max.abs());
            let scale = abs_max / qmax;

            QuantParams {
                scale,
                zero_point: 0,
                quant_type,
            }
        } else {
            // Asymmetric: map [min, max] to [qmin, qmax]
            let scale = (max - min) / (qmax - qmin);
            let zero_point = (qmin - min / scale).round() as i32;

            QuantParams {
                scale,
                zero_point,
                quant_type,
            }
        }
    }

    /// Create symmetric INT8 quantization parameters
    pub fn int8_symmetric(abs_max: f32) -> Self {
        QuantParams {
            scale: abs_max / 127.0,
            zero_point: 0,
            quant_type: QuantType::INT8,
        }
    }

    /// Create symmetric INT4 quantization parameters
    pub fn int4_symmetric(abs_max: f32) -> Self {
        QuantParams {
            scale: abs_max / 7.0,
            zero_point: 0,
            quant_type: QuantType::INT4,
        }
    }

    /// Quantize a single f32 value
    pub fn quantize(&self, value: f32) -> i32 {
        let quantized = (value / self.scale).round() as i32 + self.zero_point;
        let (qmin, qmax) = self.quant_type.range();
        quantized.clamp(qmin, qmax)
    }

    /// Dequantize a single quantized value back to f32
    pub fn dequantize(&self, quantized: i32) -> f32 {
        (quantized - self.zero_point) as f32 * self.scale
    }
}

/// A tensor with quantized values
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    /// Quantized data (stored as i8 or packed i4)
    pub data: Vec<i8>,

    /// Original shape of the tensor
    pub shape: Vec<usize>,

    /// Quantization parameters
    pub params: QuantParams,

    /// Whether data is packed (for INT4)
    pub packed: bool,
}

impl QuantizedTensor {
    /// Create a new quantized tensor
    pub fn new(data: Vec<i8>, shape: Vec<usize>, params: QuantParams) -> Self {
        QuantizedTensor {
            data,
            shape,
            params,
            packed: false,
        }
    }

    /// Get the total number of elements
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Get the memory size in bytes
    pub fn memory_bytes(&self) -> usize {
        if self.packed && self.params.quant_type == QuantType::INT4 {
            // INT4: 2 values per byte
            self.numel().div_ceil(2)
        } else {
            // INT8: 1 byte per value
            self.numel()
        }
    }

    /// Get the memory reduction factor vs FP32
    pub fn memory_reduction(&self) -> f32 {
        let fp32_bytes = self.numel() * 4; // 4 bytes per float
        let quant_bytes = self.memory_bytes();
        fp32_bytes as f32 / quant_bytes as f32
    }

    /// Pack INT4 data (2 values per byte)
    /// Only works for INT4 quantization
    pub fn pack(&mut self) -> Result<(), String> {
        if self.params.quant_type != QuantType::INT4 {
            return Err("Can only pack INT4 data".to_string());
        }

        if self.packed {
            return Ok(()); // Already packed
        }

        let mut packed_data = Vec::with_capacity(self.data.len().div_ceil(2));

        for chunk in self.data.chunks(2) {
            let low = chunk[0] & 0x0F; // Keep lower 4 bits
            let high = if chunk.len() > 1 {
                (chunk[1] & 0x0F) << 4 // Shift to upper 4 bits
            } else {
                0
            };

            packed_data.push(low | high);
        }

        self.data = packed_data;
        self.packed = true;

        Ok(())
    }

    /// Unpack INT4 data
    pub fn unpack(&mut self) -> Result<(), String> {
        if self.params.quant_type != QuantType::INT4 {
            return Err("Can only unpack INT4 data".to_string());
        }

        if !self.packed {
            return Ok(()); // Already unpacked
        }

        let numel = self.numel();
        let mut unpacked_data = Vec::with_capacity(numel);

        for &byte in &self.data {
            // Extract lower 4 bits (sign-extend)
            let low = (byte << 4) >> 4;
            unpacked_data.push(low);

            // Extract upper 4 bits (sign-extend)
            if unpacked_data.len() < numel {
                let high = byte >> 4;
                unpacked_data.push(high);
            }
        }

        self.data = unpacked_data;
        self.packed = false;

        Ok(())
    }

    /// Dequantize to f32 tensor
    pub fn dequantize(&self) -> Vec<f32> {
        if self.packed {
            // Need to unpack first
            let mut unpacked = self.clone();
            unpacked.unpack().unwrap();
            return unpacked.dequantize();
        }

        self.data
            .iter()
            .map(|&q| self.params.dequantize(q as i32))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quant_type_properties() {
        assert_eq!(QuantType::INT8.bits(), 8);
        assert_eq!(QuantType::INT4.bits(), 4);
        assert_eq!(QuantType::INT8.reduction_factor(), 4);
        assert_eq!(QuantType::INT4.reduction_factor(), 8);
        assert!(QuantType::INT8.is_integer());
        assert!(QuantType::FP16.is_float());
    }

    #[test]
    fn test_quant_params_symmetric() {
        let params = QuantParams::int8_symmetric(127.0);
        assert_eq!(params.scale, 1.0);
        assert_eq!(params.zero_point, 0);

        // Test quantize/dequantize roundtrip
        let value = 50.0;
        let quantized = params.quantize(value);
        let dequantized = params.dequantize(quantized);
        assert!((dequantized - value).abs() < 1.0); // Allow some error
    }

    #[test]
    fn test_quant_params_from_min_max() {
        let params = QuantParams::from_min_max(-10.0, 10.0, QuantType::INT8, true);
        assert!(params.scale > 0.0);
        assert_eq!(params.zero_point, 0); // Symmetric

        // Test asymmetric
        let params = QuantParams::from_min_max(0.0, 10.0, QuantType::INT8, false);
        assert!(params.scale > 0.0);
        assert_ne!(params.zero_point, 0); // Asymmetric
    }

    #[test]
    fn test_quantized_tensor_creation() {
        let data = vec![10, 20, 30, 40];
        let shape = vec![2, 2];
        let params = QuantParams::int8_symmetric(127.0);

        let tensor = QuantizedTensor::new(data.clone(), shape.clone(), params);

        assert_eq!(tensor.data, data);
        assert_eq!(tensor.shape, shape);
        assert_eq!(tensor.numel(), 4);
        assert!(!tensor.packed);
    }

    #[test]
    fn test_int4_packing() {
        let data = vec![1, 2, 3, 4, 5, 6];
        let shape = vec![6];
        let params = QuantParams::int4_symmetric(7.0);

        let mut tensor = QuantizedTensor::new(data.clone(), shape, params);

        // Pack
        tensor.pack().unwrap();
        assert!(tensor.packed);
        assert_eq!(tensor.memory_bytes(), 3); // 6 values → 3 bytes

        // Unpack
        tensor.unpack().unwrap();
        assert!(!tensor.packed);
        assert_eq!(tensor.data.len(), 6);
    }

    #[test]
    fn test_memory_reduction() {
        let data = vec![0i8; 1000];
        let shape = vec![1000];

        // INT8
        let params_int8 = QuantParams::int8_symmetric(127.0);
        let tensor_int8 = QuantizedTensor::new(data.clone(), shape.clone(), params_int8);
        assert_eq!(tensor_int8.memory_reduction(), 4.0); // FP32 → INT8

        // INT4
        let params_int4 = QuantParams::int4_symmetric(7.0);
        let mut tensor_int4 = QuantizedTensor::new(data.clone(), shape.clone(), params_int4);
        tensor_int4.pack().unwrap();
        assert_eq!(tensor_int4.memory_reduction(), 8.0); // FP32 → INT4 packed
    }

    #[test]
    fn test_dequantize_tensor() {
        let original = vec![1.0, 2.0, 3.0, 4.0];
        let params = QuantParams::int8_symmetric(4.0);

        // Quantize
        let quantized: Vec<i8> = original.iter().map(|&v| params.quantize(v) as i8).collect();

        let tensor = QuantizedTensor::new(quantized, vec![4], params);

        // Dequantize
        let dequantized = tensor.dequantize();

        // Check roundtrip accuracy
        for (i, (&orig, &deq)) in original.iter().zip(dequantized.iter()).enumerate() {
            assert!(
                (orig - deq).abs() < 0.1,
                "Mismatch at {}: {} vs {}",
                i,
                orig,
                deq
            );
        }
    }
}
