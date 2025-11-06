// Standard Library - Built-in functions for Charl
// Provides essential functions like print(), len(), range()

use crate::interpreter::Value;

/// Type alias for built-in function signature
pub type BuiltinFn = fn(Vec<Value>) -> Result<Value, String>;

/// print() - Print values to console
/// Usage: print(x), print(x, y, z)
/// Returns: Null
pub fn builtin_print(args: Vec<Value>) -> Result<Value, String> {
    if args.is_empty() {
        println!();
        return Ok(Value::Null);
    }

    for (i, arg) in args.iter().enumerate() {
        if i > 0 {
            print!(" ");
        }
        print!("{}", format_value(arg));
    }
    println!();
    Ok(Value::Null)
}

/// len() - Get length of array/string
/// Usage: len(array) -> int32
/// Returns: Integer length
pub fn builtin_len(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err(format!(
            "len() takes exactly 1 argument, got {}",
            args.len()
        ));
    }

    match &args[0] {
        Value::Array(arr) => Ok(Value::Integer(arr.len() as i64)),
        Value::String(s) => Ok(Value::Integer(s.len() as i64)),
        Value::Tensor { data, .. } => Ok(Value::Integer(data.len() as i64)),
        _ => Err(format!(
            "len() requires array, string, or tensor, got {}",
            args[0].type_name()
        )),
    }
}

/// range() - Create range of integers
/// Usage: range(end) or range(start, end) or range(start, end, step)
/// Returns: Array of integers
pub fn builtin_range(args: Vec<Value>) -> Result<Value, String> {
    let (start, end, step) = match args.len() {
        1 => {
            // range(end) -> 0..end with step 1
            let end = args[0].to_integer()?;
            (0, end, 1)
        }
        2 => {
            // range(start, end) -> start..end with step 1
            let start = args[0].to_integer()?;
            let end = args[1].to_integer()?;
            (start, end, 1)
        }
        3 => {
            // range(start, end, step)
            let start = args[0].to_integer()?;
            let end = args[1].to_integer()?;
            let step = args[2].to_integer()?;
            if step == 0 {
                return Err("range() step cannot be zero".to_string());
            }
            (start, end, step)
        }
        _ => {
            return Err(format!(
                "range() takes 1, 2, or 3 arguments, got {}",
                args.len()
            ));
        }
    };

    // Generate range
    let mut result = Vec::new();
    if step > 0 {
        let mut i = start;
        while i < end {
            result.push(Value::Integer(i));
            i += step;
        }
    } else {
        let mut i = start;
        while i > end {
            result.push(Value::Integer(i));
            i += step;
        }
    }

    Ok(Value::Array(result))
}

/// push() - Add element to end of array
/// Usage: push(array, element) -> null (modifies array in place)
/// Note: In Charl, this is a mutating operation
pub fn builtin_push(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err(format!(
            "push() takes exactly 2 arguments, got {}",
            args.len()
        ));
    }

    // Note: Since we don't have mutable references in the interpreter yet,
    // we return an error for now. This will be properly implemented when
    // we add mutable references or make arrays mutable.
    Err("push() requires mutable array support (not yet implemented)".to_string())
}

/// pop() - Remove and return last element from array
/// Usage: pop(array) -> element (modifies array in place)
/// Note: In Charl, this is a mutating operation
pub fn builtin_pop(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err(format!("pop() takes exactly 1 argument, got {}", args.len()));
    }

    // Note: Same as push(), requires mutable references
    Err("pop() requires mutable array support (not yet implemented)".to_string())
}

/// type() - Get type name of a value as string
/// Usage: type(value) -> string
/// Returns: String representation of the type
pub fn builtin_type(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err(format!(
            "type() takes exactly 1 argument, got {}",
            args.len()
        ));
    }

    Ok(Value::String(args[0].type_name().to_string()))
}

/// str() - Convert value to string
/// Usage: str(value) -> string
/// Returns: String representation of the value
pub fn builtin_str(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err(format!("str() takes exactly 1 argument, got {}", args.len()));
    }

    Ok(Value::String(format_value(&args[0])))
}

/// assert() - Assert that a condition is true
/// Usage: assert(condition) or assert(condition, message)
/// Returns: Null if condition is true, Error otherwise
pub fn builtin_assert(args: Vec<Value>) -> Result<Value, String> {
    if args.is_empty() || args.len() > 2 {
        return Err(format!(
            "assert() takes 1 or 2 arguments, got {}",
            args.len()
        ));
    }

    let condition = &args[0];
    let is_true = match condition {
        Value::Boolean(b) => *b,
        Value::Integer(i) => *i != 0,
        Value::Float(f) => *f != 0.0,
        Value::Null => false,
        _ => true, // Other values are truthy
    };

    if !is_true {
        let message = if args.len() == 2 {
            match &args[1] {
                Value::String(s) => s.clone(),
                v => format_value(v),
            }
        } else {
            "Assertion failed".to_string()
        };
        return Err(format!("❌ {}", message));
    }

    Ok(Value::Null)
}

/// Format a value for display (used by print)
fn format_value(value: &Value) -> String {
    match value {
        Value::Integer(i) => i.to_string(),
        Value::Float(f) => {
            // Format floats nicely
            if f.fract() == 0.0 {
                format!("{:.1}", f)
            } else {
                f.to_string()
            }
        }
        Value::Boolean(b) => b.to_string(),
        Value::String(s) => s.clone(),
        Value::Array(arr) => {
            let elements: Vec<String> = arr.iter().map(|v| format_value(v)).collect();
            format!("[{}]", elements.join(", "))
        }
        Value::Tensor { data, shape } => {
            let elements: Vec<String> = data.iter().map(|v| format_value(v)).collect();
            format!("Tensor({:?}, [{}])", shape, elements.join(", "))
        }
        Value::AutogradTensor(tensor) => {
            format!(
                "AutogradTensor({:?}, shape={:?})",
                tensor.data, tensor.shape
            )
        }
        Value::GPUTensor(gpu_tensor) => {
            format!(
                "GPUTensor(shape={:?}, device={:?})",
                gpu_tensor.tensor.shape, gpu_tensor.device()
            )
        }
        Value::LinearLayer(layer) => {
            format!(
                "Linear({} → {})",
                layer.in_features, layer.out_features
            )
        }
        Value::Conv2dLayer(layer) => {
            format!(
                "Conv2d({} → {}, kernel={}, stride={}, padding={})",
                layer.in_channels, layer.out_channels, layer.kernel_size, layer.stride, layer.padding
            )
        }
        Value::MaxPool2dLayer(layer) => {
            format!(
                "MaxPool2d(kernel={}, stride={})",
                layer.kernel_size, layer.stride
            )
        }
        Value::AvgPool2dLayer(layer) => {
            format!(
                "AvgPool2d(kernel={}, stride={})",
                layer.kernel_size, layer.stride
            )
        }
        Value::Function { .. } => "<function>".to_string(),
        Value::Tuple(elements) => {
            let formatted: Vec<String> = elements.iter().map(|v| format_value(v)).collect();
            format!("({})", formatted.join(", "))
        }
        Value::Null => "null".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_print_single_value() {
        let result = builtin_print(vec![Value::Integer(42)]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Value::Null);
    }

    #[test]
    fn test_print_multiple_values() {
        let result = builtin_print(vec![
            Value::Integer(42),
            Value::String("hello".to_string()),
            Value::Boolean(true),
        ]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_len_array() {
        let arr = Value::Array(vec![
            Value::Integer(1),
            Value::Integer(2),
            Value::Integer(3),
        ]);
        let result = builtin_len(vec![arr]).unwrap();
        assert_eq!(result, Value::Integer(3));
    }

    #[test]
    fn test_len_string() {
        let s = Value::String("hello".to_string());
        let result = builtin_len(vec![s]).unwrap();
        assert_eq!(result, Value::Integer(5));
    }

    #[test]
    fn test_len_error_wrong_args() {
        let result = builtin_len(vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_len_error_wrong_type() {
        let result = builtin_len(vec![Value::Integer(42)]);
        assert!(result.is_err());
    }

    #[test]
    fn test_range_single_arg() {
        let result = builtin_range(vec![Value::Integer(5)]).unwrap();
        assert_eq!(
            result,
            Value::Array(vec![
                Value::Integer(0),
                Value::Integer(1),
                Value::Integer(2),
                Value::Integer(3),
                Value::Integer(4),
            ])
        );
    }

    #[test]
    fn test_range_two_args() {
        let result = builtin_range(vec![Value::Integer(2), Value::Integer(5)]).unwrap();
        assert_eq!(
            result,
            Value::Array(vec![
                Value::Integer(2),
                Value::Integer(3),
                Value::Integer(4),
            ])
        );
    }

    #[test]
    fn test_range_three_args() {
        let result =
            builtin_range(vec![Value::Integer(0), Value::Integer(10), Value::Integer(2)])
                .unwrap();
        assert_eq!(
            result,
            Value::Array(vec![
                Value::Integer(0),
                Value::Integer(2),
                Value::Integer(4),
                Value::Integer(6),
                Value::Integer(8),
            ])
        );
    }

    #[test]
    fn test_range_negative_step() {
        let result =
            builtin_range(vec![Value::Integer(10), Value::Integer(0), Value::Integer(-2)])
                .unwrap();
        assert_eq!(
            result,
            Value::Array(vec![
                Value::Integer(10),
                Value::Integer(8),
                Value::Integer(6),
                Value::Integer(4),
                Value::Integer(2),
            ])
        );
    }

    #[test]
    fn test_range_error_zero_step() {
        let result = builtin_range(vec![Value::Integer(0), Value::Integer(10), Value::Integer(0)]);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("step cannot be zero"));
    }
}
