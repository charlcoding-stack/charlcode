# Charl Language Specification v1.0

## 1. Introduction

Charl is a statically-typed programming language designed specifically for Artificial Intelligence and Deep Learning applications. This document defines the syntax and semantics of Charl v1.0.

## 2. Lexical Structure

### 2.1 Comments

```charl
// Single-line comment

/*
   Multi-line comment
   Can span multiple lines
*/
```

### 2.2 Identifiers

Identifiers start with a letter or underscore, followed by letters, digits, or underscores.

```charl
valid_identifier
_private
modelName
var123
```

### 2.3 Keywords

Reserved keywords in Charl:

```
let         fn          return      if          else
for         while       break       continue    true
false       and         or          not         model
layer       layers      autograd    gradient    dense
conv2d      activation  dropout     relu        sigmoid
tanh        softmax     int32       int64       float32
float64     bool        tensor
```

### 2.4 Literals

#### Integer Literals
```charl
42
1000
-15
```

#### Float Literals
```charl
3.14
2.718
-0.5
1.0e-5
```

#### Boolean Literals
```charl
true
false
```

#### String Literals
```charl
"hello world"
"neural network"
```

#### Array Literals
```charl
[1, 2, 3, 4, 5]
[1.0, 2.5, 3.7]
[[1, 2], [3, 4]]  // 2D array
```

## 3. Type System

### 3.1 Primitive Types

```charl
int32      // 32-bit signed integer
int64      // 64-bit signed integer
float32    // 32-bit floating point
float64    // 64-bit floating point
bool       // Boolean: true or false
```

### 3.2 Tensor Type

The tensor type is a first-class citizen in Charl with compile-time shape checking.

#### Syntax
```charl
tensor<DataType, [Dimensions]>
```

#### Examples
```charl
// 1D tensor (vector) of 5 float32 elements
tensor<float32, [5]> vec

// 2D tensor (matrix) of 3x4 float64 elements
tensor<float64, [3, 4]> matrix

// 3D tensor for image data: batch_size=32, height=28, width=28
tensor<float32, [32, 28, 28]> images

// 4D tensor for convolutional layers: batch=16, channels=3, height=224, width=224
tensor<float32, [16, 3, 224, 224]> conv_input
```

### 3.3 Type Inference

Charl supports type inference in many contexts:

```charl
let x = 42              // Inferred as int32
let y = 3.14            // Inferred as float64
let z = true            // Inferred as bool
let arr = [1, 2, 3]     // Inferred as tensor<int32, [3]>
```

## 4. Variables and Constants

### 4.1 Variable Declaration

```charl
// With type annotation
let x: int32 = 10

// With type inference
let y = 20

// Tensors
let weights: tensor<float32, [10, 5]> = [[...]]
```

### 4.2 Mutability

By default, variables are immutable. Use `mut` for mutable variables (future feature):

```charl
// Future syntax
let mut counter = 0
counter = counter + 1
```

## 5. Operators

### 5.1 Arithmetic Operators

```charl
+    // Addition
-    // Subtraction
*    // Multiplication (element-wise for tensors)
/    // Division
%    // Modulo
@    // Matrix multiplication (for tensors)
```

### 5.2 Comparison Operators

```charl
==   // Equal
!=   // Not equal
<    // Less than
<=   // Less than or equal
>    // Greater than
>=   // Greater than or equal
```

### 5.3 Logical Operators

```charl
and  // Logical AND
or   // Logical OR
not  // Logical NOT
```

### 5.4 Tensor Operations

```charl
// Element-wise operations
let a: tensor<float32, [2, 3]> = [[1, 2, 3], [4, 5, 6]]
let b: tensor<float32, [2, 3]> = [[1, 1, 1], [2, 2, 2]]

let c = a + b    // Element-wise addition
let d = a * b    // Element-wise multiplication

// Matrix multiplication
let m1: tensor<float32, [2, 3]> = [[1, 2, 3], [4, 5, 6]]
let m2: tensor<float32, [3, 2]> = [[1, 2], [3, 4], [5, 6]]
let result = m1 @ m2  // Shape: [2, 2]
```

## 6. Control Flow

### 6.1 If-Else Statements

```charl
if x > 0 {
    // positive
} else if x < 0 {
    // negative
} else {
    // zero
}
```

### 6.2 While Loops

```charl
let mut i = 0
while i < 10 {
    i = i + 1
}
```

### 6.3 For Loops

```charl
for i in 0..10 {
    // Loop body
}

for element in array {
    // Iterate over array
}
```

## 7. Functions

### 7.1 Function Declaration

```charl
fn add(x: int32, y: int32) -> int32 {
    return x + y
}

// Type inference for return type
fn multiply(x: float32, y: float32) {
    return x * y  // Inferred as float32
}
```

### 7.2 Tensor Functions

```charl
fn dot_product(a: tensor<float32, [N]>, b: tensor<float32, [N]>) -> float32 {
    return sum(a * b)
}

fn matrix_multiply(a: tensor<float32, [M, K]>,
                   b: tensor<float32, [K, N]>) -> tensor<float32, [M, N]> {
    return a @ b
}
```

## 8. Automatic Differentiation

### 8.1 Autograd Syntax

Charl provides native automatic differentiation:

```charl
// Simple function with autograd
fn loss_function(x: tensor<float32, [10]>) -> float32 {
    return sum(x * x)
}

let x: tensor<float32, [10]> = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
let grad = autograd(loss_function, x)
```

### 8.2 Gradient Type

```charl
// The gradient has the same shape as the input
gradient<tensor<float32, [10]>>
```

## 9. Model Definition DSL

### 9.1 Model Declaration

Charl provides a declarative DSL for defining neural network models:

```charl
model MnistClassifier {
    layers {
        dense(784, 128, activation: relu)
        dropout(0.2)
        dense(128, 64, activation: relu)
        dropout(0.2)
        dense(64, 10, activation: softmax)
    }
}
```

### 9.2 Layer Types

#### Dense Layer
```charl
dense(input_size, output_size, activation: function)
```

#### Convolutional Layer
```charl
conv2d(in_channels, out_channels, kernel_size, stride, padding)
```

#### Dropout Layer
```charl
dropout(probability)
```

### 9.3 Activation Functions

Built-in activation functions:

```charl
relu       // Rectified Linear Unit
sigmoid    // Sigmoid function
tanh       // Hyperbolic tangent
softmax    // Softmax function
```

### 9.4 Using Models

```charl
// Create model instance
let model = MnistClassifier()

// Forward pass
let input: tensor<float32, [1, 784]> = load_image("digit.png")
let output = model.forward(input)

// Training with autograd
let loss = cross_entropy(output, target)
let gradients = autograd(loss, model.parameters())
```

## 10. Standard Library Functions

### 10.1 Tensor Operations

```charl
// Shape manipulation
reshape(tensor, new_shape)
transpose(tensor)
flatten(tensor)

// Aggregation
sum(tensor)              // Sum all elements
mean(tensor)             // Average all elements
max(tensor)              // Maximum element
min(tensor)              // Minimum element

// Math operations
exp(tensor)              // Exponential
log(tensor)              // Natural logarithm
sqrt(tensor)             // Square root
pow(tensor, exponent)    // Power

// Random
random_normal(shape)     // Normal distribution
random_uniform(shape)    // Uniform distribution
```

### 10.2 Loss Functions

```charl
mse_loss(predictions, targets)           // Mean Squared Error
cross_entropy(predictions, targets)      // Cross Entropy Loss
binary_cross_entropy(predictions, targets)
```

### 10.3 Optimizers (Future)

```charl
sgd(parameters, learning_rate)
adam(parameters, learning_rate, beta1, beta2)
```

## 11. Examples

### 11.1 Simple Linear Regression

```charl
fn linear_model(x: tensor<float32, [N]>,
                w: float32,
                b: float32) -> tensor<float32, [N]> {
    return x * w + b
}

fn mse(predictions: tensor<float32, [N]>,
       targets: tensor<float32, [N]>) -> float32 {
    let diff = predictions - targets
    return mean(diff * diff)
}

// Training
let x: tensor<float32, [100]> = load_data("x.csv")
let y: tensor<float32, [100]> = load_data("y.csv")

let w: float32 = 0.0
let b: float32 = 0.0

for epoch in 0..1000 {
    let predictions = linear_model(x, w, b)
    let loss = mse(predictions, y)

    let gradients = autograd(loss, [w, b])
    w = w - 0.01 * gradients[0]
    b = b - 0.01 * gradients[1]
}
```

### 11.2 Neural Network

```charl
model SimpleNet {
    layers {
        dense(10, 64, activation: relu)
        dense(64, 32, activation: relu)
        dense(32, 1, activation: sigmoid)
    }
}

fn train(model: SimpleNet,
         data: tensor<float32, [1000, 10]>,
         labels: tensor<float32, [1000, 1]>) {

    for epoch in 0..100 {
        let predictions = model.forward(data)
        let loss = binary_cross_entropy(predictions, labels)

        let gradients = autograd(loss, model.parameters())
        model.update(gradients, learning_rate: 0.001)

        if epoch % 10 == 0 {
            print("Epoch: ", epoch, " Loss: ", loss)
        }
    }
}
```

## 12. File Extensions

Charl source files use the following extensions:

- `.charl` - Full extension
- `.ch` - Short extension

## 13. Future Features (Post v1.0)

- Generics
- Traits/Interfaces
- Pattern matching
- Async/await for distributed training
- Module system and package manager
- GPU kernel customization
- Quantization annotations
- Model compression directives

---

**Version:** 1.0
**Last Updated:** 2025-11-04
**Status:** Draft - Under Development
