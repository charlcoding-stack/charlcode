// Test cross_entropy_logits con autograd

print("=== TEST: cross_entropy_logits + autograd ===\n\n")

// Dataset simple: 4 examples, 2 classes
let X = tensor([
    0.0, 0.0,  // class 0
    1.0, 0.0,  // class 0
    5.0, 5.0,  // class 1
    6.0, 6.0   // class 1
], [4, 2])

// Targets one-hot
let Y = tensor([
    1.0, 0.0,  // class 0
    1.0, 0.0,  // class 0
    0.0, 1.0,  // class 1
    0.0, 1.0   // class 1
], [4, 2])

// Red simple: 2 → 2
let W = tensor_randn_seeded([2, 2], 42)
W = tensor_requires_grad(W, true)
let b = tensor_zeros([2])
b = tensor_requires_grad(b, true)

print("Entrenando 10 epochs...\n\n")

let lr = 0.1
let optimizer = sgd_create(lr)

let epoch = 0
while epoch < 10 {
    let logits = nn_linear(X, W, b)
    let loss = nn_cross_entropy_logits(logits, Y)

    print("Epoch ")
    print(epoch)
    print(" - Loss: ")
    tensor_print(loss)
    print("\n")

    tensor_backward(loss)

    let params = [W, b]
    let updated = sgd_step(optimizer, params)

    W = updated[0]
    b = updated[1]

    epoch = epoch + 1
}

print("\n=== TEST FINAL ===\n")

let test1 = tensor([0.5, 0.5], [1, 2])
let out1 = nn_linear(test1, W, b)
print("Input [0.5, 0.5] → logits: ")
tensor_print(out1)
print(" → pred: ")
print(argmax(out1))
print(" (esperado: 0)\n")

let test2 = tensor([5.5, 5.5], [1, 2])
let out2 = nn_linear(test2, W, b)
print("Input [5.5, 5.5] → logits: ")
tensor_print(out2)
print(" → pred: ")
print(argmax(out2))
print(" (esperado: 1)\n")
