// Test: tensor_randn_seeded genera valores reproducibles

print("=== TEST: tensor_randn_seeded ===\n\n")

// Generar tensor con seed 42
print("Primera ejecución con seed=42:\n")
let t1 = tensor_randn_seeded([2, 3], 42)
print("  Tensor: ")
print(t1)
print("\n\n")

// Generar otro tensor con el mismo seed
print("Segunda ejecución con seed=42:\n")
let t2 = tensor_randn_seeded([2, 3], 42)
print("  Tensor: ")
print(t2)
print("\n\n")

// Generar tensor con seed diferente
print("Tercera ejecución con seed=123:\n")
let t3 = tensor_randn_seeded([2, 3], 123)
print("  Tensor: ")
print(t3)
print("\n\n")

print("✅ Si t1 y t2 son idénticos → ÉXITO (reproducible)\n")
print("✅ Si t3 es diferente → ÉXITO (seed diferente genera valores diferentes)\n")
