"""Sample code to test QAIA TSB, USB, LSB algorithms"""
import numpy as np
from scipy.sparse import csr_matrix
from mindquantum.algorithm.qaia import TSB, USB, LSB

# Create a simple Ising problem
np.random.seed(42)
N = 10  # Number of spins

# Create random coupling matrix (symmetric)
J_dense = np.random.randn(N, N)
J_dense = (J_dense + J_dense.T) / 2  # Make symmetric
np.fill_diagonal(J_dense, 0)  # Zero diagonal
J = csr_matrix(J_dense)

# External field
h = np.random.randn(N, 1)

# Initial spin values
x_init = 0.01 * (np.random.rand(N, 1) - 0.5)

print("=" * 60)
print("Testing TSB (Ternary Simulated Bifurcation)")
print("=" * 60)

# Test TSB with different strategies
for strategy in ['linear', 'exponential', 'logarithmic']:
    print(f"\nTSB with {strategy} strategy:")
    tsb = TSB(J, h=h, x=x_init.copy(), n_iter=100, batch_size=1, 
              backend="cpu-float32", dt=1.0, xi=None, strategy=strategy)
    tsb.update()
    cut = tsb.calc_cut()
    energy = tsb.calc_energy()
    print(f"  Cut value: {cut}")
    print(f"  Energy: {energy}")
    print(f"  Final spins shape: {tsb.x.shape}")

print("\n" + "=" * 60)
print("Testing USB (Uniformly Quantized Simulated Bifurcation)")
print("=" * 60)

# Test USB with different thresholds
for threshold in [5, 10, 20]:
    print(f"\nUSB with threshold={threshold}:")
    usb = USB(J, h=h, x=x_init.copy(), n_iter=100, batch_size=1,
              backend="cpu-float32", dt=1.0, xi=None, threshold=threshold)
    usb.update()
    cut = usb.calc_cut()
    energy = usb.calc_energy()
    print(f"  Cut value: {cut}")
    print(f"  Energy: {energy}")

print("\n" + "=" * 60)
print("Testing LSB (Logarithmic Quantized Simulated Bifurcation)")
print("=" * 60)

# Test LSB with different thresholds
for threshold in [3, 5, 8]:
    print(f"\nLSB with threshold={threshold}:")
    lsb = LSB(J, h=h, x=x_init.copy(), n_iter=100, batch_size=1,
              backend="cpu-float32", dt=1.0, xi=None, threshold=threshold)
    lsb.update()
    cut = lsb.calc_cut()
    energy = lsb.calc_energy()
    print(f"  Cut value: {cut}")
    print(f"  Energy: {energy}")

print("\n" + "=" * 60)
print("Testing batch processing")
print("=" * 60)

# Test batch processing
batch_size = 5
x_init_batch = 0.01 * (np.random.rand(N, batch_size) - 0.5)

print(f"\nTSB with batch_size={batch_size}:")
tsb_batch = TSB(J, h=h, x=x_init_batch.copy(), n_iter=100, batch_size=batch_size,
                backend="cpu-float32", strategy="linear")
tsb_batch.update()
cut_batch = tsb_batch.calc_cut()
energy_batch = tsb_batch.calc_energy()
print(f"  Cut values: {cut_batch}")
print(f"  Average cut: {np.mean(cut_batch):.2f}")
print(f"  Energy values: {energy_batch}")
print(f"  Average energy: {np.mean(energy_batch):.2f}")

print("\n" + "=" * 60)
print("Testing without external field (h=None)")
print("=" * 60)

print("\nTSB without h:")
tsb_no_h = TSB(J, h=None, x=x_init.copy(), n_iter=100, batch_size=1,
               backend="cpu-float32", strategy="linear")
tsb_no_h.update()
cut_no_h = tsb_no_h.calc_cut()
print(f"  Cut value: {cut_no_h}")

print("\nUSB without h:")
usb_no_h = USB(J, h=None, x=x_init.copy(), n_iter=100, batch_size=1,
               backend="cpu-float32", threshold=10)
usb_no_h.update()
cut_no_h_usb = usb_no_h.calc_cut()
print(f"  Cut value: {cut_no_h_usb}")

print("\n" + "=" * 60)
print("Testing convergence with different iterations")
print("=" * 60)

for n_iter in [50, 100, 200, 500]:
    tsb_iter = TSB(J, h=h, x=x_init.copy(), n_iter=n_iter, batch_size=1,
                   backend="cpu-float32", strategy="linear")
    tsb_iter.update()
    cut_iter = tsb_iter.calc_cut()
    print(f"  n_iter={n_iter}: Cut = {float(cut_iter):.2f}")

print("\nQAIA tests completed successfully!")