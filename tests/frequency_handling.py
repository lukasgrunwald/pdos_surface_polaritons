import numpy as np
from pdos_surf import io_manager as pdf

# Create frequency array
freq_array = pdf.create_freq_array(w_top=250e12, Nw=10001)
split_freqs = pdf.partition_freq_array(freq_array, n_subdivisions=200)
merged = pdf.merge_freq_array(split_freqs)

# Split into 4 subdivisions
print(f"Full array length: {len(freq_array)}")
print(f"Subdivision lengths: {[len(arr) for arr in split_freqs]}")
print(f"Merged length: {len(merged)}")

print(np.array_equal(freq_array, merged))
print(len(freq_array) == sum(len(arr) for arr in split_freqs))
