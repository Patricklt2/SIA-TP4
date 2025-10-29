#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

# Create a simple 4x4 grid test
fig, ax = plt.subplots(1, 1, figsize=(8, 8))

# Create fake data matrix
test_matrix = np.random.rand(4, 4)
im = ax.imshow(test_matrix, cmap='viridis')

# Test different positioning approaches
for i in range(4):
    for j in range(4):
        # Method 1: j, i (integer coordinates)
        ax.text(j, i, f"({j},{i})", 
               ha='center', va='center', 
               color='white', fontweight='bold', fontsize=12)

ax.set_title('Testing Text Positioning in imshow Grid')
ax.axis('off')
plt.tight_layout()
plt.savefig('results/position_test.png', dpi=150, bbox_inches='tight')
plt.close()

print("Position test saved to results/position_test.png")
print("Check if text is centered in cells")