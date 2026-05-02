"""
冒烟测试:直接调用 simulation_OTA_two 的核心仿真函数,
绕过 MCP/LLM/BO,看 ngspice + netlist + LUT 链路能不能跑通。
"""
import sys
import os
import logging
import torch

# 把 examples 加入路径(simulation_OTA_two 在那里)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'examples'))

from simulation_OTA_two import OTA_two_simulation_gmid_pro

logging.basicConfig(level=logging.INFO)

# 一组手工选的合理参数
# x = [Cap, k1, k2, L1, L2, L3, L4, L5, R]
x = torch.tensor([[
    2e-12,    # Cap = 2pF
    1.0,      # k1 = 1 (current mirror ratio)
    1.0,      # k2 = 1
    1e-6,     # L1 = 1um
    1e-6,     # L2 = 1um
    1e-6,     # L3 = 1um
    1e-6,     # L4 = 1um
    1e-6,     # L5 = 1um
    3000.0,   # R = 3k
]])

# 5 个 gm/ID 整数值(覆盖 LUT 范围)
gmid1, gmid2, gmid3, gmid4, gmid5 = 15, 12, 10, 12, 15

print("=" * 60)
print("Smoke test: triggering one ngspice simulation")
print("=" * 60)

result = OTA_two_simulation_gmid_pro(x, gmid1, gmid2, gmid3, gmid4, gmid5)

print("\n=== Result ===")
print(f"result.shape: {result.shape}")
print(f"result: {result}")
print("\nFields: [log10(|Av|), log(I_DC), log(PM), log(GBW)]")
