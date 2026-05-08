"""
快速检查 Stage 1 / Stage 2 的 feasible 结果。

用法:
  python check_feasible.py                        # 自动找最新的 Stage 1 输出
  python check_feasible.py --task <task_id>       # 指定 task_id
  python check_feasible.py --stage 2              # 看 Stage 2 (FocalOpt) 输出
  python check_feasible.py --top 10               # 显示 top 10 而不是默认 5
  python check_feasible.py --plot                 # 画收敛曲线 (需要 matplotlib)

约束(对齐 config.py 的 CONSTRAINTS):
  Gain   > 60 dB
  DC Current * 1.8 < 1e-3 A
  PM     > 60 deg
  GBW    > 4 MHz
"""

import argparse
import glob
import os
import sys
import pandas as pd

# 跟 config.py 对齐
CONSTRAINTS = {
    "gain": 60,
    "current_limit": 1e-3,
    "current_multiplier": 1.8,
    "phase": 60,
    "gbw": 4e6,
}


def find_latest_csv(stage: int, task_id: str = None) -> str:
    """找到最新的 y.csv (Stage 1) 或 .csv (Stage 2)。"""
    if stage == 1:
        if task_id:
            pattern = f"./store/design_{task_id}_SEED_*_y.csv"
        else:
            pattern = "./store/design_*_y.csv"
    else:  # stage 2
        if task_id:
            pattern = f"./store/focalopt_{task_id}_SEED_*.csv"
        else:
            pattern = "./store/focalopt_*.csv"

    files = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not files:
        sys.exit(f"找不到匹配的 CSV: {pattern}")
    return files[-1]


def load_data(filepath: str, stage: int) -> pd.DataFrame:
    """加载并标准化列名。"""
    df = pd.read_csv(filepath)
    df.columns = [c.strip() for c in df.columns]

    # Stage 2 列名是 gain(db), dc_current, phase, GBW(MHZ)
    # Stage 1 列名是 gain, dc_current, phase, GBW
    rename_map = {
        'gain(db)': 'gain',
        'GBW(MHZ)': 'GBW_MHZ',  # Stage 2 单位是 MHz
    }
    df = df.rename(columns=rename_map)

    # 如果是 Stage 2,GBW 单位是 MHz,转回 Hz 方便统一比较
    if 'GBW_MHZ' in df.columns:
        df['GBW'] = df['GBW_MHZ'] * 1e6

    return df


def filter_feasible(df: pd.DataFrame) -> pd.DataFrame:
    """按约束筛选 feasible 点。"""
    c = CONSTRAINTS
    mask = (
        (df['gain'] > c['gain']) &
        (df['dc_current'] * c['current_multiplier'] < c['current_limit']) &
        (df['phase'] > c['phase']) &
        (df['GBW'] > c['gbw'])
    )
    return df[mask].copy()


def print_summary(df: pd.DataFrame, fe: pd.DataFrame, top: int):
    print(f"\n{'='*70}")
    print(f"总仿真次数: {len(df)}")
    print(f"Feasible 点数: {len(fe)}  ({len(fe)/len(df)*100:.1f}%)")

    if len(fe) == 0:
        print("\n没有找到 feasible 点。各约束的通过率:")
        c = CONSTRAINTS
        for name, cond in [
            ('Gain > 60 dB', df['gain'] > c['gain']),
            ('Current*1.8 < 1mA', df['dc_current'] * c['current_multiplier'] < c['current_limit']),
            ('PM > 60°', df['phase'] > c['phase']),
            ('GBW > 4 MHz', df['GBW'] > c['gbw']),
        ]:
            print(f"  {name:30s}: {cond.sum():4d}/{len(df)} ({cond.mean()*100:.1f}%)")
        return

    print(f"\nFeasible 中 dc_current 最低的 {min(top, len(fe))} 个点:\n")
    best = fe.nsmallest(top, 'dc_current').copy()

    # 友好显示:Current 用 μA,GBW 用 MHz
    best['Idd_uA'] = best['dc_current'] * 1e6
    best['GBW_MHz'] = best['GBW'] / 1e6
    display_cols = ['iter_times', 'gain', 'Idd_uA', 'phase', 'GBW_MHz']
    display_cols = [c for c in display_cols if c in best.columns]
    print(best[display_cols].to_string(index=False, float_format=lambda x: f'{x:.2f}'))

    print(f"\n最优点:")
    b = best.iloc[0]
    print(f"  Gain     = {b['gain']:.2f} dB    (margin: +{b['gain']-60:.2f} dB)")
    print(f"  Idd      = {b['Idd_uA']:.2f} μA   (margin: -{(1e-3/1.8 - b['dc_current'])*1e6:.2f} μA)")
    print(f"  PM       = {b['phase']:.2f}°     (margin: +{b['phase']-60:.2f}°)")
    print(f"  GBW      = {b['GBW_MHz']:.2f} MHz  (margin: +{b['GBW_MHz']-4:.2f} MHz)")
    print(f"  iter     = {int(b['iter_times'])}")


def plot_convergence(df: pd.DataFrame, fe: pd.DataFrame):
    """画 dc_current 随迭代的收敛曲线。"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[跳过画图] 需要安装 matplotlib: pip install matplotlib")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(df['iter_times'], df['dc_current']*1e6, s=10, alpha=0.3, c='gray', label='all sims')
    if len(fe):
        ax.scatter(fe['iter_times'], fe['dc_current']*1e6, s=20, c='green', label='feasible')
        # best so far
        fe_sorted = fe.sort_values('iter_times').copy()
        fe_sorted['best_so_far'] = fe_sorted['dc_current'].cummin() * 1e6
        ax.plot(fe_sorted['iter_times'], fe_sorted['best_so_far'], 'r-', label='best feasible so far')
    ax.set_xlabel('iteration')
    ax.set_ylabel('DC Current (μA)')
    ax.legend()
    ax.set_title('Stage 1 BO Convergence')
    ax.grid(alpha=0.3)
    out = './bo_convergence.png'
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    print(f"\n收敛曲线已保存: {out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=int, default=1, choices=[1, 2])
    parser.add_argument('--task', type=str, default=None, help='指定 task_id')
    parser.add_argument('--top', type=int, default=5)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()

    csv_path = find_latest_csv(args.stage, args.task)
    print(f"读取: {csv_path}")

    df = load_data(csv_path, args.stage)
    fe = filter_feasible(df)

    print_summary(df, fe, args.top)

    if args.plot:
        plot_convergence(df, fe)