"""
Linear Probe — 在原始 CLIP 全图嵌入上直接训练线性分类器，作为 DCBM 概念瓶颈的上限。

使用 Step 4 生成的全图嵌入 .torch 文件（train/test）进行逻辑回归，
无需经过概念提取/聚类瓶颈。对应 utils/dcbm.py 第 467 行 linear_probe() 方法。

.torch 文件格式: {filename: [768-dim tensor, class_label_string]}
由 pipeline Step 4 (main_image_emb_to_clip_4.py) 生成。

Usage:
    python linear_probe_all.py --dataset CUB_200_2011
    python linear_probe_all.py --dataset derm7pt
    python linear_probe_all.py --dataset BUS_UCLM
    python linear_probe_all.py --dataset ISIC-2019
    python linear_probe_all.py --dataset places365_part
    python linear_probe_all.py --dataset CUB_200_2011 --model_name "CLIP-ViT-L14"
"""
import os, sys, argparse, datetime, time, numpy as np, torch, warnings
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.exceptions import ConvergenceWarning

# warm_start + max_iter=1 的逐 epoch 训练是预期行为，不需要 sklearn 提示未收敛
warnings.filterwarnings('ignore', category=ConvergenceWarning)


# ══════════════════════════════════════════════════════════════════════
#  数据集 → *_split/ 目录映射 (Step 4 输出位置)
# ══════════════════════════════════════════════════════════════════════

DATASET_CONFIGS = {
    "CUB_200_2011":   "./CUB_200_2011_split/",
    "derm7pt":        "./derm7pt_split/",
    "BUS_UCLM":       "./BUS_UCLM_split/",
    "ISIC-2019":      "./ISIC-2019_split/",
    "places365_part": "./places365_part_split/",
}


# ══════════════════════════════════════════════════════════════════════
#  数据加载
# ══════════════════════════════════════════════════════════════════════

def _to_1d(arr):
    """将嵌入张量统一转为 1-D numpy: [768] 或 [1,768] → [768]"""
    if torch.is_tensor(arr):
        arr = arr.numpy()
    else:
        arr = np.array(arr)
    return arr.squeeze()


def _drop_nan(X, y_str, name="", device="cuda:0"):
    """过滤 X 中包含 NaN 的行, 同步删除对应的 y."""
    nan_mask = np.isnan(X).any(axis=1)
    n_nan = nan_mask.sum()
    if n_nan > 0:
        print(f"  ⚠ {name}: 发现 {n_nan} 条 NaN 嵌入 (可能来自损坏图片), 已跳过")
        X = X[~nan_mask]
        y_str = [l for i, l in enumerate(y_str) if not nan_mask[i]]
    return X, y_str


def load_embeddings(split_dir, dataset, model_name, device="cuda:0"):
    """加载 train / test 的 .torch 嵌入文件, 返回 (X, y_int, label_encoder).

    文件名约定: images_{dataset}_{split}_{model_name}.torch
    由 main_image_emb_to_clip_4.py 生成。
    """
    train_file = os.path.join(split_dir, f"images_{dataset}_train_{model_name}.torch")
    test_file  = os.path.join(split_dir, f"images_{dataset}_test_{model_name}.torch")

    if not os.path.exists(train_file):
        raise FileNotFoundError(
            f"Train embedding 不存在: {train_file}\n"
            f"  请先运行 Step 4: python main_image_emb_to_clip_4.py "
            f"--dataset {dataset} --emb ViT-L/14 --device {device}")
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test embedding 不存在: {test_file}")

    print(f"  加载 train: {train_file}")
    train_dict = torch.load(train_file, weights_only=False, map_location="cpu")
    print(f"  加载 test:  {test_file}")
    test_dict  = torch.load(test_file,  weights_only=False, map_location="cpu")

    # 提取 X (embeddings → numpy) 和 y (label strings)
    # _to_1d 统一处理 [768] 和 [1,768] 两种 shape
    X_train = np.stack([_to_1d(v[0]) for v in train_dict.values()])
    y_train_str = [v[1] for v in train_dict.values()]

    X_test = np.stack([_to_1d(v[0]) for v in test_dict.values()])
    y_test_str = [v[1] for v in test_dict.values()]

    # 过滤 NaN 嵌入 (损坏图片可能导致 CLIP 输出 NaN)
    X_train, y_train_str = _drop_nan(X_train, y_train_str, "train")
    X_test, y_test_str = _drop_nan(X_test, y_test_str, "test")

    # 标签编码 → 整数
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_str)

    # 先过滤 test 中 train 未见过的类别，再 transform (否则会报 KeyError)
    unknown = set(y_test_str) - set(le.classes_)
    if unknown:
        print(f"  ⚠ 测试集包含训练集未出现的类别: {unknown} (样本将被跳过)")
        mask = np.isin(y_test_str, list(le.classes_))
        X_test = X_test[mask]
        y_test_str = [l for l, m in zip(y_test_str, mask) if m]
    y_test = le.transform(y_test_str)

    print(f"  Train: X={X_train.shape}, {len(le.classes_)} classes: {list(le.classes_)}")
    print(f"  Test:  X={X_test.shape}")

    # 安全检查
    assert len(le.classes_) > 1, f"训练集只有 {len(le.classes_)} 个类别!"

    return X_train, X_test, y_train, y_test, le


def load_val_if_exists(split_dir, dataset, model_name, label_encoder):
    """尝试加载 val 嵌入文件（如果存在），返回 (X_val, y_val) 或 (None, None)."""
    val_file = os.path.join(split_dir, f"images_{dataset}_val_{model_name}.torch")
    if not os.path.exists(val_file):
        print(f"  val 文件不存在，跳过: {val_file}")
        return None, None

    print(f"  加载 val:   {val_file}")
    val_dict = torch.load(val_file, weights_only=False, map_location="cpu")
    X_val = np.stack([_to_1d(v[0]) for v in val_dict.values()])
    y_val_str = [v[1] for v in val_dict.values()]

    # 过滤 NaN 嵌入
    X_val, y_val_str = _drop_nan(X_val, y_val_str, "val")

    # 过滤 val 中 train 未见的类别
    known = set(label_encoder.classes_)
    mask = np.array([l in known for l in y_val_str])
    X_val = X_val[mask]
    y_val = label_encoder.transform([l for l, m in zip(y_val_str, mask) if m])
    print(f"  Val:   X={X_val.shape}")
    return X_val, y_val


# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Linear Probe: 直接在 CLIP 全图嵌入上训练逻辑回归 (DCBM 上限)")

    p.add_argument('--dataset', type=str, required=True,
                   choices=list(DATASET_CONFIGS.keys()),
                   help='数据集名称')
    p.add_argument('--epochs', type=int, default=200,
                   help='SGD 训练轮数 (default: 200)')
    p.add_argument('--lr', type=float, default=1e-3,
                   help='SGD 学习率 (default: 1e-3)')
    p.add_argument('--log_interval', type=int, default=0,
                   help='日志打印间隔轮数 (0=自适应)')
    p.add_argument('--model_name', type=str, default='CLIP-ViT-L14',
                   help='嵌入文件名中的模型标识 (default: CLIP-ViT-L14, '
                        '即 ViT-L/14 编码器)')
    p.add_argument('--device', type=str, default='cuda:0',
                   help='GPU 设备名 (与 --use_gpu 配合使用, 默认 cuda:0)')
    p.add_argument('--use_gpu', action='store_true',
                   help='使用 PyTorch GPU 训练 (默认 sklearn CPU; '
                        '365 类场景强烈建议开启, GPU 快 50-100 倍)')
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    split_dir = DATASET_CONFIGS[args.dataset]

    print(f"\n{'='*60}")
    print(f"  Linear Probe — {args.dataset}")
    print(f"  Model: {args.model_name}")
    print(f"  Epochs: {args.epochs},  LR: {args.lr}")
    backend = f"PyTorch GPU ({args.device})" if args.use_gpu else "sklearn CPU"
    print(f"  Backend: {backend}")
    print(f"  Embedding dir: {split_dir}")
    print(f"{'='*60}\n")

    # 1. 加载 train / test 嵌入
    X_train, X_test, y_train, y_test, le = load_embeddings(
        split_dir, args.dataset, args.model_name, args.device)

    # 2. 尝试加载 val
    X_val, y_val = load_val_if_exists(
        split_dir, args.dataset, args.model_name, le)

    # 3. 标准化 (SGD 对特征尺度敏感)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    X_val_s   = scaler.transform(X_val) if X_val is not None else None

    # 4. 训练
    log_interval = args.log_interval
    if log_interval <= 0:
        log_interval = max(1, min(5, args.epochs // 40))

    n_classes = len(le.classes_)
    has_val = X_val_s is not None
    print(f"  {n_classes} 类, {X_train_s.shape[0]} 样本, "
          f"日志间隔 {log_interval} epoch")
    print(f"  {'Epoch':>6s}  {'Train Acc':>10s}  {'Test Acc':>9s}", end="")
    if has_val: print(f"  {'Val Acc':>9s}", end="")
    print(f"  {'Elapsed':>8s}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*9}", end="")
    if has_val: print(f"  {'-'*9}", end="")
    print(f"  {'-'*8}")

    # ── GPU 路径 ──
    if args.use_gpu:
        X_train_t = torch.tensor(X_train_s, dtype=torch.float32).to(args.device)
        y_train_t = torch.tensor(y_train, dtype=torch.long).to(args.device)
        X_test_t  = torch.tensor(X_test_s,  dtype=torch.float32).to(args.device)
        y_test_t  = torch.tensor(y_test,  dtype=torch.long).to(args.device)
        if has_val:
            X_val_t = torch.tensor(X_val_s, dtype=torch.float32).to(args.device)
            y_val_t = torch.tensor(y_val, dtype=torch.long).to(args.device)

        model = torch.nn.Linear(X_train_s.shape[1], n_classes).to(args.device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                     weight_decay=1e-4)
        loader = DataLoader(TensorDataset(X_train_t, y_train_t),
                            batch_size=256, shuffle=True)

        t_start = time.time()
        best_test = 0.0; best_val = 0.0
        for epoch in range(1, args.epochs + 1):
            model.train()
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                criterion(model(batch_X), batch_y).backward()
                optimizer.step()

            if epoch % log_interval == 0 or epoch == 1 or epoch == args.epochs:
                model.eval()
                elapsed = time.time() - t_start
                with torch.no_grad():
                    tr_acc = (model(X_train_t).argmax(1) == y_train_t
                              ).float().mean().item() * 100
                    te_acc = (model(X_test_t).argmax(1) == y_test_t
                              ).float().mean().item() * 100
                sys.stdout.write(f"  {epoch:6d}  {tr_acc:10.2f}%  {te_acc:9.2f}%")
                if te_acc > best_test: best_test = te_acc
                if has_val:
                    with torch.no_grad():
                        va_acc = (model(X_val_t).argmax(1) == y_val_t
                                  ).float().mean().item() * 100
                    sys.stdout.write(f"  {va_acc:9.2f}%")
                    if va_acc > best_val: best_val = va_acc
                sys.stdout.write(f"  {elapsed:7.0f}s\n")
                sys.stdout.flush()

        # 最终评估 — GPU 张量上直接算
        model.eval()
        with torch.no_grad():
            train_acc = (model(X_train_t).argmax(1) == y_train_t
                         ).float().mean().item() * 100
            test_acc  = (model(X_test_t).argmax(1) == y_test_t
                         ).float().mean().item() * 100
            val_acc = None
            if has_val:
                val_acc = (model(X_val_t).argmax(1) == y_val_t
                           ).float().mean().item() * 100
        backend_tag = f"PyTorch GPU ({args.device})"

    # ── CPU 路径 (sklearn SGD, 原版) ──
    else:
        clf = SGDClassifier(loss='log_loss', penalty='l2', alpha=1e-4,
                            max_iter=1, learning_rate='constant', eta0=args.lr,
                            random_state=42, warm_start=True)

        t_start = time.time()
        best_test = 0.0; best_val = 0.0
        for epoch in range(1, args.epochs + 1):
            clf.fit(X_train_s, y_train)

            if epoch % log_interval == 0 or epoch == 1 or epoch == args.epochs:
                elapsed = time.time() - t_start
                tr_acc = accuracy_score(y_train, clf.predict(X_train_s)) * 100
                te_acc = accuracy_score(y_test,  clf.predict(X_test_s))  * 100
                sys.stdout.write(f"  {epoch:6d}  {tr_acc:10.2f}%  {te_acc:9.2f}%")
                if te_acc > best_test: best_test = te_acc
                if has_val:
                    va_acc = accuracy_score(y_val, clf.predict(X_val_s)) * 100
                    sys.stdout.write(f"  {va_acc:9.2f}%")
                    if va_acc > best_val: best_val = va_acc
                sys.stdout.write(f"  {elapsed:7.0f}s\n")
                sys.stdout.flush()

        train_acc = accuracy_score(y_train, clf.predict(X_train_s)) * 100
        test_acc  = accuracy_score(y_test,  clf.predict(X_test_s))  * 100
        val_acc   = accuracy_score(y_val, clf.predict(X_val_s)) * 100 if has_val else None
        backend_tag = "sklearn SGD (CPU)"

    print()

    # 5. 最终评估
    print(f"{'='*60}")
    print(f"  Linear Probe 结果 — {args.dataset}")
    print(f"  Backend: {backend_tag}")
    print(f"  Train Accuracy:  {train_acc:.2f}%")
    print(f"  Test Accuracy:   {test_acc:.2f}%  (best={best_test:.2f}%)")
    if has_val and val_acc is not None:
        print(f"  Val Accuracy:    {val_acc:.2f}%  (best={best_val:.2f}%)")
    print(f"{'='*60}\n")

    # 6. 保存结果
    os.makedirs("./linear_probes", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_path = f"./linear_probes/{args.dataset}_{timestamp}.txt"
    with open(result_path, 'w') as f:
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Model:   {args.model_name}\n")
        f.write(f"Backend: {backend_tag}\n")
        f.write(f"Optimizer: SGD (lr={args.lr}, epochs={args.epochs})\n")
        f.write(f"Train samples: {X_train_s.shape[0]}\n")
        f.write(f"Test samples:  {X_test_s.shape[0]}\n")
        if has_val:
            f.write(f"Val samples:   {X_val_s.shape[0]}\n")
        f.write(f"Classes: {len(le.classes_)}\n")
        f.write(f"Train Accuracy: {train_acc:.2f}%\n")
        f.write(f"Test Accuracy:  {test_acc:.2f}%  (best={best_test:.2f}%)\n")
        if has_val and val_acc is not None:
            f.write(f"Val Accuracy:   {val_acc:.2f}%  (best={best_val:.2f}%)\n")
    print(f"  结果已保存: {result_path}")
