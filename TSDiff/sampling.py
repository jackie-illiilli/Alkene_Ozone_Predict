from easydict import EasyDict
import os, pickle, torch
from tqdm.auto import tqdm
from .models import sampler
from torch_geometric.transforms import Compose
from .models.epsnet import get_model
from .utils.datasets import generate_ts_data2
from .utils.transforms import CountNodesPerGraph
from .utils.misc import seed_all, get_logger
from torch_geometric.data import Batch

def sample(parameter: dict):
    """
    完整的分子采样函数（原脚本的 if __name__ == "__main__" 全部封装）

    参数
    ----
    parameter : dict
        所有原 argparse 参数的默认值已内置，你只用传想改的部分即可。

    返回
    ----
    save_path : str
        最终生成的 samples_all.pkl 完整路径
    """


    # ==================== 默认参数（和原脚本完全一致） ====================
    default = EasyDict({
        # Model
        "ckpt": 'logs/train_config_Kwon_Conf/checkpoints/81000.pt',
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "batch_size": 500,
        "resume": False,                    # 现在是 bool，不是 None

        # IO
        "save_traj": False,
        "save_dir": 'reproduct/Kwon_Chiral2',

        # Data
        "feat_dict": "data/TS/wb97xd3/Kwon_1000_ts1&2/feat_dict.pkl",
        "test_set": 'data/TS/wb97xd3/Kwon_Chiral2/test_data.pkl',
        "start_idx": 0,
        "end_idx": 99999,
        "repeat": 1,

        # TS Guess
        "from_ts_guess": True,
        "denoise_from_time_t": 1500,
        "noise_from_time_t": None,

        # Sampling
        "clip": 1000.0,
        "n_steps": 1500,
        "sampling_type": "ld",
        "eta": 1.0,
        "step_lr": 1e-7,

        "seed": 2022,
    })

    # 合并用户参数（覆盖默认）
    args = EasyDict(default)
    args.update(parameter)

    # 确保 device 格式正确
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA 不可用，自动切换到 CPU")
        args.device = "cpu"

    # ==================== 日志 & 保存目录 ====================
    os.makedirs(args.save_dir, exist_ok=True)
    logger = get_logger("sample", args.save_dir)
    logger.info(f"Sampling arguments: {args}")

    # ==================== 固定随机种子 ====================
    seed_all(args.seed)

    # ==================== 加载模型 ====================
    logger.info("Loading checkpoint...")
    if isinstance(args.ckpt, str):
        ckpt_paths = [args.ckpt]
    else:
        ckpt_paths = args.ckpt  # 支持传入 list 做 ensemble

    models = []
    for ckpt_path in ckpt_paths:
        try:
            ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
            config = ckpt["config"].model
            model = get_model(config).to(args.device)
            model.load_state_dict(ckpt["model"])
            models.append(model)
            logger.info(f"Loaded {ckpt_path}")
        except:
            logger.info(f"load model from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model = get_model(ckpt['hyper_parameters']['model']).to(args.device)
            state_dict = {'.'.join(key.split('.')[1:]):ckpt['state_dict'][key] for key in ckpt['state_dict'].keys()}
            model.load_state_dict(state_dict)
            models.append(model)

    model = sampler.EnsembleSampler(models).to(args.device)

    # ==================== 加载测试集 ====================
    logger.info("Loading test set...")
    transforms = Compose([CountNodesPerGraph()])

    if isinstance(args.test_set, str):
        if args.test_set.endswith(".txt") or args.test_set.endswith((".pkl", ".pck")):
            path = args.test_set
        else:
            path = None
            smarts_list = [args.test_set]
    else:
        # 直接传入 data_list
        test_set = args.test_set
        path = None
        smarts_list = None

    if path is not None:
        if not os.path.isfile(path):
            logger.error(f"Test file not found: {path}")
            raise FileNotFoundError(path)

        if path.endswith(".txt"):
            smarts_list = [line.strip() for line in open(path) if line.strip()]
            test_set = preprocessing(smarts_list, args.feat_dict)
        else:
            test_set = pickle.load(open(path, "rb"))
    elif smarts_list is not None:
        test_set = preprocessing(smarts_list, args.feat_dict)

    # 切片选择
    test_set = test_set[args.start_idx : args.end_idx]

    # ==================== Resume 支持 ====================
    results = []
    done_smiles = set()

    resume_path = os.path.join(args.save_dir, "samples_all.pkl")
    if args.resume and os.path.exists(resume_path):
        logger.info(f"Resuming from {resume_path}")
        with open(resume_path, "rb") as f:
            results = pickle.load(f)
        done_smiles = {data.smiles for data in results}
        logger.info(f"Already done {len(done_smiles)} molecules")

    # 过滤已完成
    test_set = [d for d in test_set if d.smiles not in done_smiles]

    if len(test_set) == 0:
        logger.info("All molecules already sampled!")
        return resume_path

    # ==================== 采样主循环 ====================
    logger.info(f"Start sampling {len(test_set)} molecules...")

    for batch_data in tqdm(batching(test_set, args.batch_size, args.repeat), total=len(test_set)//args.batch_size + 1):
        batch = Batch.from_data_list(batch_data).to(args.device)

        for retry in range(3):  # 最多重试 3 次
            try:
                if args.from_ts_guess:
                    if hasattr(batch, "ts_guess") and batch.ts_guess is not None:
                        init_guess = batch.ts_guess
                    else:
                        init_guess = batch.pos

                    start_t = args.noise_from_time_t or args.denoise_from_time_t
                    sqrt_a = model.alphas[start_t - 1].sqrt() if start_t > 0 else torch.tensor(1.0)
                    pos_init = (init_guess / sqrt_a).to(args.device)
                else:
                    pos_init = torch.randn(batch.num_nodes, 3, device=args.device)

                pos_gen, pos_traj = model.dynamic_sampling(
                    atom_type=batch.atom_type,
                    r_feat=batch.r_feat,
                    p_feat=batch.p_feat,
                    pos_init=pos_init,
                    bond_index=batch.edge_index,
                    bond_type=batch.edge_type,
                    batch=batch.batch,
                    num_graphs=batch.num_graphs,
                    extend_order=True,
                    n_steps=args.n_steps,
                    step_lr=args.step_lr,
                    clip=args.clip if retry == 0 else 20.0,   # 失败后自动加强 clip
                    sampling_type=args.sampling_type,
                    eta=args.eta,
                    noise_from_time_t=args.noise_from_time_t,
                    denoise_from_time_t=args.denoise_from_time_t,
                )

                # 轨迹后处理
                alphas = model.alphas.detach().cpu()
                if args.denoise_from_time_t is not None:
                    alphas = alphas[args.denoise_from_time_t - args.n_steps : args.denoise_from_time_t]
                else:
                    alphas = alphas[-args.n_steps:]

                alphas = alphas.flip(0).view(-1, 1, 1).sqrt()
                pos_traj_scaled = torch.stack(pos_traj).cpu() * alphas

                # 保存结果
                for j, data in enumerate(batch.to_data_list()):
                    mask = batch.batch == j
                    if args.save_traj:
                        data.pos_gen = pos_traj_scaled[:, mask]
                    else:
                        data.pos_gen = pos_gen[mask].cpu()

                    data = data.cpu()
                    results.append(data)

                # 中间保存
                temp_path = os.path.join(args.save_dir, "samples_not_all.pkl")
                with open(temp_path, "wb") as f:
                    pickle.dump(results, f)

                break  # 成功了，跳出重试

            except FloatingPointError as e:
                logger.warning(f"FloatingPointError, retry {retry+1}/3 with stronger clip...")
                if retry == 2:
                    raise e

    # ==================== 最终保存 ====================
    final_path = os.path.join(args.save_dir, "samples_all.pkl")
    # 删除中间文件
    temp_path = os.path.join(args.save_dir, "samples_not_all.pkl")
    if os.path.exists(temp_path):
        os.remove(temp_path)

    with open(final_path, "wb") as f:
        pickle.dump(results, f)

    logger.info(f"Sampling completed! Total {len(results)} conformations saved to:")
    logger.info(f"    {final_path}")

    return final_path


# ==================== 辅助函数（保持不变） ====================
def preprocessing(smarts_list, feat_dict_path="feat_dict.pkl"):
    import pickle, torch
    feat_dict = pickle.load(open(feat_dict_path, "rb"))
    from utils.datasets import generate_ts_data2

    data_list = []
    for smarts in smarts_list:
        r, p = smarts.split(">>")
        data, _ = generate_ts_data2(r, p, None, feat_dict=feat_dict)
        data_list.append(data)

    num_cls = [len(v) for v in feat_dict.values()]
    for data in data_list:
        def onehot_feat(feat_tensor):
            feats = feat_tensor.T
            onehot = []
            for feat, n in zip(feats, num_cls):
                onehot.append(torch.nn.functional.one_hot(feat.long(), num_classes=n))
            return torch.cat(onehot, dim=-1)

        data.r_feat = onehot_feat(data.r_feat)
        data.p_feat = onehot_feat(data.p_feat)

    return data_list


def batching(iterable, batch_size, repeat_num=1):
    iterable = [item.clone() for item in iterable for _ in range(repeat_num)]
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]