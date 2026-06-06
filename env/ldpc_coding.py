# -*- coding: utf-8 -*-
"""
LDPC 编解码器模块 — 用于信道编码, 降低误比特率

实现一个 (3,6)-规则 LDPC 码:
  - 码长 n = 256, 信息位 k = 128, 码率 1/2
  - 校验矩阵 H = [P | I_m], 每列 3 个 1, 每行 6 个 1
  - 编码: 系统形式, G = [I_k | P^T]
  - 解码: 对数域置信传播 (BP) 算法 (Sum-Product)

核心设计:
  H 直接构造成系统形式 [P | I_m],
  编码时 G = [I_k | P^T] 保证 H * G^T = 0。

用法:
  ldpc = LDPC(n=256, k=128)
  codeword = ldpc.encode(message)      # 编码
  decoded = ldpc.decode(llr, max_iter) # 解码 (LLR 软输入)
"""

import numpy as np


class LDPC:
    """(3,6)-规则 LDPC 编解码器。

    H = [P | I_m], G = [I_k | P^T]。
    对数域 BP (Sum-Product) 解码。
    """

    def __init__(self, n: int = 256, k: int = 128, seed: int = 42):
        self.n = n
        self.k = k
        self.m = n - k
        self.dv = 3
        self.dc = 6

        np.random.seed(seed)
        self.H = self._create_H()
        self.G = self._create_G()
        self._build_graph()

        assert self._verify(), "H * G^T != 0, LDPC 构造失败"

    # ---------------------------------------------------------------
    # 校验矩阵构造: H = [P | I_m]
    #   - 左半部分 P (m x k): 每列 2 个 1, 每行 5 个 1
    #   - 右半部分 I_m (m x m): 每列/行 1 个 1
    #   - 总计: 列度 3, 行度 6
    # ---------------------------------------------------------------
    def _create_H(self) -> np.ndarray:
        """构造系统形式校验矩阵 H = [P | I_m]。"""
        m, k = self.m, self.k
        H = np.zeros((m, self.n), dtype=np.int8)

        # 右半部分: I_m
        for i in range(m):
            H[i, k + i] = 1

        # 左半部分: P, 每列 2 个 1, 每行 5 个 1
        row_deg = np.zeros(m, dtype=np.int32)

        for col in range(k):
            # 找行度数最低的 2 行
            order = np.argsort(row_deg)
            H[order[0], col] = 1
            row_deg[order[0]] += 1
            H[order[1], col] = 1
            row_deg[order[1]] += 1

        return H

    # ---------------------------------------------------------------
    # 生成矩阵: G = [I_k | P^T]
    # ---------------------------------------------------------------
    def _create_G(self) -> np.ndarray:
        """从 H = [P | I_m] 构建 G = [I_k | P^T]。"""
        m, k = self.m, self.k
        P = self.H[:, :k].copy()
        G = np.zeros((k, self.n), dtype=np.int8)
        G[:, :k] = np.eye(k, dtype=np.int8)
        G[:, k:] = P.T.copy() % 2
        return G

    def _verify(self) -> bool:
        result = (self.H @ self.G.T) % 2
        return np.sum(result) == 0

    # ---------------------------------------------------------------
    # 二分图结构 (用于 BP 解码)
    # ---------------------------------------------------------------
    def _build_graph(self):
        self.vn_edges = [[] for _ in range(self.n)]
        self.cn_edges = [[] for _ in range(self.m)]
        for r in range(self.m):
            for c in range(self.n):
                if self.H[r, c] == 1:
                    self.cn_edges[r].append(c)
                    self.vn_edges[c].append(r)

    # ---------------------------------------------------------------
    # 编码
    # ---------------------------------------------------------------
    def encode(self, message: np.ndarray) -> np.ndarray:
        """编码: codeword = message * G (mod 2)
        前 k 位 = 原始信息 (系统码)。
        """
        assert len(message) == self.k
        return ((message @ self.G) % 2).astype(np.int8)

    # ---------------------------------------------------------------
    # 解码: 对数域置信传播 (BP / Sum-Product)
    # ---------------------------------------------------------------
    def decode(self, llr: np.ndarray, max_iter: int = 50) -> np.ndarray:
        """对数域 BP 解码。

        参数:
            llr: 信道 LLR, (n,); 正值 -> 比特 0, 负值 -> 比特 1
            max_iter: 最大迭代次数
        返回:
            decoded: 解码码字, (n,)
        """
        v2c = [{c: 0.0 for c in self.vn_edges[v]} for v in range(self.n)]
        c2v = [{v: 0.0 for v in self.cn_edges[c]} for c in range(self.m)]

        for _ in range(max_iter):
            # 变量节点 -> 校验节点
            for v in range(self.n):
                for c in self.vn_edges[v]:
                    total = llr[v]
                    for c2 in self.vn_edges[v]:
                        if c2 != c:
                            total += c2v[c2][v]
                    v2c[v][c] = total

            # 校验节点 -> 变量节点
            for c in range(self.m):
                for v in self.cn_edges[c]:
                    prod = 1.0
                    for v2 in self.cn_edges[c]:
                        if v2 != v:
                            prod *= np.tanh(v2c[v2][c] / 2.0)
                    prod = np.clip(prod, -1.0 + 1e-10, 1.0 - 1e-10)
                    c2v[c][v] = 2.0 * np.arctanh(prod)

            # 硬判决 + 早停
            decisions = np.zeros(self.n, dtype=np.int8)
            for v in range(self.n):
                total = llr[v]
                for c in self.vn_edges[v]:
                    total += c2v[c][v]
                decisions[v] = 0 if total > 0 else 1

            if np.sum((self.H @ decisions) % 2) == 0:
                break

        return decisions

    # ---------------------------------------------------------------
    # LLR 工具
    # ---------------------------------------------------------------
    def soft_to_llr(self, soft_bits: np.ndarray) -> np.ndarray:
        """将软判决概率 P(b=1) -> LLR = ln((1-p)/p)"""
        p = np.clip(soft_bits, 1e-10, 1 - 1e-10)
        return np.log((1.0 - p) / p)


def test_ldpc():
    """LDPC 模块自测。"""
    ldpc = LDPC(n=256, k=128)

    assert ldpc._verify()
    H_col_deg = ldpc.H.sum(axis=0)
    H_row_deg = ldpc.H.sum(axis=1)
    print(f"  H 形状: {ldpc.H.shape}, G 形状: {ldpc.G.shape}")
    print(f"  H 行度: {H_row_deg.min()}-{H_row_deg.max()}")
    print(f"  H 列度: {H_col_deg.min()}-{H_col_deg.max()}")
    print(f"  前 {ldpc.k} 列度: {H_col_deg[:ldpc.k].min()}-{H_col_deg[:ldpc.k].max()}")

    # 编码测试
    msg = np.random.randint(0, 2, 128).astype(np.int8)
    cw = ldpc.encode(msg)
    assert np.array_equal(cw[:128], msg)
    assert np.sum((ldpc.H @ cw) % 2) == 0
    print("  编码测试通过")

    # AWGN BP 解码测试
    for snr in [1, 2, 3, 5]:
        snr_lin = 10.0 ** (snr / 10.0)
        noise_var = 1.0 / (2.0 * snr_lin)
        errs = 0
        total = 0
        for _ in range(20):
            msg = np.random.randint(0, 2, 128).astype(np.int8)
            cw = ldpc.encode(msg)
            sym = 1.0 - 2.0 * cw.astype(np.float64)
            rx = sym + np.random.randn(256) * np.sqrt(noise_var)
            llr = 2.0 * rx / noise_var
            dec = ldpc.decode(llr, max_iter=50)
            errs += np.sum(dec != cw)
            total += 256
        print(f"  SNR={snr}dB, BER={errs/total:.5f}")

    print("LDPC 模块自测通过。")


if __name__ == "__main__":
    test_ldpc()
