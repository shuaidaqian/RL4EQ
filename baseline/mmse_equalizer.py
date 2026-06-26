# -*- coding: utf-8 -*-
"""MMSE 时域均衡器 (基于训练序列)"""
import torch

class MMSEEqualizer:
    def __init__(self, num_taps=16, filter_len=31):
        self.num_taps = num_taps
        self.filter_len = filter_len if filter_len%2==1 else filter_len+1
        self.half = self.filter_len // 2

    def estimate_channel(self, tx, rx):
        """时域互相关信道估计"""
        h = torch.zeros(self.num_taps, dtype=torch.complex64)
        rxc = torch.complex(rx[:,0], rx[:,1])
        for i in range(self.num_taps):
            h[i] = (rxc[i:] * tx[:len(tx)-i].conj()).mean() if i < len(tx) else 0
        return h * (len(tx) / (tx.abs()**2).sum())

    def design_mmse_filter(self, h, snr_lin):
        """设计 MMSE 均衡滤波器 (时域)
        最小化 E[|w^T y - x|^2], 解: w = (R_yy)^{-1} r_yx
        其中 R_yy = H H^H + sigma^2 I, r_yx = h_0 (第0列)
        """
        L = self.filter_len; half = self.half
        # 构造信道卷积矩阵 H (L x (L+num_taps-1))
        # 简化: 用 Toeplitz 构造
        H = torch.zeros(L, L + self.num_taps - 1, dtype=torch.complex64)
        for i in range(L):
            for j in range(self.num_taps):
                if i+j < H.shape[1]:
                    H[i, i+j] = h[j]
        # R_yy = H H^H + sigma^2 I, 取中间delay
        Ryy = H @ H.T.conj() + (1/snr_lin) * torch.eye(L, dtype=torch.complex64)
        # r_yx = H * e_d (第delay列)
        delay = half
        e_d = torch.zeros(L + self.num_taps - 1, dtype=torch.complex64)
        if delay < len(e_d): e_d[delay] = 1.0
        ryx = H @ e_d
        # w = Ryy^{-1} ryx
        w = torch.linalg.solve(Ryy, ryx)
        return w

    def __call__(self, rx, tx_tr, rx_tr, snr_db):
        L = rx.shape[0]; snr_lin = 10**(snr_db/10)
        tx_sym = (1 - 2*tx_tr[:len(tx_tr)])
        rx_tr_c = torch.complex(rx_tr[:,0], rx_tr[:,1])
        h = self.estimate_channel(tx_sym, rx_tr)
        # 简化: 直接用频域MMSE (时域太慢)
        N = 512; hp = torch.zeros(N, dtype=torch.complex64)
        hp[:self.num_taps] = h
        H = torch.fft.fft(hp); Hp = H.abs()**2
        W = H.conj() / (Hp + 1/snr_lin + 1e-10)
        rxc = torch.complex(rx[:,0], rx[:,1])
        Se = W * torch.fft.fft(rxc, n=N)
        return torch.fft.ifft(Se).real[:L], h

def test():
    import sys,os;sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from env.rayleigh_channel import RayleighMultipathChannel
    from env.frame_structure import FrameConfig,FrameGenerator
    torch.manual_seed(42)
    cfg=FrameConfig();gen=FrameGenerator(cfg)
    chan=RayleighMultipathChannel(num_taps=16,delay_spread=10,snr_db=20)
    bits=gen.generate();tx=gen.modulate(bits);rx=chan(tx)
    tx_bits=bits[:128];tx_tr=(1-2*tx_bits);rx_tr=rx[:128]
    eq=MMSEEqualizer(num_taps=16)
    s_est,_=eq(rx,tx_tr,rx_tr,20)
    ber=((s_est>0).float()!=bits).float().mean().item()
    print(f"MMSE BER={ber:.5f} @SNR=20dB")
    for snr in [0,5,10,15,20,25,30]:
        c2=RayleighMultipathChannel(num_taps=16,delay_spread=10,snr_db=snr)
        rx2=c2(tx); s2,_=eq(rx2,tx_tr,rx2[:128],snr)
        b2=((s2>0).float()!=bits).float().mean().item()
        print(f"  SNR={snr:2d}dB BER={b2:.5f}")
    print("OK")
if __name__=="__main__":test()