# -*- coding: utf-8 -*-
"""3GPP 无线信道模型模块"""
import math
import torch
import torch.nn.functional as F

EPA=[(0,0),(30,-1),(70,-2),(90,-3),(110,-8),(190,-17.2),(410,-20.8)]
EVA=[(0,0),(30,-1.5),(150,-1.4),(310,-3.6),(370,-0.6),(710,-9.1),(1090,-7),(1730,-12),(2510,-16.9)]
ETU=[(0,-1),(50,-1),(120,-1),(200,0),(230,0),(500,0),(1600,-3),(2300,-5),(5000,-7)]
PROF={"epa":{"pd":EPA,"d":"EPA 7抽头"},"eva":{"pd":EVA,"d":"EVA 9抽头"},"etu":{"pd":ETU,"d":"ETU 9抽头"}}

class ThreeGPPChannel:
    def __init__(s,p="eva",snr_db=10,tv=False,dp=70,sr=1e6,sd=42,**kw):
        s.pn=p.lower();s.pr=PROF[s.pn];s.snr_db=snr_db;s.sl=10**(snr_db/10)
        s.tv=tv;s.dp=dp;s.sr=sr;s._taps=None
        pd=s.pr["pd"];s.dl=torch.tensor([d/1000 for d,_ in pd]);s.nt=len(pd)
        pl=[10**(p/10) for _,p in pd];pt=torch.tensor(pl);s.pdp=pt/pt.sum()
        s.md=int(s.dl.max().ceil().item())

    def reset_taps(s):
        r=torch.randn(s.nt)*s.pdp.sqrt()
        i=torch.randn(s.nt)*s.pdp.sqrt()
        s._taps=torch.complex(r,i)

    def set_taps(s,t):s._taps=t.clone()
    def set_snr(s,snr):s.snr_db=snr;s.sl=10**(snr/10)
    def update_taps_if_needed(s,t):
        if not s.tv or s._taps is None:return
        rh=math.exp(-2*math.pi*s.dp/s.sr)
        nr=torch.randn(s.nt)*(1-rh**2)**0.5
        ni=torch.randn(s.nt)*(1-rh**2)**0.5
        s._taps=rh*s._taps+torch.complex(nr,ni)*s.pdp.sqrt()

    def convolve(s,sym):
        L=sym.shape[0];ir=sym[:,0].view(1,1,L);ii=sym[:,1].view(1,1,L)
        tr=torch.zeros(s.md+1);ti=torch.zeros(s.md+1)
        for k,d in enumerate(s.dl):
            p=int(round(d.item()))
            if p<=s.md:tr[p]+=s._taps[k].real;ti[p]+=s._taps[k].imag
        nz=(tr.abs()+ti.abs())>1e-10;el=max(nz.int().sum().item(),1)
        tr=tr.flip(0)[-el:].view(1,1,-1);ti=ti.flip(0)[-el:].view(1,1,-1)
        ir=F.pad(ir,(el-1,0));ii=F.pad(ii,(el-1,0))
        yr=(F.conv1d(ir,tr)-F.conv1d(ii,ti)).squeeze()
        yi=(F.conv1d(ir,ti)+F.conv1d(ii,tr)).squeeze()
        return torch.stack([yr[:L],yi[:L]],-1)

    def add_awgn(s,sig):
        sp=(sig**2).mean();np=sp/s.sl;ns=torch.sqrt(np/2)
        return sig+torch.randn_like(sig)*ns

    def __call__(s,sym):s.reset_taps();r=s.convolve(sym);return s.add_awgn(r)
    def summary(s):return "3GPP "+s.pn.upper()+" | "+s.pr["d"]+" | SNR="+str(s.snr_db)+"dB | 时延="+str(s.md)+"sym"

class RayleighMultipathChannel(ThreeGPPChannel):
    def __init__(s,**kw):
        kw.pop("num_taps",None);kw.pop("delay_spread",None);kw.pop("seed",None)
        if "profile" not in kw:kw["p"]="eva"
        elif "profile" in kw:kw["p"]=kw.pop("profile")
        super().__init__(**kw)

def test_channel():
    print("3GPP信道测试")
    for n in ["epa","eva","etu"]:
        c=ThreeGPPChannel(p=n,snr_db=15)
        st=torch.tensor([[1.,0.],[-1.,0.]]*50)
        r=c(st);assert r.shape==st.shape
        print(" "+c.summary(),"out",r.shape)
    l=RayleighMultipathChannel(snr_db=20)
    r=l(torch.tensor([[1.,0.],[-1.,0.]]*50))
    print(" 兼容层:",l.summary(),"out",r.shape)
    l2=RayleighMultipathChannel(num_taps=16,delay_spread=10,snr_db=10.0,seed=42)
    print(" 旧参数:",l2.summary());l2.reset_taps();l2.convolve(st)
    l2.set_snr(5);l2.update_taps_if_needed(0);print(" set_snr+update ok")
    l3=RayleighMultipathChannel(snr_db=15,profile="etu")
    print(" ETU:",l3.summary())
    print("全部测试通过")

if __name__=="__main__":test_channel()
