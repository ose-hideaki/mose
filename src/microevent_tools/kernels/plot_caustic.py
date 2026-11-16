#!/usr/bin/env python3
"""Plot binary-lens caustics, lens positions, and the source trajectory."""
from __future__ import annotations

# Support both installed modules and the internal subprocess entry points.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "microevent_tools.kernels"

import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from VBMicrolensing import VBMicrolensing
from .jax_binary_static_fit import read_best

def source_trajectory(pars, tau, *, radec=None, reference_time=None):
    """Use radians and the same center-of-mass trajectory as the fit backend."""
    if radec is None:
        a, u0 = pars["alpha"], pars["u0"]
        return u0*np.sin(a)-tau*np.cos(a), -u0*np.cos(a)-tau*np.sin(a)
    from autodiff_vbm import AutodiffVBM
    engine = AutodiffVBM()
    engine.SetObjectCoordinates(" ".join(radec))
    engine.t0_par_fixed = 1
    engine.t0_par = float(reference_time)
    p = [np.log(pars["s"]), np.log(pars["q"]), pars["u0"], pars["alpha"],
         np.log(pars["rho"]), np.log(pars["tE"]), pars["t0"], pars["pai1"], pars["pai2"]]
    _, x, y = engine.BinaryLightCurveParallax(p, pars["t0"]+tau*pars["tE"])
    return np.asarray(x), np.asarray(y)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--trajectory",type=float,default=5.0,help="half range in tE")
    p.add_argument("--parallax", action="store_true")
    p.add_argument("--radec", type=Path)
    args=p.parse_args()
    pars=read_best(args.model,{"t0":0.,"tE":1.,"u0":.1,"q":1e-3,"s":1.,"alpha":0.,"rho":0.})
    q=float(pars["q"]); s=float(pars["s"]); alpha=float(pars["alpha"]); u0=float(pars["u0"])
    fig,ax=plt.subplots(figsize=(8,7),constrained_layout=True)
    if q <= 0 or s <= 0:
        ax.text(.5,.5,"FSPL: no binary caustic",ha="center",va="center",transform=ax.transAxes)
        ax.set_axis_off()
    else:
        v=VBMicrolensing(); caustics=v.Caustics(s,q)
        for k,c in enumerate(caustics):
            ax.plot(c[0],c[1],color="#c53030",lw=1.5,label="caustic" if k==0 else None)
        # center-of-mass coordinates, matching binfit.pl convention
        x1=-q*s/(1+q); x2=s/(1+q)
        ax.scatter([x1,x2],[0,0],marker="o",s=90,facecolors="white",
                   edgecolors=["#3569b0","#2f855a"],linewidths=1.8,
                   zorder=4,label="lens")
        tau=np.linspace(-args.trajectory,args.trajectory,1000)
        radec = None
        reference = None
        if args.parallax:
            if args.radec is None:
                raise ValueError("--parallax requires --radec")
            from .jax_binary_parallax_fit import read_radec, read_parallax, read_reference_time
            radec = read_radec(args.radec)
            pars["pai1"], pars["pai2"] = read_parallax(args.model)
            reference = read_reference_time(args.model, pars["t0"])
        x, y = source_trajectory(pars, tau, radec=radec, reference_time=reference)
        x0, y0 = source_trajectory(pars, np.array([0.]), radec=radec, reference_time=reference)
        ax.plot(x,y,color="black",lw=1.2,label="source trajectory")
        ax.scatter(x0,y0,s=45,color="black",zorder=5,label="source at t0")
        # Use a fixed source-plane range for consistent model comparisons.
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
        ax.set_aspect("equal",adjustable="box")
        ax.set_xlabel("source-plane x (Einstein radius)"); ax.set_ylabel("source-plane y (Einstein radius)")
        ax.set_title(f"{args.model.name}  q={q:.4g}, s={s:.4g}, alpha={alpha:.4g}")
        ax.grid(alpha=.2); ax.legend(fontsize=9)
    args.output.parent.mkdir(parents=True,exist_ok=True); fig.savefig(args.output,dpi=180); plt.close(fig)
    print(args.output)
if __name__=="__main__": main()
