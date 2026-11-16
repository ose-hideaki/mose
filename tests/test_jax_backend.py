"""Optional checks for the differentiable backend and legacy model round trips."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np


AVAILABLE = all(importlib.util.find_spec(x) for x in ('jax','numpyro','autodiff_vbm','VBMicrolensing'))


@unittest.skipUnless(AVAILABLE,'install and build the jax extra and vendor backend')
class JaxBackendTests(unittest.TestCase):
    def test_caustic_trajectory_matches_backend(self):
        from microevent_tools.kernels.plot_caustic import source_trajectory
        from autodiff_vbm import AutodiffVBM
        p=dict(t0=8000., tE=12., u0=-.2, q=.001, s=1.2, alpha=1.1, rho=.002)
        tau=np.linspace(-1,1,9)
        raw=[np.log(p['s']),np.log(p['q']),p['u0'],p['alpha'],np.log(p['rho']),np.log(p['tE']),p['t0']]
        expected=AutodiffVBM().BinaryLightCurve(raw,p['t0']+tau*p['tE'])
        x,y=source_trajectory(p,tau)
        np.testing.assert_allclose(x,expected[1],atol=1e-12)
        np.testing.assert_allclose(y,expected[2],atol=1e-12)

    def test_native_static_magnifications_and_gradient(self):
        import jax
        import jax.numpy as jnp
        import VBMicrolensing
        from autodiff_vbm import AutodiffVBM
        jax.config.update('jax_enable_x64',True)
        engine=AutodiffVBM();native=VBMicrolensing.VBMicrolensing()
        # Match BOTH tolerance knobs. Different RelTol defaults otherwise
        # select different finite-source approximations near high magnification.
        engine.Tol=engine.RelTol=native.Tol=native.RelTol=1e-7
        t=np.linspace(7990,8010,11)
        p=np.array([np.log(1.2),np.log(.001),-.2,1.,np.log(.002),np.log(12.),8000.])
        result=np.asarray(engine.BinaryLightCurve(p,t)[0]);reference=np.asarray(native.BinaryLightCurve(p.tolist(),t.tolist())[0])
        np.testing.assert_allclose(result,reference,rtol=1e-5,atol=1e-5)
        gradient=np.asarray(jax.grad(lambda value:jnp.sum(engine.BinaryLightCurve(value,t)[0]))(jnp.asarray(p)))
        self.assertTrue(np.isfinite(gradient).all())

    def test_short_parallax_nuts_preserves_angle_and_columns(self):
        root=Path(__file__).resolve().parents[1]
        listing=root/'examples/demo/data/2026/DEMO/v1/list'
        radec=root/'examples/demo/data/2026/DEMO/RADec'
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);model=base/'initial'
            # t0,tE,u0,q,s,alpha,rho,reserved,pai1,pai2
            values=[8000.,12.,.2,.001,1.2,1.,.002,0.,.01,-.02]+[0.]*8
            model.write_text('#chi2/dof= 1000 Ndata= 241 nparam= 13 redchi2= 4.38\n#BEST '+' '.join(map(str,values))+'\n# t_ref= 8000\n')
            output=base/'chain.npz';best=base/'best'
            command=[sys.executable,'-m','microevent_tools.kernels.jax_binary_mcmc',
                '--model',str(model),'--list',str(listing),'--radec',str(radec),
                '--output',str(output),'--bestfit-output',str(best),'--parallax',
                '--data-center','8000','--time-window','40','--warmup','8','--samples','8',
                '--chains','1','--max-tree-depth','2','--seed','42']
            done=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(done.returncode,0,done.stderr)
            with np.load(output) as d:
                self.assertEqual(d['raw_samples_by_chain'].shape,(1,8,9))
                np.testing.assert_array_equal(d['samples'][:,5],d['raw_samples'][:,5])
                summary=json.loads(Path(str(output)+'.json').read_text())
                chosen=d['samples'][summary['best_sample_index']]
            line=next(x for x in best.read_text().splitlines() if x.startswith('#BEST'))
            saved=np.asarray([float(x) for x in line.split()[1:]])
            np.testing.assert_allclose(saved[:7],chosen[:7],rtol=1e-10)
            self.assertEqual(saved[7],0.)
            np.testing.assert_allclose(saved[8:10],chosen[7:9],rtol=1e-10)
            self.assertEqual(summary['diagnostics']['chain_count'],1)


if __name__=='__main__':unittest.main()
