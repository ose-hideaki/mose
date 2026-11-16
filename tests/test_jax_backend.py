"""Optional checks for the differentiable backend and legacy model round trips."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np


AVAILABLE = all(importlib.util.find_spec(x) for x in ('jax','numpyro','lcbinint'))


@unittest.skipUnless(AVAILABLE,'install and build the jax extra')
class JaxBackendTests(unittest.TestCase):
    def test_caustic_trajectory_matches_backend(self):
        from microevent_tools.kernels.plot_caustic import source_trajectory
        import lcbinint
        p=dict(t0=8000., tE=12., u0=-.2, q=.001, s=1.2, alpha=1.1, rho=.002)
        tau=np.linspace(-1,1,9)
        curve=lcbinint.LightCurve(options=lcbinint.Options(coordinates='vbm_center_of_mass'))
        expected=curve.source_trajectory(p['t0']+tau*p['tE'],p)
        x,y=source_trajectory(p,tau)
        np.testing.assert_allclose(x,expected.x,atol=1e-12)
        np.testing.assert_allclose(y,expected.y,atol=1e-12)

    def test_native_static_magnifications_and_gradient(self):
        import jax
        import jax.numpy as jnp
        import lcbinint
        jax.config.update('jax_enable_x64',True)
        options=dict(coordinates='vbm_center_of_mass',tol=1e-4,reltol=1e-4,max_source_bins=400)
        engine=lcbinint.LightCurve(options=lcbinint.Options(jax=True,**options))
        native=lcbinint.LightCurve(options=lcbinint.Options(**options))
        t=np.linspace(7990,8010,11)
        p=dict(t0=8000.,tE=12.,u0=-.2,q=.001,s=1.2,alpha=1.,rho=.002)
        result=np.asarray(engine(jnp.asarray(t),p));reference=np.asarray(native(t,p))
        np.testing.assert_allclose(result,reference,rtol=2e-4,atol=2e-4)
        loss=lambda u:jnp.sum(engine(jnp.asarray(t),dict(p,u0=u)))
        if importlib.util.find_spec('VBMicrolensing'):
            import VBMicrolensing
            vbm=VBMicrolensing.VBMicrolensing();vbm.Tol=vbm.RelTol=1e-4
            raw=[np.log(p['s']),np.log(p['q']),p['u0'],p['alpha'],np.log(p['rho']),np.log(p['tE']),p['t0']]
            np.testing.assert_allclose(result,vbm.BinaryLightCurve(raw,t.tolist())[0],rtol=5e-4,atol=5e-4)
        gradient=float(jax.grad(loss)(p['u0']))
        h=1e-5
        finite_difference=float((loss(p['u0']+h)-loss(p['u0']-h))/(2*h))
        self.assertTrue(np.isfinite(gradient))
        np.testing.assert_allclose(gradient,finite_difference,rtol=.02,atol=.02)

    def test_single_lens_and_prior_basis(self):
        import jax.numpy as jnp
        from microevent_tools.kernels.point_backend import PointLens
        from microevent_tools.kernels.jax_binary_parallax_fit import read_parallax
        import jax
        lens=PointLens()
        p=jnp.array([np.log(.2),np.log(12.),8000.])
        t=jnp.array([7990.,8000.,8010.])
        u=np.sqrt(.2**2+((np.asarray(t)-8000.)/12.)**2)
        expected=(u*u+2)/(u*np.sqrt(u*u+4))
        np.testing.assert_allclose(lens.PSPLLightCurve(p,t)[0],expected,rtol=1e-12)
        self.assertTrue(np.isfinite(jax.grad(lambda x:jnp.sum(lens.PSPLLightCurve(x,t)[0]))(p)).all())
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'old'
            path.write_text('#BEST 8000 12 .2 .001 1.2 1 .002 0 .1 .2\n')
            with self.assertRaisesRegex(ValueError,'basis'):
                read_parallax(path)

    def test_short_parallax_nuts_preserves_angle_and_columns(self):
        root=Path(__file__).resolve().parents[1]
        listing=root/'examples/demo/data/2026/DEMO/v1/list'
        radec=root/'examples/demo/data/2026/DEMO/RADec'
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);model=base/'initial'
            # t0,tE,u0,q,s,alpha,rho,reserved,piEN,piEE
            values=[8000.,12.,.2,.001,1.2,1.,.002,0.,.01,-.02]+[0.]*8
            model.write_text('# parallax_basis= NE\n#chi2/dof= 1000 Ndata= 241 nparam= 13 redchi2= 4.38\n#BEST '+' '.join(map(str,values))+'\n# t_ref= 8000\n')
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
