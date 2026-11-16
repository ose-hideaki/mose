"""Single-lens physics checks independent of binary inverse-ray dispatch."""
import importlib.util
import unittest
import numpy as np

AVAILABLE=all(importlib.util.find_spec(x) for x in ('jax','lcbinint'))

@unittest.skipUnless(AVAILABLE,'install lcbinint/JAX')
class PointBackendTests(unittest.TestCase):
    def test_uniform_central_source_and_gradient(self):
        import jax
        import jax.numpy as jnp
        from microevent_tools.kernels.jax_fspl_contour import fspl_magnification
        rho=.01
        result=fspl_magnification(0.,rho)
        self.assertTrue(bool(result.converged))
        self.assertAlmostEqual(float(result.magnification)/(np.sqrt(rho*rho+4)/rho),1.,places=5)
        function=lambda u:fspl_magnification(u,rho).magnification
        u=.02;h=1e-6
        gradient=float(jax.grad(function)(u))
        difference=float((function(u+h)-function(u-h))/(2*h))
        np.testing.assert_allclose(gradient,difference,rtol=1e-4)

    def test_annual_single_lens_is_from_lcbinint_trajectory(self):
        import jax.numpy as jnp
        import lcbinint
        from microevent_tools.kernels.point_backend import PointLens
        engine=PointLens();engine.SetObjectCoordinates('18:00:00 -29:00:00');engine.t0_par=8000.
        times=jnp.array([7900.,8000.,8100.]);raw=jnp.array([.2,np.log(100.),8000.,.3,-.2])
        import jax
        result=jax.jit(lambda p:engine.PSPLLightCurveParallax(p,times)[0])(raw)
        gradient=jax.jit(jax.grad(lambda p:jnp.sum(engine.PSPLLightCurveParallax(p,times)[0])))(raw)
        self.assertTrue(np.isfinite(gradient).all())
        curve=lcbinint.LightCurve(model=lcbinint.Model(finite_source=False,parallax=True,
            sky=lcbinint.obs.SkyCoord('18:00:00','-29:00:00'),t_ref=8000.),
            options=lcbinint.Options(coordinates='vbm_center_of_mass'))
        path=curve.source_trajectory(np.asarray(times),dict(u0=.2,tE=100.,t0=8000.,alpha=0.,s=1.2,q=.001,piEN=.3,piEE=-.2))
        u=np.hypot(path.x,path.y)
        np.testing.assert_allclose(result,(u*u+2)/(u*np.sqrt(u*u+4)),rtol=1e-5)

    @unittest.skipUnless(importlib.util.find_spec('VBMicrolensing'),'optional independent VBM reference')
    def test_uniform_source_reference(self):
        from pathlib import Path
        import jax.numpy as jnp
        import VBMicrolensing
        from microevent_tools.kernels.point_backend import PointLens
        native=VBMicrolensing.VBMicrolensing();native.Tol=1e-6
        native.LoadESPLTable(str(Path(VBMicrolensing.__file__).parent/'data/ESPL.tbl'))
        rho=.01
        t=np.array([-.02,-.005,0.,.005,.02,1.])
        raw=jnp.array([np.log(.001),0.,0.,np.log(rho)])
        result=PointLens().ESPLLightCurve(raw,jnp.asarray(t))[0]
        expected=[native.ESPLMag2(float(np.hypot(.001,value)),rho) for value in t]
        np.testing.assert_allclose(result,expected,rtol=5e-4)
