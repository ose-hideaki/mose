"""Physical native-backend checks and an end-to-end short MCMC."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np


AVAILABLE = all(importlib.util.find_spec(name) for name in ('emcee','VBMicrolensing'))


@unittest.skipUnless(AVAILABLE,'install the native extra')
class NativeTests(unittest.TestCase):
    def test_native_input_conventions(self):
        from argparse import Namespace
        from microevent_tools.kernels.native_fit import coordinates, load_data
        root=Path(__file__).resolve().parents[1]/'examples/demo/data/2026/DEMO'
        self.assertEqual(coordinates(Namespace(coords=None,radec_file=root/'RADec')),
                         '18:00:00 -29:00:00')
        data=load_data(root/'v1/list',8000.,40.)
        self.assertEqual(sum(len(d['values']) for d in data),241)

    def test_pspl_matches_analytic_and_profiles_flux(self):
        from microevent_tools.kernels.native_fit import NativeModel,linear_flux
        import VBMicrolensing
        table=Path(VBMicrolensing.__file__).parent/'data/ESPL.tbl'
        model=NativeModel('pspl',[],None,None,1e-4,table)
        t=np.linspace(-20,20,50);p=np.array([0.,10.,.2]);u=np.sqrt(.2**2+(t/10)**2)
        expected=(u*u+2)/(u*np.sqrt(u*u+4))
        result=model.magnification(p,t)
        np.testing.assert_allclose(result,expected,rtol=1e-12)
        fs,fb,cost=linear_flux(result,1.7*result+.3,np.full(len(t),.01))
        self.assertAlmostEqual(fs,1.7);self.assertAlmostEqual(fb,.3);self.assertLess(cost,1e-15)

    def test_signed_binary_and_seeded_mcmc(self):
        import VBMicrolensing
        from microevent_tools.kernels.native_fit import NativeModel
        table=Path(VBMicrolensing.__file__).parent/'data/ESPL.tbl'
        binary=NativeModel('binary_static',[],None,None,1e-4,table)
        result=binary.magnification(np.array([0.,10.,-.2,.002,.001,1.2,1.]),np.linspace(-20,20,20))
        self.assertTrue(np.isfinite(result).all())
        root=Path(__file__).resolve().parents[1]
        listing=root/'examples/demo/data/2026/DEMO/v1/list'
        with tempfile.TemporaryDirectory() as tmp:
            outputs=[]
            for name in ['a','b']:
                prefix=Path(tmp)/name
                command=[sys.executable,'-m','microevent_tools.kernels.native_fit',
                    '--model','pspl','--datalist',str(listing),'--init','8000','12','.2',
                    '--lower','7999','10','.1','--upper','8001','14','.3',
                    '--nwalkers','12','--steps','20','--discard','5','--thin','1',
                    '--skip-minimize','--no-progress','--seed','42','--output-prefix',str(prefix)]
                done=subprocess.run(command,capture_output=True,text=True)
                self.assertEqual(done.returncode,0,done.stderr)
                with np.load(str(prefix)+'.chain.npz') as d:outputs.append(d['chain'].copy())
                report=json.loads(Path(str(prefix)+'.summary.json').read_text())
                self.assertFalse(report['converged'])
                self.assertEqual(report['steps'],20)
            np.testing.assert_array_equal(*outputs)


if __name__=='__main__':unittest.main()
