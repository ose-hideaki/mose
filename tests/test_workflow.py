"""Portable configuration, data selection and resumable stage behavior."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from microevent_tools.config import load
from microevent_tools.data import load_datasets
from microevent_tools.execution import Runner, Step
from microevent_tools.pipeline import plan
from microevent_tools.scheduler import write_script


EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='microevent test ')
        self.root = Path(self.temp.name)
        shutil.copytree(EXAMPLES, self.root/'examples')
        self.config = self.root/'examples/point.toml'
        self.settings = load(self.config)

    def tearDown(self):
        self.temp.cleanup()

    def test_plan_is_read_only_and_keeps_same_cut(self):
        paths, data, steps = plan(self.settings)
        self.assertFalse(paths.output.exists())
        self.assertEqual(sum(len(d.values) for d in data), 241)
        self.assertEqual(len(steps), 10)
        self.assertEqual(len(steps[-1].outputs), 4)
        self.assertEqual(steps[-1].outputs[0].name, 'DEMO-0.POINT_LIGHTCURVES.pspl.png')
        self.assertEqual(len({s.name for s in steps}), len(steps))
        self.assertTrue(all('--data-center' in s.command for s in steps[:8]))
        centers = [s.command[s.command.index('--data-center')+1] for s in steps[:8]]
        self.assertEqual(set(centers), {'8000.0'})
        self.assertTrue(all(' -- ' not in item for s in steps for item in s.command))

    def test_unknown_keys_and_path_traversal_rejected(self):
        for extra in ['\n[unknown]\nx=1\n', '\n[mcmc]\nsampels=10\n']:
            p=self.root/'bad.toml';p.write_text('[event]\nname="DEMO"\n'+extra)
            with self.assertRaises(ValueError):load(p)
        p=self.root/'bad.toml';p.write_text('[event]\nname="../escape"\n')
        with self.assertRaises(ValueError):load(p)

    def test_ambiguous_relative_data_fails(self):
        source=self.root/'data';(source/'v1').mkdir(parents=True)
        for path in [source/'v1/a.flux',source/'a.flux']:path.write_text('1 2 .1\n')
        listing=source/'v1/list';listing.write_text('a.flux OBS I\n')
        with self.assertRaises(ValueError):load_datasets(listing)

    def test_scheduler_is_configured_not_site_specific(self):
        path=write_script(self.settings,self.root/'job.sh')
        text=path.read_text()
        self.assertIn('OMP_NUM_THREADS=2',text)
        self.assertIn('nice -n 19',text)
        self.assertNotIn('/whome/',text)
        subprocess.run(['bash','-n',str(path)],check=True)

    def test_resume_and_changed_inputs(self):
        output=self.root/'run';result=output/'result'
        code='from pathlib import Path; import sys; Path(sys.argv[1]).write_text("done")'
        step=Step('test',(sys.executable,'-c',code,str(result)),(result,))
        runner=Runner(self.settings,output,[self.config])
        runner.run(step);stamp=result.stat().st_mtime_ns
        Runner(self.settings,output,[self.config]).run(step)
        self.assertEqual(stamp,result.stat().st_mtime_ns)
        self.config.write_text(self.config.read_text()+'\n# changed input\n')
        with self.assertRaises(RuntimeError):Runner(self.settings,output,[self.config]).run(step)

    def test_gate_is_not_a_failure(self):
        output=self.root/'run';result=output/'gate.tsv'
        code='from pathlib import Path; import sys; Path(sys.argv[1]).write_text("skip"); sys.exit(20)'
        step=Step('binary_gate',(sys.executable,'-c',code,str(result)),(result,),accepted_codes=(0,20))
        runner=Runner(self.settings,output,[self.config]);runner.run(step)
        self.assertFalse(runner.binary_passed)
        skipped=Step('skip',(sys.executable,'-c','raise RuntimeError("should not run")'),condition='binary')
        runner.run(skipped)
        restored=Runner(self.settings,output,[self.config]);restored.run(step)
        self.assertFalse(restored.binary_passed)

    def test_full_plan_contains_post_binary_and_gate(self):
        config=load(self.root/'examples/full.toml')
        _,_,steps=plan(config)
        names=[s.name for s in steps]
        for key in ('external_grid','collect_grid_modes','binary_gate','binary_static_mcmc','binary_parallax_mcmc','select_binary','post_pspl_mcmc','report_binary'):
            self.assertIn(key,names)
        self.assertTrue(all(s.condition=='binary' for s in steps if s.name.startswith('post_')))


if __name__=='__main__':unittest.main()
