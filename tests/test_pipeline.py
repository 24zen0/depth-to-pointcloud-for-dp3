import json
from pathlib import Path
import sys
import unittest
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'Sustech_pcGenerate'))
from Convert_PointCloud import PointCloudGenerator
from Cloud_Process import preprocess_point_cloud
from main import convert_frames

class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.cal = json.loads((ROOT/'examples/calibration.json').read_text())
    def test_depth_scale_and_single_transform(self):
        c=self.cal
        g=PointCloudGenerator(camera_matrix=c['intrinsics'],camera_to_world=c['camera_to_world'],depth_scale=c['depth_scale'],width=6,height=4)
        pts,_=g.generateCroppedPointCloud(np.full((4,6),1000))
        np.testing.assert_allclose(pts[0],[-0.15,-0.15,1],atol=1e-7)
    def test_crop_is_saved_and_invalid_frame_keeps_shape(self):
        clouds,valid=convert_frames(np.load(ROOT/'examples/depth.npy'),self.cal,32)
        self.assertEqual(clouds.shape,(2,32,3))
        np.testing.assert_array_equal(valid,[True,False])
        self.assertTrue(np.all(clouds[0,:,0]>-0.1))
        self.assertTrue(np.all(clouds[0,:,0]<0.31))
        np.testing.assert_array_equal(clouds[1],0)
        np.testing.assert_allclose(clouds[0,:,2],1)
    def test_rgb_alignment_and_repeat_sampling(self):
        points=np.array([[0,0,1,10,20,30],[2,0,1,40,50,60]],dtype=float)
        sampled=preprocess_point_cloud(points,workspace=[[-1,1],[-1,1],[0,2]],num_points=4)
        np.testing.assert_array_equal(sampled,np.tile(points[0],(4,1)))
    def test_bad_image_size_rejected(self):
        with self.assertRaises(ValueError):
            convert_frames(np.ones((2,3,6)),self.cal)
    def test_empty_crop_flagged(self):
        c={**self.cal,'workspace':[[10,11],[10,11],[10,11]]}
        clouds,valid=convert_frames(np.load(ROOT/'examples/depth.npy'),c,4)
        self.assertFalse(valid.any())
        self.assertEqual(clouds.shape,(2,4,3))

if __name__=='__main__': unittest.main()
