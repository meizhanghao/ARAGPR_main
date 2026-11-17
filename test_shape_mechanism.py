"""Scientific invariants for the mechanism experiment, not accuracy expectations."""
import unittest
from pathlib import Path
import tempfile

import cv2
import numpy as np

from run_derm7pt_mechanism import training_permutation,case_metadata
from utils.shape_perturbations import (appearance_blur,recover_crop_position,
                                      warp_image_and_masks)


class MechanismInvariants(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(71)
        self.image=rng.integers(0,256,(128,160,3),dtype=np.uint8)
        self.mask=np.zeros((128,160),np.uint8)
        cv2.ellipse(self.mask,(80,64),(26,35),0,0,360,255,-1)

    def test_complete_vector_permutation_and_class_preservation(self):
        labels=np.repeat([0,1],20)
        for mode in ("shuffled","within_class_shuffled"):
            p=training_permutation(labels,mode,42)
            np.testing.assert_array_equal(np.sort(p),np.arange(40))
            self.assertFalse(np.array_equal(p,np.arange(40)))
            np.testing.assert_array_equal(p,training_permutation(labels,mode,42))
            if mode=="within_class_shuffled":
                np.testing.assert_array_equal(labels[p],labels)

    def test_metadata_duplicate_modality_keeps_case(self):
        with tempfile.TemporaryDirectory() as temporary:
            metadata = Path(temporary) / "meta.csv"
            metadata.write_text("case_num,clinic,derm\nsynthetic-case,example.jpg,example.jpg\n",
                                encoding="utf-8")
            cases,modalities=case_metadata(["example.jpg"], metadata_path=metadata)
        self.assertEqual(cases[0],"synthetic-case")
        self.assertEqual(modalities[0],"clinic_and_derm_metadata")

    def test_crop_registration_requires_correct_unique_position(self):
        crop=self.image[20:90,35:100].copy()
        bbox,diagnostic=recover_crop_position(self.image,crop)
        self.assertEqual(bbox,(35,20,65,70))
        self.assertGreater(diagnostic["match_score"],.999)
        with self.assertRaises(ValueError):
            recover_crop_position(self.image,np.zeros_like(crop))
        with self.assertRaises(ValueError):
            recover_crop_position(self.image,np.roll(crop,7,axis=0))

    def test_identity_warp_and_shared_nonfolding_geometry(self):
        image,masks,jac=warp_image_and_masks(self.image,[self.mask],0.)
        np.testing.assert_array_equal(image,self.image)
        np.testing.assert_array_equal(masks[0],self.mask)
        moved,regions,jac=warp_image_and_masks(self.image,[self.mask],.2)
        self.assertGreater(jac,.25)
        self.assertFalse(np.array_equal(moved,self.image))
        self.assertFalse(np.array_equal(regions[0],self.mask))
        self.assertTrue(np.isin(regions[0],[0,255]).all())

    def test_appearance_edit_keeps_masks_and_boundary_pixels(self):
        boundary=cv2.morphologyEx(self.mask,cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))>0
        for scope in ("inside","outside"):
            image,masks,fraction=appearance_blur(self.image,[self.mask],1.5,scope)
            np.testing.assert_array_equal(masks[0],self.mask)
            np.testing.assert_array_equal(image[boundary],self.image[boundary])
            self.assertGreater(fraction,0)
            self.assertFalse(np.array_equal(image,self.image))


if __name__=="__main__": unittest.main()
