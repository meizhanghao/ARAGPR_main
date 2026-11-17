"""Protect fixed denominators, class contrasts, and undefined retention cases."""
import unittest

import numpy as np

from experiments.diagnostics.run_teacher_alignment_audit import summarize_alignment


class AlignmentTests(unittest.TestCase):
    def test_same_accuracy_can_hide_lost_original_corrections(self):
        labels=np.zeros(4,dtype=int)
        frozen=np.array([1,1,0,0]);decoded=np.array([0,1,0,1]);teacher=np.array([1,1,0,0])
        self.assertEqual(int((decoded==labels).sum()),int((teacher==labels).sum()))
        q=np.tile([3.,-2.,1.,0.],(4,1))
        row,_=summarize_alignment(labels,frozen,decoded,teacher,q,q)
        self.assertEqual(row['n_original_corrections'],1)
        self.assertEqual(row['n_retained_corrections'],0)
        self.assertEqual(row['correction_retention_pct'],0.)

    def test_decoded_topk_is_fixed_and_zero_terms_are_excluded(self):
        qd=np.array([[3.,-2.,0.,0.]])
        qt=np.array([[0.,1.,0.,5.]])
        row,detail=summarize_alignment([0],[1],[0],[0],qd,qt)
        self.assertEqual(row['correction_retention_pct'],100.)
        self.assertEqual(row['sign_eligible_coordinates'],1)
        self.assertEqual(row['sign_matches'],0)
        self.assertEqual(row['top3_sign_agreement_pct'],0.)
        np.testing.assert_array_equal(detail['top_decoded'],[[0,1,2]])
        self.assertAlmostEqual(row['top3_overlap_pct'],200/3)

    def test_no_original_corrections_has_undefined_retention(self):
        row,_=summarize_alignment([0],[0],[1],[0],[[1.,2.,3.]],[[3.,2.,1.]])
        self.assertIsNone(row['correction_retention_pct'])
        self.assertEqual(row['n_original_corrections'],0)


if __name__=='__main__':
    unittest.main(verbosity=2)
