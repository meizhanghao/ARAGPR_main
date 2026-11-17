"""Mathematical invariants that catch missing offsets and changed class contrasts."""
import numpy as np
import unittest

from experiments.diagnostics.run_posthoc_shape_diagnostics import compose_affine, compose_input_scaler, fixed_contributions, ranked_alignment


def test_composition_keeps_both_biases_and_nonidentity_coordinate_scaler():
    rng=np.random.default_rng(7)
    B=rng.normal(size=(5,7));bd=rng.normal(size=5)
    W=rng.normal(size=(3,5));bh=rng.normal(size=3)
    mu=rng.normal(size=5);scale=np.exp(rng.normal(size=5));x=rng.normal(size=(11,7))
    A,c=compose_affine(B,bd,W,bh,mu,scale)
    reference=((x@B.T+bd-mu)/scale)@W.T+bh
    np.testing.assert_allclose(x@A.T+c,reference,atol=2e-12)
    np.testing.assert_allclose(c,((bd-mu)/scale)@W.T+bh,atol=2e-12)
    assert not np.allclose(c,bh)  # This is deliberately stronger than head-bias-only.


def test_invalid_scale_is_rejected_before_intervention():
    with unittest.TestCase().assertRaises(ValueError):
        compose_affine(np.ones((2,3)),np.ones(2),np.ones((4,2)),np.ones(4),np.zeros(2),np.zeros(2))


def test_teacher_replacement_keeps_original_pair_even_when_argmax_changes():
    semantic=np.array([[2.,0.,-1.],[0.,3.,1.]])
    W=np.eye(3);bias=np.array([.1,-.2,.3]);eta=.7
    vd=np.array([[0.,5.,0.],[4.,0.,0.]])
    vt=np.array([[0.,0.,9.],[0.,0.,9.]])
    full=semantic+eta*(vd@W.T+bias)
    results=fixed_contributions(semantic,full,vd,vt,W,bias,eta)
    assert np.array_equal(results['b'],[0,1])
    assert np.array_equal(results['f'],[1,0])
    assert np.array_equal(results['teacher_logits'].argmax(1),[2,2])
    # Contrast remains [1 vs 0] / [0 vs 1], not [2 vs 0] / [2 vs 1].
    np.testing.assert_allclose(results['teacher_margin_same_pair'],[-2.21,-2.79])


def test_zero_terms_do_not_inflate_direction_agreement():
    result=ranked_alignment([3.,-2.,0.,0.],[0.,1.,0.,0.],topk=3)
    assert result['sign_eligible']==1
    assert result['sign_matches']==0
    assert result['top3_sign_agreement_pct']==0.


def test_input_scaler_offsets_are_kept_in_raw_clip_composition():
    rng=np.random.default_rng(14)
    A=rng.normal(size=(3,7));c=rng.normal(size=3)
    mean=rng.normal(size=7);scale=np.exp(rng.normal(size=7));x=rng.normal(size=(9,7))
    raw_A,raw_c=compose_input_scaler(A,c,mean,scale)
    np.testing.assert_allclose(x@raw_A.T+raw_c,((x-mean)/scale)@A.T+c,atol=2e-12)


if __name__ == '__main__':
    suite = unittest.TestSuite(unittest.FunctionTestCase(test) for name, test in list(globals().items())
                               if name.startswith('test_') and callable(test))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
