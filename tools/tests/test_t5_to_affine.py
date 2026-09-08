"""The T5 -> affine Q2 re-pack must reproduce the ternary values exactly."""

import unittest

import torch

from tools import quantize_qwen4_flash_next_t5 as q
from tools import t5_to_affine


class T5ToAffineTests(unittest.TestCase):
    def test_repack_is_lossless(self):
        generator = torch.Generator().manual_seed(11)
        weight = torch.randn((3, 8, 256), generator=generator).bfloat16()
        packed, scales = q.weighted_ternary_chunk(weight, fitter="prefix")
        codes = q.unpack_t5(packed, 256).float()
        ternary = ((codes.reshape(3, 8, 2, 128) - 1) * scales.float()[..., None]).reshape(3, 8, 256)

        affine_w, affine_s, affine_b = t5_to_affine.t5_bank_to_affine(packed, scales, device="cpu", experts_per_chunk=2)
        self.assertEqual((tuple(affine_w.shape), affine_w.dtype), ((3, 8, 16), torch.uint32))
        self.assertTrue(torch.equal(affine_s, scales))
        self.assertEqual(affine_b.dtype, torch.bfloat16)
        rebuilt = q.unpack_affine(affine_w, 256, 2).float().reshape(3, 8, 2, 128)
        rebuilt = (rebuilt * affine_s.float()[..., None] + affine_b.float()[..., None]).reshape(3, 8, 256)
        self.assertTrue(torch.equal(rebuilt, ternary))
        self.assertLessEqual(int(q.unpack_affine(affine_w, 256, 2).max()), 2)


if __name__ == "__main__":
    unittest.main()
