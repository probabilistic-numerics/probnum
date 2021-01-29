# import unittest
#
# import probnum.filtsmooth as pnfs
# import probnum.random_variables as pnrv
#
#
# class TestDefaultStoppingCriterion(unittest.TestCase):
#     """The default stoppingcriterion should make sure that no filter updates are
#     repeated but also make sure that whenever iterated filtsmooth is attempted, an
#     exception is thrown."""
#
#     def setUp(self):
#         self.stopcrit = pnfs.StoppingCriterion()
#
#     def test_continue_predict_iteration(self):
#         self.assertFalse(self.stopcrit.continue_predict_iteration())
#
#     def test_continue_update_iteration(self):
#         self.assertFalse(self.stopcrit.continue_update_iteration())
#
#     def test_continue_filtsmooth_iteration(self):
#         with self.assertRaises(NotImplementedError):
#             self.assertTrue(self.stopcrit.continue_filtsmooth_iteration())
#
#
# class TestFixedPointIteration(unittest.TestCase):
#     def setUp(self):
#         self.stopcrit = pnfs.FixedPointStopping(
#             atol=1e-4,
#             rtol=1e-4,
#             max_num_predicts_per_step=10,
#             max_num_updates_per_step=10,
#             max_num_filtsmooth_iterations=10,
#         )
#
#     def test_continue_predict_iteration(self):
#         self.assertEqual(self.stopcrit.num_predict_iterations, 0)
#         x0 = 1.0
#         while self.stopcrit.continue_predict_iteration(pred_rv=pnrv.Constant(x0)):
#             x0 *= 0.1
#         self.assertGreaterEqual(self.stopcrit.num_predict_iterations, 1)
#
#     def test_continue_predict_iteration_exception(self):
#         """No improvement at all raises error eventually."""
#         worsening = 0.1
#         value = 0.0
#         with self.assertRaises(RuntimeError):
#             while self.stopcrit.continue_predict_iteration(
#                 pred_rv=pnrv.Constant(value)
#             ):
#                 value += worsening
#
#     def test_continue_update_iteration(self):
#         self.assertEqual(self.stopcrit.num_update_iterations, 0)
#         x0 = 1.0
#         while self.stopcrit.continue_update_iteration(upd_rv=pnrv.Constant(x0)):
#             x0 *= 0.1
#         self.assertGreaterEqual(self.stopcrit.num_update_iterations, 1)
#
#     def test_continue_update_iteration_exception(self):
#         """No improvement at all raises error eventually."""
#         worsening = 0.1
#         value = 0.0
#         with self.assertRaises(RuntimeError):
#             while self.stopcrit.continue_update_iteration(upd_rv=pnrv.Constant(value)):
#                 value += worsening
#
#
# if __name__ == "__main__":
#     unittest.main()
