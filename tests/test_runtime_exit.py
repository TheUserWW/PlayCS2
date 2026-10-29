"""Run autograd in a child process so native shutdown crashes fail the test."""

import subprocess
import sys
import textwrap
import unittest


class RuntimeExitTests(unittest.TestCase):
    def test_cpu_cuda_and_worker_backward_exit_cleanly(self):
        program = textwrap.dedent("""
            import threading
            import torch

            torch.set_num_threads(2)
            devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
            errors = []

            def backward(device):
                try:
                    x = torch.ones(10, device=device, requires_grad=True)
                    (x * x).sum().backward()
                    torch.testing.assert_close(x.grad, torch.full_like(x, 2))
                    if device == 'cuda':
                        torch.cuda.synchronize()
                except BaseException as error:
                    errors.append(error)

            for device in devices:
                backward(device)
                worker = threading.Thread(target=backward, args=(device,))
                worker.start()
                worker.join()
            if errors:
                raise errors[0]
            print('AUTOGRAD_OK:' + ','.join(devices), flush=True)
        """)
        result = subprocess.run(
            [sys.executable, "-X", "faulthandler", "-u", "-c", program],
            capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("AUTOGRAD_OK:cpu", result.stdout)


if __name__ == "__main__":
    unittest.main()
