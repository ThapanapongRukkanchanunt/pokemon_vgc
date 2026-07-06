import argparse


def main():
    parser = argparse.ArgumentParser(description="Check PyTorch GPU availability.")
    parser.add_argument("--require-cuda", action="store_true", help="Fail if CUDA is not available.")
    args = parser.parse_args()

    try:
        import torch
    except ImportError as error:
        raise SystemExit(f"FAIL PyTorch is not installed: {error}")

    print(f"torch_version={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"cuda_device_count={torch.cuda.device_count()}")
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            print(f"cuda_device_{index}={torch.cuda.get_device_name(index)}")
        device = torch.device("cuda")
        a = torch.randn(512, 512, device=device)
        b = torch.randn(512, 512, device=device)
        c = a @ b
        torch.cuda.synchronize()
        print(f"cuda_matmul_mean={float(c.mean().item()):.6f}")
    elif args.require_cuda:
        raise SystemExit("FAIL CUDA is required but not available")
    print("PASS")


if __name__ == "__main__":
    main()
