import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import time

    from tqdm import tqdm
    import polars as pl
    import torch

    return pl, time, torch, tqdm


@app.cell
def _(torch):
    print("Torch:", torch.__version__)
    print("GPU available:", torch.cuda.is_available())
    print("CUDA:", torch.version.cuda)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        allocated_memory = torch.cuda.memory_allocated(0) / (1024**3)
        cached_memory = torch.cuda.memory_reserved(0) / (1024**3)
        available_memory = total_memory - allocated_memory
        print(f"Total GPU Memory: {total_memory:.2f} GiB")
        print(f"Allocated Memory: {allocated_memory:.2f} GiB")
        print(f"Cached/Reserved Memory: {cached_memory:.2f} GiB")
        print(f"Available Memory: {available_memory:.2f} GiB")
    return


@app.cell
def _(pl, time, torch, tqdm):
    N = 8192
    ITERS = 10
    WARMUP = 2

    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")

    def bench_matmul(device: str) -> float:
        dtype = torch.float16 if device == "cuda" else torch.float32
        a = torch.randn((N, N), device=device, dtype=dtype)
        b = torch.randn((N, N), device=device, dtype=dtype)

        # warmup
        for _ in tqdm(range(WARMUP)):
            _ = a @ b
        if device == "cuda":
            torch.cuda.synchronize()

        # timed
        t0 = time.perf_counter()
        for _ in tqdm(range(ITERS)):
            _ = a @ b
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        return (t1 - t0) / ITERS

    sec_per_iter = [bench_matmul(d) for d in devices]

    df = (pl
        .DataFrame({
            "device": devices, 
            "sec_per_iter": sec_per_iter})
        .with_columns(
            (pl.col("sec_per_iter").first() / pl.col("sec_per_iter")).alias("speedup_vs_cpu")
        )
    )

    df
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
