import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell
def _():
    import torch
    return (torch,)


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
def _():
    return


if __name__ == "__main__":
    app.run()
