"""Inspect the fitted W12 decoder without retraining or public tunnels."""
from pathlib import Path
import tempfile

import numpy as np

from advml_w12 import decoded_means


def decoder_view(model, metrics, z1, z2, seed=941):
    import matplotlib.pyplot as plt

    image = decoded_means(model, [[z1, z2]])[0]
    sample = (np.random.default_rng(seed).random(image.shape) < image).astype(float)
    fig, axes = plt.subplots(1, 2, figsize=(5.8, 2.6), layout="constrained")
    for ax, arr, title in zip(axes, (image, sample), ("Decoder mean", "Bernoulli observation")):
        ax.imshow(arr, vmin=0, vmax=1, cmap="Blues")
        ax.set_title(title)
        ax.set_axis_off()
    row = dict(z1=float(z1), z2=float(z2), mean_pixel_probability=float(image.mean()),
               sampled_fraction=float(sample.mean()), **metrics)
    return fig, row


def export_view(model, metrics, z1, z2):
    import csv
    import matplotlib.pyplot as plt

    fig, row = decoder_view(model, metrics, z1, z2)
    plt.close(fig)
    with tempfile.NamedTemporaryFile(prefix="w12-decoder-", suffix=".csv", mode="w",
                                     encoding="utf-8", newline="", delete=False) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
        return str(Path(stream.name))


def build_decoder_app(model, metrics):
    import gradio as gr
    import pandas as pd

    def show(z1, z2):
        figure, row = decoder_view(model, metrics, z1, z2)
        return figure, pd.DataFrame([row])

    with gr.Blocks(title="W12 Latent Decoder", analytics_enabled=False) as app:
        gr.Markdown("## W12 · Latent Decoder\nMove one coordinate and compare a probability map with a sampled image.")
        with gr.Row():
            z1 = gr.Slider(-3, 3, value=0, step=.1, label="z1")
            z2 = gr.Slider(-3, 3, value=0, step=.1, label="z2")
        image = gr.Plot(label="Decoded probability and observation")
        table = gr.Dataframe(interactive=False, label="Current coordinates and held-out metrics")
        export = gr.Button("Export current row")
        file = gr.File(label="CSV")
        app.load(show, [z1, z2], [image, table], api_name="initial")
        z1.release(show, [z1, z2], [image, table], api_name="decode")
        z2.release(show, [z1, z2], [image, table], api_name=False)
        export.click(lambda a, b: export_view(model, metrics, a, b), [z1, z2], file, api_name="export")
    return app
