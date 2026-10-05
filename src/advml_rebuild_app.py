"""Inference Workbench for W6; CPU, no credentials, no uploaded data."""
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from advml_rebuild import (SensorModel, exact_posterior, exact_filter, mean_field,
                           sample_paths, particle_filter, distribution_summary)


def static_experiment(method='VI', stay=.8, n=400, seed=19, initial=.5,
                      proposal_one=.5, extra=None):
    """Return Figure, JSON-safe metrics dict, and CSV path for a fixed target.

    method: Exact/VI/MC/IS/MH/Gibbs. n: samples or max VI sweeps. Model has
    y=(1,0,1), accuracy=.8. extra is an optional callable(metrics)->dict;
    use it to add a student diagnostic without editing the model or sampler.
    Temporary CSV contains target settings, seed, budget and exact reference.
    """
    model = SensorModel(stay=float(stay)); ref = exact_posterior(model)
    details = {}
    if method == 'Exact':
        approximate = ref['probabilities']
    elif method == 'VI':
        result = mean_field(model, initial=float(initial), sweeps=int(n))
        approximate = result['probabilities']
        details = dict(vi_sweeps=result['sweeps'], converged=result['converged'], reverse_kl=result['reverse_kl'])
    else:
        result = sample_paths(model, method, int(n), int(seed), float(proposal_one))
        approximate = result['probabilities']
        details = dict(weight_ess=result['weight_ess'], move_fraction=result['move_fraction'])
    exact = distribution_summary(ref['states'], ref['probabilities'])
    estimate = distribution_summary(ref['states'], approximate)
    metrics = dict(method=method, stay=float(stay), accuracy=.8, observations='1,0,1',
                   budget=int(n), seed=int(seed), initial=float(initial), proposal_one=float(proposal_one),
                   exact_probability=exact['probability_last_one'], estimate_probability=estimate['probability_last_one'],
                   absolute_error=abs(exact['probability_last_one']-estimate['probability_last_one']),
                   total_variation=float(.5*np.abs(approximate-ref['probabilities']).sum()),
                   exact_covariance=exact['covariance_first_last'], estimate_covariance=estimate['covariance_first_last'],
                   posterior_sd=exact['posterior_sd_last'], **details)
    if extra is not None:
        added = extra(dict(metrics))
        if not isinstance(added, dict) or set(added).intersection(metrics):
            raise ValueError('extra must return a dict with new diagnostic names.')
        metrics.update(added)
    fig = Figure(figsize=(8, 3.4), layout='constrained'); ax = fig.subplots()
    x = np.arange(8)
    ax.bar(x-.18, ref['probabilities'], .36, color='#2264a5', label='Exact posterior')
    ax.bar(x+.18, approximate, .36, color='#d07827', label=method)
    ax.set(xticks=x, xticklabels=[''.join(map(str, row)) for row in ref['states']],
           xlabel='Hidden path z1 z2 z3', ylabel='Probability', ylim=(0, max(.6, approximate.max()+.05)))
    ax.legend(); ax.set_title('Same observations; compare the complete distribution')
    with tempfile.NamedTemporaryFile(prefix='w6-static-', suffix='.csv', delete=False) as handle:
        filename = handle.name
    pd.DataFrame([metrics]).to_csv(filename, index=False)
    return fig, metrics, filename


def sequential_experiment(n=400, seed=19, stay=.8):
    """Return Figure, per-time DataFrame, CSV for SMC vs exact FILTERING.

    n positive particle count, seed integer, stay in (0,1). No smoother
    reference is used. CSV has one row per observed time, not per MCMC step.
    """
    model = SensorModel(stay=float(stay)); exact = exact_filter(model)
    result = particle_filter(model, int(n), int(seed))
    frame = pd.DataFrame(dict(time=[1, 2, 3], observation=model.observations,
        exact_filter=exact['prob_one'], particle_filter=result['prob_one'], weight_ess=result['weight_ess'],
        absolute_error=np.abs(result['prob_one']-exact['prob_one']), particles=int(n), seed=int(seed), stay=float(stay)))
    fig = Figure(figsize=(8, 3.4), layout='constrained'); ax = fig.subplots()
    ax.plot(frame.time, frame.exact_filter, 'o-', label='Exact filter', color='#2264a5')
    ax.plot(frame.time, frame.particle_filter, 's--', label='Particles', color='#d07827')
    ax.set(xticks=[1, 2, 3], ylim=(0, 1), xlabel='Observation time (prefix only)', ylabel='P(current state = 1)')
    ax.legend(); ax.set_title('The target changes when an observation arrives')
    with tempfile.NamedTemporaryFile(prefix='w6-sequential-', suffix='.csv', delete=False) as handle:
        filename = handle.name
    frame.to_csv(filename, index=False)
    return fig, frame, filename


def create_app(extra=None):
    """Build (do not launch) a gradio.Blocks workbench; optional extra callback.

    Use demo.launch(share=True) in Colab only while you need a temporary URL.
    create_app itself starts no server. Both experiments are independently runnable.
    """
    import gradio as gr
    with gr.Blocks(title='W6 Inference Workbench') as demo:
        gr.Markdown('# Inference Workbench\nPredict → run → compare → explain. Fixed observations: **1, 0, 1**; sensor accuracy: **0.8**.')
        with gr.Tab('Fixed posterior'):
            gr.Markdown('Compare one target. VI can retain family error; MC uses an exact oracle. MH/Gibbs draws are dependent. IS weight ESS is not chain ESS.')
            method = gr.Dropdown(['Exact', 'VI', 'MC', 'IS', 'MH', 'Gibbs'], value='VI', label='Method')
            stay = gr.Slider(.5, .98, value=.8, step=.01, label='Probability the hidden state stays unchanged')
            n = gr.Slider(10, 2000, value=400, step=10, label='Samples / maximum VI sweeps')
            seed = gr.Number(value=19, precision=0, label='Random seed')
            with gr.Accordion('VI / IS settings', open=False):
                initial = gr.Slider(.05, .95, value=.5, step=.05, label='VI initial probability')
                proposal = gr.Slider(.05, .95, value=.5, step=.05, label='IS proposal probability of one')
            run = gr.Button('Compare with exact posterior', variant='primary')
            plot = gr.Plot(label='Eight-path distribution'); metrics = gr.JSON(label='Experiment record')
            download = gr.File(label='Download comparison CSV')
            def callback(method, stay, n, seed, initial, proposal):
                return static_experiment(method, stay, n, seed, initial, proposal, extra)
            run.click(callback, [method, stay, n, seed, initial, proposal], [plot, metrics, download], api_name='compare')
        with gr.Tab('Sequential observations'):
            gr.Markdown('At time t, use only observations 1 through t. Compare filtering with filtering.')
            particles = gr.Slider(10, 2000, value=400, step=10, label='Number of particles')
            pseed = gr.Number(value=19, precision=0, label='Particle seed')
            pstay = gr.Slider(.5, .98, value=.8, step=.01, label='Transition stay probability')
            update = gr.Button('Reveal observations and filter')
            pplot = gr.Plot(); table = gr.Dataframe(interactive=False); csv = gr.File(label='Download filtering CSV')
            update.click(sequential_experiment, [particles, pseed, pstay], [pplot, table, csv], api_name='filter')
        gr.Markdown('Change one setting at a time. Repeat seeds before making an accuracy claim. A Gradio share link stops when its runtime stops.')
    return demo


if __name__ == '__main__':
    create_app().launch(server_name='127.0.0.1', server_port=7876)
