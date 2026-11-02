"""Three CPU workbenches with explicit inputs, validation and current-result CSVs.

No network, hidden data, learned weights or global RNG changes. Gradio may
cache figures; an export writes a new CSV to the system temporary directory.
An optional student diagnostic is called only by the loss callback.
"""
from __future__ import annotations
from collections.abc import Callable
from io import StringIO
from pathlib import Path
import re
import tempfile
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
import gradio as gr
from advml_review import (discrete_importance, gaussian_loss, paired_runs,
                          scalar_posterior, scalar_prediction)


def _numbers(text: str, name: str) -> list[float]:
    try:
        result = [float(s.strip()) for s in str(text).split(',')]
    except ValueError as exc:
        raise ValueError(f'{name}: enter comma-separated finite numbers, without empty entries.') from exc
    if not result or not np.isfinite(result).all():
        raise ValueError(f'{name}: enter comma-separated finite numbers.')
    return result


def _export(table: pd.DataFrame, prefix: str, enabled: bool) -> str | None:
    if not enabled:
        return None
    with tempfile.NamedTemporaryFile(prefix=prefix, suffix='.csv', delete=False) as f:
        path = Path(f.name)
    table.to_csv(path, index=False, encoding='utf-8-sig')
    return str(path)


def importance_view(p: str, q: str, draws: str, export: bool = False) -> tuple:
    """Compute fixed-draw weights and all estimates; optionally export this table."""
    r = discrete_importance(_numbers(p, 'Target p'), _numbers(q, 'Proposal q'), _numbers(draws, 'Draws'))
    fig = Figure(figsize=(7, 3), layout='constrained'); ax = fig.subplots()
    ax.bar(r['table'].draw_id, r['normalized_weights'], color='#2868a6')
    ax.set(xlabel='Draw ID (duplicates remain separate)', ylabel='Normalized weight',
           xticks=r['table'].draw_id, ylim=(0, 1), title='Per-draw weight concentration')
    summary = {k: r[k] for k in ('draw_count', 'numerator', 'weight_sum', 'exact', 'direct_is', 'snis', 'weight_ess')}
    return summary, r['table'], fig, _export(r['table'], 'importance-', export)


def loss_view(y: str, mean_a: str, mean_b: str, variance: float,
              diagnostic: Callable | None = None, export: bool = False) -> tuple:
    """Compare two means on the SAME observations and common noise variance.

    An unfinished diagnostic returning None is explicitly labeled; it is not
    replaced by a completed answer. Other core results remain independent.
    """
    observed, ma, mb = _numbers(y, 'Observed y'), _numbers(mean_a, 'Mean A'), _numbers(mean_b, 'Mean B')
    a, b = gaussian_loss(observed, ma, variance), gaussian_loss(observed, mb, variance)
    table = pd.concat([a['table'].assign(model='A'), b['table'].assign(model='B')], ignore_index=True)
    custom = diagnostic(np.asarray(observed), np.asarray(ma)) if diagnostic is not None else None
    if custom is None:
        custom = 'Your E6 diagnostic is not implemented yet; core calculations are complete.'
    else:
        if not np.isscalar(custom) or not np.isfinite(float(custom)):
            raise ValueError('The custom diagnostic must return a finite scalar or None.')
        custom = float(custom)
    summary = {'A': {k: a[k] for k in ('sse', 'mse', 'sum_nll')},
               'B': {k: b[k] for k in ('sse', 'mse', 'sum_nll')},
               'NLL_A_minus_B': a['sum_nll']-b['sum_nll'],
               'common_variance': float(variance), 'custom_diagnostic_A': custom}
    fig = Figure(figsize=(7, 3), layout='constrained'); ax = fig.subplots()
    for result, label, shift, color in ((a, 'A', -.07, '#2868a6'), (b, 'B', .07, '#d57226')):
        ax.scatter(result['table'].observation_id+shift, result['table'].residual, label=label, color=color)
    ax.axhline(0, color='gray', linewidth=.8)
    ax.set(xlabel='Observation ID', ylabel='Residual y - mean', xticks=a['table'].observation_id)
    ax.legend()
    return summary, table, fig, _export(table, 'gaussian-loss-', export)


def _records(text: str) -> pd.DataFrame:
    frame = pd.read_csv(StringIO(str(text).strip()), dtype={'run_id': str})
    if set(frame.columns) != {'run_id', 'error'}:
        raise ValueError('Records must have exactly the CSV columns run_id,error.')
    if not all(isinstance(k, str) and re.fullmatch(r'[A-Za-z0-9_-]+', k) for k in frame.run_id):
        raise ValueError('Use simple run IDs: letters, digits, underscore or hyphen.')
    return frame


def paired_view(a: str, b: str, export: bool = False) -> tuple:
    """Align independent CSV records by ID and report every signed difference."""
    r = paired_runs(_records(a), _records(b))
    fig = Figure(figsize=(6, 3), layout='constrained'); ax = fig.subplots()
    for row in r['table'].itertuples():
        ax.plot([0, 1], [row.error_A, row.error_B], marker='o', label=row.run_id)
    ax.set(xticks=[0, 1], xticklabels=['Model A', 'Model B'], ylabel='Recorded error (lower is better)')
    ax.legend(title='Matched run ID')
    return ({k: r[k] for k in ('run_count', 'mean_difference', 'a_wins', 'b_wins', 'ties')},
            r['table'], fig, _export(r['table'], 'paired-runs-', export))


def bayes_view(x: str, y: str, prior_variance: float, noise_sd: float,
               x_star: float, draws: int, seed: int, show_response: bool = True,
               export: bool = False) -> tuple:
    """Exact one-slope posterior, exact prediction and separately labeled iid MC.

    Monte Carlo draw count changes numerical estimation, not the exact posterior
    or its predictive variance. Each function draw shares one weight over x.
    """
    xx, yy = np.asarray(_numbers(x, 'Inputs x')), np.asarray(_numbers(y, 'Observed y'))
    for value, label in ((draws, 'Draw count'), (seed, 'Seed')):
        if not np.isfinite(value) or int(value) != value:
            raise ValueError(f'{label} must be an integer.')
    if not 2 <= int(draws) <= 100000 or not 0 <= int(seed) < 2**32:
        raise ValueError('Use draw count 2..100000 and a seed 0..2**32-1.')
    posterior = scalar_posterior(xx, yy, prior_variance=prior_variance, noise_sd=noise_sd)
    pred = scalar_prediction(posterior, x_star)
    ws = np.random.default_rng(int(seed)).normal(posterior['mean'], np.sqrt(posterior['variance']), int(draws))
    grid = np.linspace(min(-3., xx.min(), x_star)-.5, max(3., xx.max(), x_star)+.5, 101)
    mean, var_f = posterior['mean']*grid, posterior['variance']*grid**2
    var_y = var_f+posterior['noise_variance']
    table = pd.DataFrame({'x': grid, 'mean_f': mean, 'variance_f': var_f,
                          'variance_new_y': var_y, 'posterior_weight_mean': posterior['mean'],
                          'posterior_weight_variance': posterior['variance'], 'noise_sd': float(noise_sd)})
    fig = Figure(figsize=(7, 4), layout='constrained'); ax = fig.subplots()
    if show_response:
        ax.fill_between(grid, mean-1.96*np.sqrt(var_y), mean+1.96*np.sqrt(var_y),
                        color='#a1a8b0', alpha=.35, label='New response: mean +/- 1.96 SD')
    ax.fill_between(grid, mean-1.96*np.sqrt(var_f), mean+1.96*np.sqrt(var_f),
                    color='#4a91c7', alpha=.35, label='Latent f: mean +/- 1.96 SD')
    for w in ws[:8]:
        ax.plot(grid, w*grid, color='#2868a6', alpha=.22, linewidth=.8)
    ax.plot(grid, mean, color='#173b60', label='Exact posterior mean')
    ax.scatter(xx, yy, color='#d57226', label='Training observations', zorder=3)
    ax.set(xlabel='Input x', ylabel='Function / response'); ax.legend(fontsize=8)
    summary = {'exact_posterior': posterior, 'exact_prediction': pred,
               'MC_estimated_mean_f': float(x_star*ws.mean()),
               'MC_standard_error_of_mean_f': float(abs(x_star)*np.sqrt(posterior['variance']/int(draws))),
               'MC_draw_count': int(draws), 'seed': int(seed)}
    return summary, table, fig, _export(table, 'bayesian-prediction-', export)


def _safe(function):
    def call(*args):
        try:
            return function(*args)
        except (ValueError, TypeError, OverflowError, pd.errors.ParserError) as exc:
            raise gr.Error(str(exc)) from exc
    return call


def build_app(diagnostic: Callable | None = None) -> gr.Blocks:
    """Return the three-tab workbench; launch separately in the notebook.

    The optional diagnostic receives y,mean_A and returns a scalar or None.
    CSV buttons recompute and update visible results from current controls.
    This function builds UI only; it starts no server or public sharing.
    """
    with gr.Blocks(analytics_enabled=False) as demo:
        gr.Markdown('# Inference to Prediction\nSynthetic teaching cases. Predict first, calculate, then explain the difference.')
        with gr.Tab('Importance weights'):
            with gr.Row():
                p = gr.Textbox(value='0.2,0.3,0.5', label='Normalized target p (states 0,1,...)')
                q = gr.Textbox(value='0.5,0.25,0.25', label='Proposal q (positive masses)')
                z = gr.Textbox(value='0,1,1,2', label='Fixed draws (keep repetitions)')
            with gr.Row(): calculate = gr.Button('Calculate weights'); export = gr.Button('Compute and export weights CSV')
            summary, table, plot, file = gr.JSON(label='Estimators and weight ESS'), gr.Dataframe(interactive=False), gr.Plot(), gr.File(label='Weights CSV', interactive=False)
            outputs = [summary, table, plot, file]
            calculate.click(_safe(importance_view), [p,q,z], outputs)
            export.click(_safe(lambda p,q,z: importance_view(p,q,z,True)), [p,q,z], outputs)
        with gr.Tab('Loss and matched runs'):
            gr.Markdown('## Gaussian loss\nThe same observations and one common variance are used for both candidates. NLL is a sum with its normalization constant.')
            with gr.Row():
                observed = gr.Textbox(value='1,2', label='Observed y')
                ma = gr.Textbox(value='1.5,1.5', label='Candidate mean A')
                mb = gr.Textbox(value='1,2', label='Candidate mean B')
                variance = gr.Number(value=1, label='Common noise VARIANCE')
            with gr.Row(): calculate = gr.Button('Compare Gaussian losses'); export = gr.Button('Compute and export loss CSV')
            outputs = [gr.JSON(label='Losses and custom diagnostic'), gr.Dataframe(interactive=False), gr.Plot(), gr.File(label='Residuals CSV',interactive=False)]
            calculate.click(_safe(lambda y,a,b,v: loss_view(y,a,b,v,diagnostic)), [observed,ma,mb,variance], outputs)
            export.click(_safe(lambda y,a,b,v: loss_view(y,a,b,v,diagnostic,True)), [observed,ma,mb,variance], outputs)
            gr.Markdown('## Separate run records\nThese errors are a separate experiment table. Match IDs before subtracting A - B. Positive differences favor B; ties are not wins.')
            with gr.Row():
                a = gr.Textbox(value='run_id,error\nr1,0.2\nr2,0.6\nr3,0.4',label='Model A run records',lines=4)
                b = gr.Textbox(value='run_id,error\nr3,0.2\nr1,0.6\nr2,0.4',label='Model B run records (shuffled)',lines=4)
            with gr.Row(): calculate = gr.Button('Match runs by ID'); export = gr.Button('Compute and export matched CSV')
            outputs = [gr.JSON(label='Paired summary'),gr.Dataframe(interactive=False),gr.Plot(),gr.File(label='Matched runs CSV',interactive=False)]
            calculate.click(_safe(paired_view), [a,b], outputs)
            export.click(_safe(lambda a,b: paired_view(a,b,True)), [a,b], outputs)
        with gr.Tab('Bayesian prediction'):
            gr.Markdown('One slope, no intercept; zero-mean Gaussian weight prior. Noise is entered as SD. A new response adds one independent noise variance. Increasing Monte Carlo draws leaves the exact posterior unchanged.')
            with gr.Row():
                x = gr.Textbox(value='1,2',label='Training inputs x')
                y = gr.Textbox(value='1,2',label='Training responses y')
                tau = gr.Number(value=1,label='Prior weight VARIANCE')
                noise = gr.Number(value=1,label='Noise STANDARD DEVIATION')
            with gr.Row():
                star = gr.Slider(-4,4,value=3,step=.5,label='Prediction input x*')
                count = gr.Slider(10,1000,value=100,step=10,label='IID posterior draws (numerical estimate)')
                seed = gr.Number(value=7,precision=0,label='Monte Carlo seed')
            response = gr.Checkbox(value=True,label='Show new-response interval')
            with gr.Row(): calculate = gr.Button('Compute posterior and prediction'); export = gr.Button('Compute and export prediction CSV')
            inputs = [x,y,tau,noise,star,count,seed,response]
            outputs = [gr.JSON(label='Exact results and separate Monte Carlo estimate'),gr.Dataframe(interactive=False),gr.Plot(),gr.File(label='Prediction CSV',interactive=False)]
            calculate.click(_safe(bayes_view),inputs,outputs)
            export.click(_safe(lambda *args: bayes_view(*args,export=True)),inputs,outputs)
        gr.Markdown('CSV files use the current controls. Shared Colab apps remain available only while their runtime is active.')
    return demo
