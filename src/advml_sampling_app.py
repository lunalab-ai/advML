"""Sampling Explorer: shared numerical experiments, plots, and Gradio callbacks."""
from __future__ import annotations
from pathlib import Path
import tempfile
import time
import numpy as np
from advml_sampling import (make_target, importance_sampling, random_walk_mh,
                            gaussian_gibbs, chain_diagnostics, snis_summary,
                            autocorrelation, _count)
from advml_smc import simulate_tracking, kalman_filter, particle_filter


def static_experiment(method: str = 'MH', kind: str = 'gaussian', rho: float = .8,
                      draws: int = 1000, proposal_scale: float = 1.5,
                      proposal_shift: float = 0., mh_scale: float = .5,
                      initial_center: float = 0., seed: int = 7) -> dict:
    """Run a fixed-target experiment and return real estimates plus raw draws.

    method: 'IID MC', 'SNIS', 'MH', or 'Gibbs'. kind: gaussian/bimodal.
    draws>=100 counts total proposals for IID/SNIS, POST-warmup draws PER CHAIN
    for MH/Gibbs (four chains, 300 warmup transitions each). proposal_scale is
    IS SD multiplier, proposal_shift shifts IS coordinate 0, mh_scale is MH
    proposal SD. initial_center offsets coordinate 0 of dispersed starts.
    seed is local; no learned state. Return target, samples (N,2), weights (N,),
    chains (4,draws,2) or None, and scalar/JSON-ready metrics. Evaluation counts
    include warmup; conditional draws and exact iid oracle calls are different
    operations. Wall time is measured locally and is not a universal ranking.
    """
    import arviz  # Include lazy diagnostic-library startup before timing.
    n = _count(draws, 100)
    if not np.isfinite(initial_center):
        raise ValueError('initial_center must be finite')
    target = make_target(kind, rho)
    start_time = time.perf_counter()
    chains, acceptance, weight_diag = None, None, None
    counts = {'target_evaluations': 0, 'proposal_evaluations': 0,
              'conditional_draws': 0, 'iid_oracle_draws': 0}
    second_metrics = {}
    if method == 'IID MC':
        z = target.sample(n, seed)
        w = np.full(n, 1./n)
        counts['iid_oracle_draws'] = n
        mcse = float(z[:,0].std(ddof=1)/np.sqrt(n))
        second_mcse = float((z[:,0]**2).std(ddof=1)/np.sqrt(n))
        note = 'IID oracle baseline; direct target sampling is not generally available.'
    elif method == 'SNIS':
        result = importance_sampling(target, n, proposal_scale, proposal_shift, seed)
        z, w = result['samples'], result['weights']
        weight_diag = result['weight_ess']
        counts.update({k: result[k] for k in ('target_evaluations','proposal_evaluations')})
        mcse = snis_summary(z[:,0],w)['mcse']
        second_mcse = snis_summary(z[:,0]**2,w)['mcse']
        note = 'Weight ESS cannot detect missed regions; SNIS and its MCSE are finite-run approximations.'
    elif method in ('MH','Gibbs'):
        if method == 'Gibbs' and kind != 'gaussian':
            raise ValueError('This Gibbs implementation needs the W4 Gaussian target.')
        offsets = np.array([[-3,-3],[-1,1],[1,-1],[3,3.]])
        initials = target.mean + offsets*np.sqrt(np.diag(target.covariance)) + [initial_center,0.]
        traces, accepted = [], []
        for j, initial in enumerate(initials):
            if method == 'MH':
                run = random_walk_mh(target.log_density, initial, n+300, mh_scale, seed+101*j)
                counts['target_evaluations'] += run['target_evaluations']
                accepted.append(run['accepted'][300:].mean())
            else:
                run = gaussian_gibbs(target.model, initial, n+300, seed+101*j)
                counts['conditional_draws'] += run['conditional_draws']
            traces.append(run['samples'][300:])
        chains = np.stack(traces)
        z = chains.reshape(-1,2)
        w = np.full(len(z),1./len(z))
        if accepted:
            acceptance = float(np.mean(accepted))
        first = chain_diagnostics(chains[:,:,0])
        second = chain_diagnostics(chains[:,:,0]**2)
        second_metrics = {**first, 'second_moment_rhat': second['rhat_rank'],
                          'second_moment_ess_mean': second['ess_mean']}
        mcse, second_mcse = first['mcse_mean'], second['mcse_mean']
        note = first['diagnostic_warning']
    else:
        raise ValueError('unknown method')
    mean = w @ z
    delta = z-mean
    covariance = np.einsum('n,ni,nj->ij', w, delta, delta)
    exact_second = float(target.covariance[0,0] + target.mean[0]**2)
    metrics = {**second_metrics, **counts, 'method': method, 'target': kind,
               'rho': float(rho), 'draws_control': n, 'retained_draws': len(z),
               'proposal_scale': float(proposal_scale), 'proposal_shift': float(proposal_shift),
               'mh_scale': float(mh_scale), 'initial_center': float(initial_center), 'seed': int(seed),
               'estimate_z0': float(mean[0]), 'exact_z0': float(target.mean[0]),
               'mean_vector_error': float(np.linalg.norm(mean-target.mean)),
               'covariance_error': float(np.linalg.norm(covariance-target.covariance)),
               'estimate_z0_squared': float(w @ z[:,0]**2), 'exact_z0_squared': exact_second,
               'mcse_z0': mcse, 'mcse_z0_squared': second_mcse,
               'weight_ess': weight_diag, 'acceptance_post_warmup': acceptance,
               'seconds': time.perf_counter()-start_time, 'interpretation': note}
    return {'target': target, 'samples': z, 'weights': w, 'chains': chains, 'metrics': metrics}


def _csv(frame, prefix):
    folder = Path(tempfile.mkdtemp(prefix='advml-w5-'))
    path = folder / (prefix+'.csv')
    frame.to_csv(path,index=False)
    return str(path)


def static_view(method='MH', kind='gaussian', rho=.8, draws=1000,
                proposal_scale=1.5, proposal_shift=0., mh_scale=.5,
                initial_center=0., seed=7):
    """Gradio callback: same controls as static_experiment -> Figure, dict, CSV path.

    Computes a NEW seeded experiment each time; writes one summary CSV in a
    fresh temporary folder for download. Plots include exact target contours,
    estimator/trace, weights/ACF and the two distinct expectation estimates.
    No server is launched. Returned Figure belongs to the caller.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    run = static_experiment(method,kind,rho,draws,proposal_scale,proposal_shift,mh_scale,initial_center,int(seed))
    target, z, w, chains, metrics = (run[k] for k in ('target','samples','weights','chains','metrics'))
    fig, axes = plt.subplots(2,2,figsize=(11,7),layout='constrained')
    xlim = (-5,5) if kind=='bimodal' else (target.mean[0]-3,target.mean[0]+3)
    ylim = (-2,2) if kind=='bimodal' else (target.mean[1]-3,target.mean[1]+3)
    x,y=np.meshgrid(np.linspace(*xlim,110),np.linspace(*ylim,100))
    density=np.exp(target.log_density(np.stack([x,y],axis=-1)))
    axes[0,0].contour(x,y,density,levels=7,cmap='Blues')
    stride=max(1,len(z)//700)
    sizes=12+1000*w[::stride] if method=='SNIS' else 10
    axes[0,0].scatter(z[::stride,0],z[::stride,1],s=sizes,alpha=.3,color='#e88a32')
    axes[0,0].set(xlim=xlim,ylim=ylim,xlabel='z0',ylabel='z1',title='Exact target + sampled positions')
    if chains is not None:
        for chain in chains:
            axes[0,1].plot(chain[:,0],lw=.6,alpha=.75)
        axes[0,1].axhline(target.mean[0],color='black',ls='--')
        axes[0,1].set(title='Four post-warmup traces',xlabel='MCMC iteration',ylabel='z0')
        for chain in chains:
            axes[1,0].plot(autocorrelation(chain[:,0],60),alpha=.7)
        axes[1,0].set(title='ACF is not weight ESS',xlabel='Lag',ylabel='Sample autocorrelation')
    else:
        estimate=np.cumsum(w*z[:,0])/np.cumsum(w)
        axes[0,1].plot(np.arange(1,len(z)+1),estimate,color='#126b83')
        axes[0,1].axhline(target.mean[0],color='black',ls='--',label='exact')
        axes[0,1].set(title='Running estimate of E[z0]',xlabel='Proposal/draw index',ylabel='Estimate')
        axes[1,0].plot(np.sort(w)[::-1],color='#e88a32')
        axes[1,0].set(title='Sorted normalized weights',xlabel='Rank',ylabel='Weight')
    axes[1,1].axis('off')
    text=(f"Method: {method}\nRetained draws: {len(z)}\n"
          f"E[z0]: {metrics['estimate_z0']:.4f}   exact: {metrics['exact_z0']:.4f}\n"
          f"E[z0²]: {metrics['estimate_z0_squared']:.4f}   exact: {metrics['exact_z0_squared']:.4f}\n"
          f"Mean-vector error: {metrics['mean_vector_error']:.4f}\n"
          f"Covariance error: {metrics['covariance_error']:.4f}\n\n"
          "Read the diagnostics below.\nA good-looking histogram is insufficient.")
    axes[1,1].text(.03,.94,text,va='top',fontsize=11,linespacing=1.7)
    return fig,metrics,_csv(pd.DataFrame([metrics]),'static-experiment')


def filter_view(particles=500, observation_sd=.5, threshold=.5, seed=7):
    """Gradio callback -> Figure, metric dict, timewise CSV path for tracking.

    particles>=2, observation_sd>0, resampling threshold in [0,1], integer seed
    for PARTICLES. Data seed stays 20260929. Changing observation_sd rescales
    the same simulated observation noise and changes the assumed likelihood;
    changing particles/threshold/particle seed leaves data identical.
    CSV includes truth, observations, Kalman/PF estimates, ESS and ancestor count.
    Shows filtering moments before resampling; no future observations are used.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    n = _count(particles,2)
    data=simulate_tracking(observation_sd=float(observation_sd))
    y=data['observations']
    reference=kalman_filter(y,observation_sd=float(observation_sd))
    start=time.perf_counter()
    run=particle_filter(y,n,observation_sd=float(observation_sd),threshold=float(threshold),seed=int(seed))
    duration=time.perf_counter()-start
    t=np.arange(len(y))
    fig,axes=plt.subplots(3,1,figsize=(11,8),layout='constrained',sharex=True)
    axes[0].scatter(t,y,s=16,color='#a0a8b5',label='observations')
    axes[0].plot(t,data['truth'],color='#30343b',ls=':',label='latent truth')
    axes[0].plot(t,reference['mean'],color='#166c9b',label='exact Kalman mean')
    band=1.96*np.sqrt(reference['variance'])
    axes[0].fill_between(t,reference['mean']-band,reference['mean']+band,color='#166c9b',alpha=.12,label='pointwise 95% filtering interval')
    axes[0].plot(t,run['mean'],color='#e47b26',label='particle mean')
    axes[0].set(ylabel='State',title='Same observation sequence: exact and particle filtering')
    axes[0].legend(fontsize=8,ncol=2,loc='upper right')
    axes[1].plot(t,run['ess_before'],label='before resampling',color='#e47b26')
    axes[1].plot(t,run['ess_after'],label='after resampling',color='#177e75',ls='--')
    axes[1].axhline(float(threshold)*n,color='gray',ls=':',label='resampling trigger')
    axes[1].set(ylabel='Weight ESS',ylim=(0,n*1.05)); axes[1].legend(fontsize=8,ncol=3)
    axes[2].step(t,run['unique_roots_after'],where='mid',color='#7356a2')
    axes[2].set(ylabel='Distinct initial ancestors',xlabel='Observation time t',ylim=(0,n*1.05))
    metrics={'particles':n,'observation_sd':float(observation_sd),'threshold':float(threshold),'particle_seed':int(seed),
             'data_seed':20260929,'rmse_vs_kalman_mean':float(np.sqrt(np.mean((run['mean']-reference['mean'])**2))),
             'rmse_vs_latent_truth':float(np.sqrt(np.mean((run['mean']-data['truth'])**2))),
             'variance_rmse_vs_kalman':float(np.sqrt(np.mean((run['variance']-reference['variance'])**2))),
             'resampling_events':int(run['resampled'].sum()),'final_unique_ancestors':int(run['unique_roots_after'][-1]),
             'min_weight_ess':float(run['ess_before'].min()),'seconds':duration,
             'interpretation':'Weight ESS resets after copying; information and ancestry do not reset.'}
    frame=pd.DataFrame({'time':t,'truth':data['truth'],'observed':y,'kalman_mean':reference['mean'],
                        'kalman_variance':reference['variance'],'particle_mean':run['mean'],'particle_variance':run['variance'],
                        'ess_before':run['ess_before'],'ess_after':run['ess_after'],
                        'unique_ancestors':run['unique_roots_after'],'resampled':run['resampled']})
    for key in ('particles','observation_sd','threshold','particle_seed','data_seed'):
        frame[key]=metrics[key]
    return fig,metrics,_csv(frame,'tracking-experiment')


def build_sampling_app():
    """Build (do not launch) a two-tab Gradio Blocks Sampling Explorer.

    Return a fresh Blocks object. Buttons map visible controls in the same
    order as static_view/filter_view arguments to plot/JSON/CSV outputs.
    CPU-only, no account keys, no user uploads. Launch in Colab with share=True;
    that temporary URL depends on the active runtime. Keep the notebook link.
    """
    import gradio as gr
    with gr.Blocks(title='Advanced ML - Sampling Explorer',analytics_enabled=False) as app:
        gr.Markdown('# Sampling Explorer\nPredict a change, change ONE control, run, inspect, and export the evidence.')
        with gr.Tab('1. Fixed target: MC / IS / MCMC'):
            gr.Markdown('IID MC is an exact benchmark oracle. For MH/Gibbs, draws are **per chain**, after 300 warmup steps; four dispersed chains run. For IID/SNIS they are total draws. Counts and time are reported separately.')
            with gr.Row():
                method=gr.Dropdown(['IID MC','SNIS','MH','Gibbs'],value='MH',label='Method')
                kind=gr.Dropdown(['gaussian','bimodal'],value='gaussian',label='Target (Gibbs: Gaussian only)')
                rho=gr.Slider(-.98,.98,value=.8,step=.02,label='W4 observation-noise correlation')
            with gr.Row():
                draws=gr.Slider(100,3000,value=1000,step=100,label='Draws (see counting rule above)')
                scale=gr.Slider(.2,4,value=1.5,step=.1,label='IS proposal SD multiplier')
                shift=gr.Slider(-3,3,value=0,step=.25,label='IS proposal mean shift in z0')
            with gr.Row():
                mh=gr.Slider(.02,4,value=.5,step=.02,label='MH proposal SD')
                initial=gr.Slider(-5,5,value=0,step=.5,label='MCMC initial center shift in z0')
                seed=gr.Number(value=7,precision=0,label='Random seed')
            button=gr.Button('Run fixed-target experiment',variant='primary')
            plot=gr.Plot(label='Geometry, traces and uncertainty')
            stats=gr.JSON(label='Diagnostics: interpret together')
            csv=gr.File(label='Download this experiment record')
            button.click(static_view,[method,kind,rho,draws,scale,shift,mh,initial,seed],[plot,stats,csv],api_name='static_experiment')
        with gr.Tab('2. Sequential target: particle filter'):
            gr.Markdown('Threshold 0 gives SIS without resampling. Data stays fixed when N, threshold or particle seed changes. Observation SD regenerates the noise with the same data seed and updates the likelihood.')
            with gr.Row():
                count=gr.Slider(50,2000,value=500,step=50,label='Particle count')
                noise=gr.Slider(.1,1.5,value=.5,step=.1,label='Observation SD (regenerates observations)')
                threshold=gr.Slider(0,1,value=.5,step=.1,label='Resample if ESS < fraction × N')
                pfseed=gr.Number(value=7,precision=0,label='Particle seed')
            run=gr.Button('Run tracking experiment',variant='primary')
            pfplot=gr.Plot(label='Filtering, weight ESS and ancestry')
            pfstats=gr.JSON(label='Numerical error versus statistical uncertainty')
            pfcsv=gr.File(label='Download the timewise experiment record')
            run.click(filter_view,[count,noise,threshold,pfseed],[pfplot,pfstats,pfcsv],api_name='tracking_experiment')
        gr.Markdown('**Explain the result:** name the target, estimator, changed setting, diagnostic limitation, and evidence. A near-one R-hat or high weight ESS alone cannot establish accuracy.')
    return app
