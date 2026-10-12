"""Interactive W7 prediction workbench; runs locally or at the end of Lab B."""
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import expit
from advml_predictive import (make_data,truth,GaussianProcess,BayesianLinear,TinyMLP,
    evaluate_predictions,gaussian_nll,bernoulli_nll,poisson_nll)

MODELS=['GP (RBF)','Bayesian linear','Neural-linear (last layer)','MLP (noise only)']


def fit_prediction(model, data, length=1.):
    """Fit on training observations only; return predictive dict and honest fit status."""
    x,y,xt=data['x'],data['y'],data['xt'];sd=data['noise']
    if model=='GP (RBF)':
        result=GaussianProcess(sd,length).fit(x,y).predict(xt)
        status='Exact GP conditional on fixed kernel and known noise SD.'
    elif model=='Bayesian linear':
        feature=lambda z: np.column_stack([np.ones(len(z)),z])
        result=BayesianLinear(sd).fit(feature(x),y).predict(feature(xt))
        status='Exact conditional linear posterior; the straight-line mean is misspecified for this simulator.'
    elif model in MODELS[2:]:
        net=TinyMLP(seed=data['seed']).fit(x,y)
        if model==MODELS[2]:
            result=BayesianLinear(sd).fit(net.features(x),y).predict(net.features(xt))
            status='Last-layer posterior conditional on learned, frozen features; hidden-weight uncertainty omitted. '
        else:
            result=dict(mean=net.predict(xt),var_f=np.full(len(xt),np.nan),
                        var_y=np.full(len(xt),sd**2),cov_f=None)
            status='Point MLP plus assumed observation noise; no epistemic uncertainty estimate. '
        status+=f"Optimizer converged={net.status['converged']}; iterations={net.status['iterations']}; "+net.status['message']
    else: raise ValueError('Unknown model.')
    return result,status


def run_experiment(model='GP (RBF)',n=32,noise=.2,length=1.,seed=19,band='Both f and y',extra=None):
    """Return figure, region metrics DataFrame, status, CSV. No test-based selection."""
    data=make_data(int(n),float(noise),int(seed));pred,status=fit_prediction(model,data,float(length))
    xt=data['xt'];m=pred['mean'];sy=np.sqrt(pred['var_y']);sf=np.sqrt(pred['var_f'])
    fig,ax=plt.subplots(figsize=(9,4.6),layout='constrained')
    ax.axvspan(-.6,.6,color='#e7eaf1',label='Training gap')
    if band in ['Both f and y','Future observation y']:
        ax.fill_between(xt,m-1.96*sy,m+1.96*sy,color='#6ca3df',alpha=.22,label='95% future y (pointwise)')
    if band in ['Both f and y','Latent function f'] and np.all(np.isfinite(sf)):
        ax.fill_between(xt,m-1.96*sf,m+1.96*sf,color='#277b84',alpha=.4,label='95% latent f (pointwise)')
    if band=='Latent function f' and np.any(~np.isfinite(sf)):
        status+=' No f-band shown: this point model does not estimate it.'
    ax.plot(xt,data['ft'],'--',c='#414b60',label='Simulator truth (evaluation only)')
    ax.plot(xt,m,c='#155e75',lw=2,label='Predictive mean')
    ax.scatter(data['x'],data['y'],c='#d97736',s=24,label='Training observations',zorder=4)
    ax.set(xlabel='Input x',ylabel='Response',title=f'{model} | n={int(n)}, noise SD={noise:g}, seed={int(seed)}')
    ax.legend(fontsize=8,loc='upper left',ncol=2);plt.close(fig)
    regions={'all':np.ones(len(xt),dtype=bool),'observed range':(abs(xt)>=.6)&(abs(xt)<=3),
             'training gap':abs(xt)<.6,'extrapolation':abs(xt)>3}
    rows=[]
    for region,mask in regions.items():
        metrics=evaluate_predictions(data['yt'][mask],m[mask],pred['var_y'][mask])
        if extra:
            additions=extra(dict(metrics))
            if not isinstance(additions,dict) or set(additions)&set(metrics):
                raise ValueError('Extension must return a dict of new metric names.')
            metrics.update(additions)
        rows.append(dict(model=model,region=region,n=int(n),noise_sd=noise,length=length,seed=int(seed),**metrics))
    table=pd.DataFrame(rows)
    with tempfile.NamedTemporaryFile(prefix='w7-prediction-',suffix='.csv',delete=False) as f: path=f.name
    table.to_csv(path,index=False)
    return fig,table,status,path


def likelihood_demo(family='Bernoulli',eta=.5,y=1.):
    """One observation: show how a real score becomes a valid distribution parameter."""
    eta=float(eta);y=float(y)
    if family=='Gaussian':
        mean=eta; loss=gaussian_nll([y],[mean],1.)
        explanation='Identity inverse link. Known variance = 1; mean can be any real number.'
    elif family=='Bernoulli':
        mean=float(expit(eta));loss=bernoulli_nll([y],[eta])
        explanation='Sigmoid inverse link. Mean is P(y=1), not a hard decision; use y=0 or 1.'
    elif family=='Poisson':
        mean=float(np.exp(eta));loss=poisson_nll([y],[eta])
        explanation='Exponential inverse link. Mean is an expected count; use nonnegative integer y.'
    else: raise ValueError('Unknown likelihood.')
    return f'Expected response = {mean:.6f}\nNegative log likelihood = {loss:.6f}\n{explanation}'


def create_app(extra=None):
    """Construct Gradio Blocks without starting a server; optional new-metric callback."""
    import gradio as gr
    initial=run_experiment(extra=extra)
    def visible_table(frame):
        return frame.drop(columns=['model','n','noise_sd','length','seed']).rename(columns={
            'region':'Region','rmse':'RMSE','mean_nll':'Mean NLL',
            'coverage_95':'95% coverage','mean_width':'Mean width','count':'Test cases'}).round(4)
    def safe_likelihood(family,eta,y):
        try:
            return likelihood_demo(family,eta,y)
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc
    with gr.Blocks(title='W7 Predictive Uncertainty Workbench') as demo:
        gr.Markdown('# Predictive Uncertainty Workbench\nPredict → change one setting → compare → explain. All data are synthetic.')
        with gr.Tab('Prediction and uncertainty'):
            with gr.Row():
                model=gr.Dropdown(MODELS,value=MODELS[0],label='Model')
                band=gr.Radio(['Both f and y','Latent function f','Future observation y'],value='Both f and y',label='Displayed interval')
            with gr.Row():
                n=gr.Slider(8,80,value=32,step=1,label='Training observations')
                noise=gr.Slider(.05,.8,value=.2,step=.05,label='Observation noise SD')
                length=gr.Slider(.2,2.5,value=1.,step=.1,label='GP length scale (GP only)')
                seed=gr.Number(value=19,precision=0,label='Seed')
            gr.Markdown('Keep n, noise SD and seed fixed for model comparisons. Increasing n retains earlier pairs. Changing noise changes the simulated observations. GP settings are fixed assumptions; this is not test-set tuning. Intervals are pointwise, not simultaneous.')
            run=gr.Button('Fit and compare',variant='primary')
            plot=gr.Plot(value=initial[0],label='Prediction')
            table=gr.Dataframe(value=visible_table(initial[1]),interactive=False,label='Independent held-out observations')
            status=gr.Textbox(value=initial[2],label='Model assumptions and optimizer status')
            download=gr.File(value=initial[3],label='Export experiment CSV')
            def callback(a,b,c,d,e,f):
                plot,frame,status,path=run_experiment(a,b,c,d,e,f,extra)
                return plot,visible_table(frame),status,path
            run.click(callback,[model,n,noise,length,seed,band],[plot,table,status,download])
        with gr.Tab('Likelihood and loss'):
            family=gr.Dropdown(['Gaussian','Bernoulli','Poisson'],value='Bernoulli',label='Distribution')
            eta=gr.Slider(-3,3,value=.5,step=.1,label='Real-valued score eta')
            y=gr.Number(value=1,label='Observed response y')
            calc=gr.Button('Calculate likelihood loss')
            output=gr.Textbox(value=likelihood_demo(),lines=4,label='Interpretation')
            calc.click(safe_likelihood,[family,eta,y],output)
        gr.Markdown('Finish: identify what was conditioned on, report a numerical difference, and name one uncertainty this model does not capture. A share URL lasts only while the runtime is active.')
    return demo


if __name__=='__main__':
    create_app().queue(default_concurrency_limit=1).launch()
