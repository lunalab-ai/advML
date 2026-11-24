# From VAEs to Diffusion Models — Why Predict Noise? · Quiz explanations

Original formative checks for the approved W13 lecture.

## 1. Which parameters does the VAE encoder learn in the Gaussian example?

The encoder learns functions of the input that set the latent Gaussian mean and scale.

## 2. Which variance appears in a direct draw of the forward marginal?

The accumulated variance is one minus the product of the alpha coefficients through time t.

## 3. Which quantity supplies the supervised label for the noise predictor?

The Gaussian noise drawn when constructing the corrupted data point at the sampled time.

## 4. Why can two reverse samples from a shared state differ?

The conditional mean is shared and the Gaussian step noise is drawn independently. In this notebook the variance is zero at time 1.

## 5. What could a pooled count across both circles conceal?

A generated sample could concentrate its points in one component. The separate left and right counts reveal that imbalance.
