# csc413-assignment-4-gan-models-solved
**TO GET THIS SOLUTION VISIT:** [CSC413 Assignment 4-Gan Models Solved](https://www.ankitcodinghub.com/product/csc413-assignment-4-gan-models-solved/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;100104&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;6&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (6 votes)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;CSC413 Assignment 4-Gan Models Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (6 votes)    </div>
    </div>
<div class="page" title="Page 2">
<div class="layoutArea">
<div class="column">
Introduction

In this assignment, you’ll get hands-on experience coding and training GANs. This assignment is divided into three parts: in the first part, we will implement a specific type of GAN designed to process images, called a Deep Convolutional GAN (DCGAN). We’ll train the DCGAN to generate emojis from samples of random noise. In the second part, we will look at a more complex GAN architecture called CycleGAN, which was designed for the task of image-to-image translation (de- scribed in more detail in Part 2). We’ll train the CycleGAN to convert between Apple-style and Windows-style emojis. In the third part, we will turn to a large-scale GAN model – BigGAN – that generates high resolution and high fidelity natural images. We will visualize the class imbeddings learned by BigGAN, and interpolate between them.

Part 1: Deep Convolutional GAN (DCGAN) [4pt]

For the first part of this assignment, we will implement a Deep Convolutional GAN (DCGAN). A DCGAN is simply a GAN that uses a convolutional neural network as the discriminator, and a network composed of transposed convolutions as the generator. To implement the DCGAN, we need to specify three things: 1) the generator, 2) the discriminator, and 3) the training procedure. We will go over each of these three components in the following subsections. Open https://colab. research.google.com/drive/1Jm2csPnWoKZ2DicPDF48Bq_xiukamPif on Colab and answer the following questions.

DCGAN

The discriminator in this DCGAN is a convolutional neural network that has the following archi- tecture:

The DCDiscriminator class is implemented for you. We strongly recommend you to carefully read the code, in particular the __init__ method. The three stages of the generator architec- tures are implemented using conv and upconv functions respectively, all of which provided in Helper Modules.

</div>
</div>
<div class="layoutArea">
<div class="column">
BatchNorm &amp; ReLU

</div>
<div class="column">
Discriminator

BatchNorm &amp; ReLU BatchNorm &amp; ReLU

</div>
</div>
<div class="layoutArea">
<div class="column">
32

Generator

</div>
<div class="column">
16

16

</div>
<div class="column">
84 8411

</div>
</div>
<div class="layoutArea">
<div class="column">
32

3

</div>
<div class="column">
32

</div>
</div>
<div class="layoutArea">
<div class="column">
64

</div>
<div class="column">
128

</div>
</div>
<div class="layoutArea">
<div class="column">
Now, we will implement the generator of the DCGAN, which consists of a sequence of transpose convolutional layers that progressively upsample the input noise sample to generate a fake image. The generator has the following architecture:

</div>
</div>
<div class="layoutArea">
<div class="column">
2

</div>
</div>
<div class="layoutArea">
<div class="column">
1

</div>
</div>
<div class="layoutArea">
<div class="column">
conv1

</div>
<div class="column">
conv2

</div>
<div class="column">
conv3

</div>
<div class="column">
conv4

</div>
</div>
</div>
<div class="page" title="Page 3">
<div class="layoutArea">
<div class="column">
Programming Assignment 4

</div>
</div>
<div class="layoutArea">
<div class="column">
Generator BatchNorm &amp; ReLU BatchNorm &amp; ReLU

148

</div>
<div class="column">
BatchNorm &amp; ReLU

upconv2

</div>
<div class="column">
tanh

upconv3

</div>
<div class="column">
32

</div>
</div>
<div class="layoutArea">
<div class="column">
100

</div>
<div class="column">
128

</div>
</div>
<div class="layoutArea">
<div class="column">
1

</div>
<div class="column">
48 64

upconv1

</div>
<div class="column">
16

16

</div>
</div>
<div class="layoutArea">
<div class="column">
Linear &amp; reshape

</div>
</div>
<div class="layoutArea">
<div class="column">
1. [1pt] Implementation: Implement this architecture by filling in the __init__ method of the DCGenerator class, shown below. Note that the forward pass of DCGenerator is already provided for you.

(Hint: You may find the provided DCDiscriminator useful.)

Note: The original DCGAN generator uses deconv function to expand the spatial dimension. Odena et al. later found the deconv creates checker board artifacts in the generated samples. In this assignment, we will use upconv that consists of an upsampling layer followed by conv2D to replace the deconv module (analogous to the conv function used for the discriminator above) in your generator implementation.

Training Loop

Next, you will implement the training loop for the DCGAN. A DCGAN is simply a GAN with a specific type of generator and discriminator; thus, we train it in exactly the same way as a standard GAN. The pseudo-code for the training procedure is shown below. The actual implementation is simpler than it may seem from the pseudo-code: this will give you practice in translating math to code.

Algorithm 1 GAN Training Loop Pseudocode

<ol>
<li>1: &nbsp;procedure TrainGAN</li>
<li>2: &nbsp;Draw m training examples {x(1), . . . , x(m)} from the data distribution pdata</li>
<li>3: &nbsp;Draw m noise samples {z(1),…,z(m)} from the noise distribution pz</li>
<li>4: &nbsp;Generate fake images from the noise: G(z(i)) for i ∈ {1, . . . .m}</li>
<li>5: &nbsp;Compute the (least-squares) discriminator loss:
( i ) 􏰌 2 􏰎 1 􏰆m 􏰍 􏰋 ( i ) 􏰌 2 􏰎
</li>
</ol>
</div>
</div>
<div class="layoutArea">
<div class="column">
3

</div>
</div>
<div class="layoutArea">
<div class="column">
32

</div>
<div class="column">
32 3

</div>
</div>
<div class="layoutArea">
<div class="column">
( D ) 1 􏰆m 􏰍 􏰋

</div>
<div class="column">
D(x )−1 +2m

</div>
</div>
<div class="layoutArea">
<div class="column">
J =2m

</div>
<div class="column">
D(G(z ))

</div>
</div>
<div class="layoutArea">
<div class="column">
i=1

</div>
<div class="column">
i=1

</div>
</div>
<div class="layoutArea">
<div class="column">
<ol start="6">
<li>6: &nbsp;Update the parameters of the discriminator</li>
<li>7: &nbsp;Draw m new noise samples {z(1),…,z(m)} from the noise distribution pz</li>
<li>8: &nbsp;Generate fake images from the noise: G(z(i)) for i ∈ {1, . . . .m}</li>
<li>9: &nbsp;Compute the (least-squares) generator loss:
J(G) = m1 􏰆m 􏰍􏰋D(G(z(i))) − 1􏰌2􏰎 i=1
</li>
<li>10: &nbsp;Update the parameters of the generator</li>
</ol>
</div>
</div>
</div>
<div class="page" title="Page 4">
<div class="layoutArea"></div>
<div class="layoutArea">
<div class="column">
1. [1pt] Implementation: Fill in the gan_training_loop function in the GAN section of the notebook.

There are 5 numbered bullets in the code to fill in for the discriminator and 3 bullets for the generator. Each of these can be done in a single line of code, although you will not lose marks for using multiple lines.

Note that in the discriminator update, we have provided you with the implementation of gradi- ent penalty. Gradient penalty adds a term in the discriminator loss, and it is another popular technique for stabilizing GAN training. Gradient penalty can take different forms, and it is an active research area to study its effect on GAN training [Gulrajani et al., 2017] [Kodali et al., 2017] [Mescheder et al., 2018].

Experiment

1. [1pt] We will train a DCGAN to generate Windows (or Apple) emojis in the Training – GAN section of the notebook. By default, the script runs for 20000 iterations, and should take approximately half an hour on Colab. The script saves the output of the generator for a fixed noise sample every 200 iterations throughout training; this allows you to see how the generator improves over time. How does the generator performance evolve over time? Include in your write-up some representative samples (e.g. one early in the training, one with satisfactory image quality, and one towards the end of training, and give the iteration number for those samples. Briefly comment on the quality of the samples.

2. [1pt] Multiple techniques can be used to stabilize GAN training. We have provided code for gradient_penalty [Thanh-Tung et al., 2019].

Try turn on the gradient_penalty flag in the args_dict and train the model again. Are you able to stabilize the training? Briefly explain why the gradient penalty can help. You are welcome to check out the related literature above for gradient penalty.

(Hint: Consider relationship between the Jacobian norm and its singular values.)

3. [0pt] Playing with some other hyperparameters such as spectral_norm. You can also try lowering lr (learning rate), and increasing d_train_iters (number of discriminator updates per generator update). Are you able to stabilize the training? Can you explain why the above measures help?

</div>
</div>
<div class="layoutArea">
<div class="column"></div>
</div>
</div>
<div class="page" title="Page 5">
<div class="layoutArea">
<div class="column">
&nbsp;

</div>
</div>
<div class="layoutArea">
<div class="column">
Part 2: CycleGAN [3pt]

Now we are going to implement the CycleGAN architecture.

Motivation: Image-to-Image Translation

Say you have a picture of a sunny landscape, and you wonder what it would look like in the rain. Or perhaps you wonder what a painter like Monet or van Gogh would see in it? These questions can be addressed through image-to-image translation wherein an input image is automatically converted into a new image with some desired appearance.

Recently, Generative Adversarial Networks have been successfully applied to image translation, and have sparked a resurgence of interest in the topic. The basic idea behind the GAN-based approaches is to use a conditional GAN to learn a mapping from input to output images. The loss functions of these approaches generally include extra terms (in addition to the standard GAN loss), to express constraints on the types of images that are generated.

A recently-introduced method for image-to-image translation called CycleGAN is particularly interesting because it allows us to use un-paired training data. This means that in order to train it to translate images from domain X to domain Y , we do not have to have exact correspondences between individual images in those domains. For example, in the paper that introduced CycleGANs, the authors are able to translate between images of horses and zebras, even though there are no images of a zebra in exactly the same position as a horse, and with exactly the same background, etc.

Thus, CycleGANs enable learning a mapping from one domain X (say, images of horses) to another domain Y (images of zebras) without having to find perfectly matched training pairs.

To summarize the differences between paired and un-paired data, we have: • Paired training data: {(x(i), y(i))}Ni=1

• Un-paired training data:

<ul>
<li>– &nbsp;Source set: {x(i)}Ni=1 with each x(i) ∈ X</li>
<li>– &nbsp;Target set: {y(j)}Mj=1 with each y(j) ∈ Y</li>
<li>– &nbsp;For example, X is the set of horse pictures, and Y is the set of zebra pictures, where there are no direct correspondences between images in X and Y
Emoji CycleGAN

Now we’ll build a CycleGAN and use it to translate emojis between two different styles, in partic- ular, Windows ↔ Apple emojis.

Generator

The generator in the CycleGAN has layers that implement three stages of computation: 1) the first stage encodes the input via a series of convolutional layers that extract the image features; 2) the second stage then transforms the features by passing them through one or more residual blocks; and 3) the third stage decodes the transformed features using a series of transpose convolutional layers, to build an output image of the same size as the input.

The residual block used in the transformation stage consists of a convolutional layer, where the input is added to the output of the convolution. This is done so that the characteristics of the output image (e.g., the shapes of objects) do not differ too much from the input.
</li>
</ul>
</div>
</div>
<div class="layoutArea">
<div class="column"></div>
</div>
</div>
<div class="page" title="Page 6">
<div class="layoutArea">
<div class="column">
Programming Assignment 4

</div>
</div>
<div class="layoutArea">
<div class="column">
Apple Emoji

</div>
<div class="column">
GXtoY

upconv

</div>
<div class="column">
GYtoX GXtoY

Windows Emoji

</div>
<div class="column">
GYtoX

upconv

[0, 1]

</div>
<div class="column">
Apple Emoji

</div>
</div>
<div class="layoutArea">
<div class="column">
conv

</div>
<div class="column">
conv

conv

</div>
</div>
<div class="layoutArea">
<div class="column">
Does the generated image look like it came from the set of Windows emojis?

</div>
</div>
<div class="layoutArea">
<div class="column">
DY

The CycleGenerator class is implemented for you. We strongly recommend you to carefully read the code, in particular the __init__ method. The three stages of the generator architectures are implemented using conv, ResnetBlock and upconv functions respectively, all of which provided in Helper Modules.

</div>
</div>
<div class="layoutArea">
<div class="column">
BatchNorm &amp; ReLU

</div>
<div class="column">
CycleGAN Generator

BatchNorm &amp; ReLU BatchNorm &amp; ReLU BatchNorm &amp; ReLU

16 8 8 16

</div>
<div class="column">
tanh

</div>
</div>
<div class="layoutArea">
<div class="column">
32

32

</div>
<div class="column">
32

</div>
</div>
<div class="layoutArea">
<div class="column">
100

</div>
<div class="column">
16

</div>
<div class="column">
88

</div>
<div class="column">
16

</div>
</div>
<div class="layoutArea">
<div class="column">
64

</div>
<div class="column">
64

</div>
</div>
<div class="layoutArea">
<div class="column">
32 33

</div>
</div>
<div class="layoutArea">
<div class="column">
conv1

</div>
<div class="column">
conv2

</div>
<div class="column">
upconv1

</div>
<div class="column">
upconv2

</div>
</div>
<div class="layoutArea">
<div class="column">
Redidual block

</div>
<div class="column">
32

</div>
<div class="column">
32

</div>
</div>
<div class="layoutArea">
<div class="column">
Note: There are two generators in the CycleGAN model, GX→Y and GY→X, but their imple- mentations are identical. Thus, in the code, GX→Y and GY →X are simply different instantiations of the same class.

CycleGAN Training Loop

Finally, we will take a look at the CycleGAN training procedure. The training objective is more involved than the procedure in Part 1, but an implementation is provided.

</div>
</div>
</div>
<div class="page" title="Page 7">
<div class="layoutArea">
<div class="column">
Programming Assignment 4

</div>
</div>
<div class="layoutArea">
<div class="column">
Algorithm 2 CycleGAN Training Loop Pseudocode

<ol>
<li>1: &nbsp;procedure TrainCycleGAN</li>
<li>2: &nbsp;Draw a minibatch of samples {x(1), . . . , x(m)} from domain X</li>
<li>3: &nbsp;Draw a minibatch of samples {y(1), . . . , y(m)} from domain Y</li>
<li>4: &nbsp;Compute the discriminator loss on real images:</li>
</ol>
</div>
</div>
<div class="layoutArea">
<div class="column">
Jfake = m

6: Update the discriminators

</div>
</div>
<div class="layoutArea">
<div class="column">
7: Compute the Y → X generator loss: 1n

</div>
</div>
<div class="layoutArea">
<div class="column">
(D) 1􏰆m

</div>
<div class="column">
1􏰆n

</div>
</div>
<div class="layoutArea">
<div class="column">
(DX(x(i)) − 1)2 + n 5: Compute the discriminator loss on fake images:

</div>
<div class="column">
(DY (y(j) − 1)2

</div>
</div>
<div class="layoutArea">
<div class="column">
Jreal = m

</div>
<div class="column">
i=1

</div>
<div class="column">
j=1

</div>
</div>
<div class="layoutArea">
<div class="column">
(D) 1􏰆m

</div>
<div class="column">
1􏰆n

(DY (GX→Y (x(i))))2 + n (DX (GY →X (y(j))))2

</div>
</div>
<div class="layoutArea">
<div class="column">
i=1

</div>
<div class="column">
j=1

</div>
</div>
<div class="layoutArea">
<div class="column">
J(GY→X) = 􏰆(DX(GY→X(y(j)))−1)2 +λcycleJ(Y→X→Y)

</div>
</div>
<div class="layoutArea">
<div class="column">
n

<ol start="8">
<li>8: &nbsp;Compute the X → Y generator loss:
1m

J (GX→Y ) = 􏰆(DY (GX→Y (x(i))) − 1)2 + λcycleJ (X→Y →X)

m
</li>
<li>9: &nbsp;Update the generators</li>
</ol>
Cycle Consistency

The most interesting idea behind CycleGANs (and the one from which they get their name) is the idea of introducing a cycle consistency loss to constrain the model. The idea is that when we translate an image from domain X to domain Y , and then translate the generated image back to domain X, the result should look like the original image that we started with.

The cycle consistency component of the loss is the L1 distance between the input images and their reconstructions obtained by passing through both generators in sequence (i.e., from domain X to Y via the X → Y generator, and then from domain Y back to X via the Y → X generator). The cycle consistency loss for the Y → X → Y cycle is expressed as follows:

i=1

CycleGAN Experiments

1. [1pt] Train the CycleGAN to translate Apple emojis to Windows emojis in the Training – CycleGAN section of the notebook. The script will train for 5000 iterations, and save generated samples in the samples_cyclegan folder. In each sample, images from the source domain are shown with their translations to the right.

</div>
</div>
<div class="layoutArea">
<div class="column">
j=1

</div>
</div>
<div class="layoutArea">
<div class="column">
i=1

</div>
</div>
<div class="layoutArea">
<div class="column">
cycle

</div>
</div>
<div class="layoutArea">
<div class="column">
cycle

</div>
</div>
<div class="layoutArea">
<div class="column">
( Y → X → Y ) 1 􏰆m

</div>
</div>
<div class="layoutArea">
<div class="column">
∥y(i) − GX→Y (GY →X (y(i)))∥1,

where λcycle is a scalar hyper-parameter balancing the two loss terms: the cycle consistant loss and

</div>
</div>
<div class="layoutArea">
<div class="column">
λcycleJcycle = λcycle m

the GAN loss. The loss for the X → Y → X cycle is analogous.

</div>
</div>
</div>
<div class="page" title="Page 8">
<div class="layoutArea">
<div class="column">
Include in your writeup the samples from both generators at either iteration 200 and samples from a later iteration.

2. [1pt] Change the random seed and train the CycleGAN again. What are the most noticible difference between the similar quality samples from the different random seeds? Explain why there is such a difference?

3. [1pt] Changing the default lambda_cycle hyperparameters and train the CycleGAN again.

Try a couple of different values including without the cycle-consistency loss. (i.e. lambda_cycle = 0)

For different values of lambda_cycle, include in your writeup some samples from both gen- erators at either iteration 200 and samples from a later iteration. Do you notice a difference between the results with and without the cycle consistency loss? Write down your observa- tions (positive or negative) in your writeup. Can you explain these results, i.e., why there is or isn’t a difference among the experiments?

</div>
</div>
</div>
<div class="page" title="Page 9">
<div class="layoutArea">
<div class="column">
Part 3: BigGAN [2pt]

For this part, we will implement the interpolation function that you see in many GAN papers to show that the GAN model can generate novel images between classes. We will use BigGAN, Brock et al. [2018], as our learned model to do the interpolation.

BigGAN is a state-of-the-art model that pulls together a suite of recent best practices in training GANs. The authors scaled up the model to large-scale datasets like Imagenet. The result is the routine generation of both high-resolution (large) and high-quality (high-fidelity) images.

</div>
</div>
<div class="layoutArea">
<div class="column">
Figure 1: “Cherry-picked” high-resolution samples from BigGAN

BigGAN benefits from larger batch size and more expressive generator and discriminator. A varied range of techniques has been proposed to improve GANs in terms of sample quality, diver- sity and training stability. BigGAN utilized self-attention Zhang et al. [2018], moving average of generator weights Karras et al. [2017] , orthogonal initialization Saxe et al. [2013], and different residual block structures. See figure 2 for the illustration of the residual block for both generator and discriminator. They used conditional GAN where generator receives both a learnable class em- bedding and a noise vector. In the generator architecture, conditional batch normalization Huang and Belongie [2017] conditioned on class embedding is used. They also proposed truncation of noise distribution to control fidelity vs quality of samples. The paper shows that applying Spectral NormalizationMiyato et al. [2018] in generator improves stability, allowing for fewer discriminator steps per iteration.

</div>
</div>
<div class="layoutArea">
<div class="column">
Figure 2: Left: Architecture of generator. Middle: Residual module of generator. Right: Residual block of discriminator (taken from original paper)

</div>
</div>
<div class="layoutArea">
<div class="column"></div>
</div>
</div>
<div class="page" title="Page 10">
<div class="layoutArea">
<div class="column">
&nbsp;

</div>
</div>
<div class="layoutArea">
<div class="column">
BigGAN Experiments

In this part we would our pre-trained BigGAN to “fantasize” a new class category and generate samples from that class. We will create a new class embedding, rnew class, using a linear interpola- tion between the two existing class embeddings, rclass1 and rclass2:

rnew class = αrclass1 + (1 − α)rclass2, ∀α ∈ [0, 1].

Open https://colab.research.google.com/drive/1mKk7isQthab5k68FqqcdSJTT4P-7DV-s on

Colab and answer the following questions.

1. [1pt] Based on T-SNE visualization of class embeddings, which two classes are good candidates for interpolation? Which two classes might not be a good match for interpolation? In each case, give 2 examples. Briefly explain your choice.

2. [1pt] Complete generate_linear_interpolate_sample function. Verify the examples you gave in the previous question by generating samples. Include the four sets of generated images in your report.

</div>
</div>
</div>
<div class="page" title="Page 11">
<div class="layoutArea">
<div class="column">
What you need to submit

<ul>
<li>Your code file: dcgan.ipynb and biggan.ipynb.</li>
<li>A PDF document titled a4-writeup.pdfcontaining code screenshots, any experiment
results or visualizations, as well as your answers to the written questions. Further Resources

For further reading on GANs in general, and CycleGANs in particular, the following links may be useful:
</li>
</ul>
<ol>
<li>Deconvolution and Checkerboard Artifacts (Odena et al., 2016)</li>
<li>Unpaired image-to-image translation using cycle-consistent adversarial networks (Zhu et al., 2017)</li>
<li>Generative Adversarial Nets (Goodfellow et al., 2014)</li>
<li>An Introduction to GANs in Tensorflow</li>
<li>Generative Models Blog Post from OpenAI</li>
<li>Official PyTorch Implementations of Pix2Pix and CycleGAN</li>
</ol>
References

Andrew Brock, Jeff Donahue, and Karen Simonyan. Large scale gan training for high fidelity natural image synthesis. arXiv preprint arXiv:1809.11096, 2018.

Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, and Aaron C Courville. Improved training of wasserstein gans. In Advances in neural information processing systems, pages 5767–5777, 2017.

Xun Huang and Serge Belongie. Arbitrary style transfer in real-time with adaptive instance nor- malization. In Proceedings of the IEEE International Conference on Computer Vision, pages 1501–1510, 2017.

Tero Karras, Timo Aila, Samuli Laine, and Jaakko Lehtinen. Progressive growing of gans for improved quality, stability, and variation. arXiv preprint arXiv:1710.10196, 2017.

Naveen Kodali, Jacob Abernethy, James Hays, and Zsolt Kira. On convergence and stability of gans. arXiv preprint arXiv:1705.07215, 2017.

Lars Mescheder, Andreas Geiger, and Sebastian Nowozin. Which training methods for gans do actually converge? arXiv preprint arXiv:1801.04406, 2018.

Takeru Miyato, Toshiki Kataoka, Masanori Koyama, and Yuichi Yoshida. Spectral normalization for generative adversarial networks. arXiv preprint arXiv:1802.05957, 2018.

Andrew M Saxe, James L McClelland, and Surya Ganguli. Exact solutions to the nonlinear dy- namics of learning in deep linear neural networks. arXiv preprint arXiv:1312.6120, 2013.

</div>
</div>
</div>
<div class="page" title="Page 12">
<div class="layoutArea">
<div class="column">
Hoang Thanh-Tung, Truyen Tran, and Svetha Venkatesh. Improving generalization and stability of generative adversarial networks. arXiv preprint arXiv:1902.03984, 2019.

Han Zhang, Ian Goodfellow, Dimitris Metaxas, and Augustus Odena. Self-attention generative adversarial networks. arXiv preprint arXiv:1805.08318, 2018.

</div>
</div>
<div class="layoutArea">
<div class="column">
12

</div>
</div>
</div>
