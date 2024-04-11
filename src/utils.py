import numpy as np
from torchvision import transforms
import torch
import torch.nn as nn
import PIL
import random
import os
import matplotlib.pyplot as plt
import torchsnooper


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def seed_everything(seed=0, cudnn_deterministic=True):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
    else:
        ## needs to be False to use conv3D
        print('Note: not using cudnn.deterministic')

def np_to_Image(x):
    if x.ndim==4:
        x=x[0]
    return PIL.Image.fromarray((x.transpose(1, 2, 0)*127.5+128).clip(0,255).astype('uint8'))

def torch_to_Image(x):
    if x.ndim==4:
        x=x[0]
    return transforms.ToPILImage()(x)

def Image_to_torch(x):
    try:
        x = (transforms.ToTensor()(x)[:3].unsqueeze(0)-.5)/.5
    except:
        x = (transforms.ToTensor()(x[0])[:3].unsqueeze(0)-.5)/.5
    return x

def torch_to_matplotlib(x,device=device):
    if torch.mean(x)>10:
        x = (x.permute(0, 2, 3, 1)).clamp(0, 255).to(torch.uint8)
    else:
        x = (x.permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8)
    if device=='cpu':
        return x[0]
    else:
        return x.cpu().numpy()[0]

def pairwise_cosine_similarity(A, B, dim=1, eps=1e-8):
    #https://stackoverflow.com/questions/67199317/pytorch-cosine-similarity-nxn-elements
    numerator = A @ B.T
    A_l2 = torch.mul(A, A).sum(axis=dim)
    B_l2 = torch.mul(B, B).sum(axis=dim)
    denominator = torch.max(torch.sqrt(torch.outer(A_l2, B_l2)), torch.tensor(eps))
    return torch.div(numerator, denominator)

def batchwise_cosine_similarity(Z,B):
    # https://www.h4pz.co/blog/2021/4/2/batch-cosine-similarity-in-pytorch-or-numpy-jax-cupy-etc
    B = B.T
    Z_norm = torch.linalg.norm(Z, dim=1, keepdim=True)  # Size (n, 1).
    B_norm = torch.linalg.norm(B, dim=0, keepdim=True)  # Size (1, b).
    cosine_similarity = ((Z @ B) / (Z_norm @ B_norm)).T
    return cosine_similarity

def topk(similarities,labels,k=5):
    if k > similarities.shape[0]:
        k = similarities.shape[0]
    topsum=0
    for i in range(k):
        topsum += torch.sum(torch.argsort(similarities,axis=1)[:,-(i+1)] == labels)/len(labels)
    return topsum

def soft_clip_loss(preds, targs, temp=0.005, eps=1e-10):
    clip_clip = (targs @ targs.T)/temp + eps
    check_loss(clip_clip, "clip_clip")
    brain_clip = (preds @ targs.T)/temp + eps
    check_loss(brain_clip, "brain_clip")
    
    loss1 = -(brain_clip.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
    check_loss(loss1, "loss1")
    loss2 = -(brain_clip.T.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
    check_loss(loss2, "loss2")
    
    loss = (loss1 + loss2)/2
    return loss

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('param counts:\n{:,} total\n{:,} trainable'.format(total, trainable))

def image_grid(imgs, rows, cols):
    w, h = imgs[0].size
    grid = PIL.Image.new('RGB', size=(cols*w, rows*h))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def check_loss(loss, message="loss"):
    if loss.isnan().any():
        raise ValueError(f'NaN loss in {message}')

def cosine_anneal(start, end, steps):
    return end + (start - end)/2 * (1 + torch.cos(torch.pi*torch.arange(steps)/(steps-1)))

def decode_latents(latents,vae):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    return image

@torch.no_grad()
def reconstruction(
    image, voxel, voxel2clip,
    clip_extractor,
    unet, vae, noise_scheduler,
    img_lowlevel = None,
    num_inference_steps = 50,
    recons_per_sample = 1,
    guidance_scale = 7.5,
    img2img_strength = .85,
    seed = 42,
    plotting=True,
    verbose=False,
    n_samples_save=1,
    device = None,
    mem_efficient = True,

):
    assert n_samples_save==1, "n_samples_save must = 1. Function must be called one image at a time"
    assert recons_per_sample>0, "recons_per_sample must > 0"
    
    brain_recons = None
    
    voxel=voxel[:n_samples_save]
    image=image[:n_samples_save]
    B = voxel.shape[0]

    if mem_efficient:
        clip_extractor.to("cpu")
        unet.to("cpu")
        vae.to("cpu")
    else:
        clip_extractor.to(device)
        unet.to(device)
        vae.to(device)

    if unet is not None:
        do_classifier_free_guidance = guidance_scale > 1.0
        vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        height = unet.config.sample_size * vae_scale_factor
        width = unet.config.sample_size * vae_scale_factor
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    if voxel2clip is not None:
        clip_results = voxel2clip(voxel)

        if mem_efficient:
            voxel2clip.to('cpu')

        brain_clip_image_embeddings, brain_clip_text_embeddings = clip_results[:2]
        brain_clip_image_embeddings = brain_clip_image_embeddings.reshape(B,-1,768)
        brain_clip_text_embeddings  = brain_clip_text_embeddings.reshape(B,-1,768)

        brain_clip_image_embeddings = brain_clip_image_embeddings.repeat(recons_per_sample, 1, 1)
        brain_clip_text_embeddings  = brain_clip_text_embeddings.repeat(recons_per_sample, 1, 1)

    if recons_per_sample > 0:
        for samp in range(len(brain_clip_image_embeddings)):
            brain_clip_image_embeddings[samp] = brain_clip_image_embeddings[samp]/(brain_clip_image_embeddings[samp,0].norm(dim=-1).reshape(-1, 1, 1) + 1e-6)
            brain_clip_text_embeddings[samp]  = brain_clip_text_embeddings[samp]/(brain_clip_text_embeddings[samp,0].norm(dim=-1).reshape(-1, 1, 1) + 1e-6)
        input_embedding = brain_clip_image_embeddings#.repeat(recons_per_sample, 1, 1)
        if verbose: print("input_embedding",input_embedding.shape)

        prompt_embeds = brain_clip_text_embeddings
        if verbose: print("prompt_embedding",prompt_embeds.shape)

        if do_classifier_free_guidance:
            input_embedding = torch.cat([torch.zeros_like(input_embedding), input_embedding]).to(device).to(unet.dtype)
            prompt_embeds = torch.cat([torch.zeros_like(prompt_embeds), prompt_embeds]).to(device).to(unet.dtype)

        # 3. dual_prompt_embeddings
        input_embedding = torch.cat([prompt_embeds, input_embedding], dim=1)

        # 4. Prepare timesteps
        noise_scheduler.set_timesteps(num_inference_steps=num_inference_steps, device=device)

        # 5b. Prepare latent variables
        batch_size = input_embedding.shape[0] // 2 # divide by 2 bc we doubled it for classifier-free guidance
        shape = (batch_size, unet.in_channels, height // vae_scale_factor, width // vae_scale_factor)
        if img_lowlevel is not None: # use img_lowlevel for img2img initialization
            init_timestep = min(int(num_inference_steps * img2img_strength), num_inference_steps)
            t_start = max(num_inference_steps - init_timestep, 0)
            timesteps = noise_scheduler.timesteps[t_start:]
            latent_timestep = timesteps[:1].repeat(batch_size)
            
            if verbose: print("img_lowlevel", img_lowlevel.shape)
            img_lowlevel_embeddings = clip_extractor.normalize(img_lowlevel)
            if verbose: print("img_lowlevel_embeddings", img_lowlevel_embeddings.shape)
            if mem_efficient:
                vae.to(device)
            init_latents = vae.encode(img_lowlevel_embeddings.to(device).to(vae.dtype)).latent_dist.sample(generator)
            init_latents = vae.config.scaling_factor * init_latents
            init_latents = init_latents.repeat(recons_per_sample, 1, 1, 1)

            noise = torch.randn([recons_per_sample, 4, 64, 64], device=device, 
                                generator=generator, dtype=input_embedding.dtype)
            init_latents = noise_scheduler.add_noise(init_latents, noise, latent_timestep)
            latents = init_latents
        else:
            timesteps = noise_scheduler.timesteps
            latents = torch.randn([recons_per_sample, 4, 64, 64], device=device,
                                  generator=generator, dtype=input_embedding.dtype)
            latents = latents * noise_scheduler.init_noise_sigma

        # 7. Denoising loop
        if mem_efficient:
            unet.to(device)
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)
            if verbose: print("timesteps: {}, latent_model_input: {}, input_embedding: {}".format(i, latent_model_input.shape, input_embedding.shape))
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=input_embedding).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = noise_scheduler.step(noise_pred, t, latents).prev_sample
        
        if mem_efficient:
            unet.to("cpu")

        recons = decode_latents(latents.to(device),vae.to(device)).detach().cpu()

        brain_recons = recons.unsqueeze(0)

    if verbose: print("brain_recons",brain_recons.shape)
                    
    # pick best reconstruction out of several
    best_picks = np.zeros(n_samples_save).astype(np.int16)
    
    if mem_efficient:
        vae.to("cpu")
        unet.to("cpu")
        clip_extractor.to(device)

    clip_image_target = clip_extractor.embed_image(image)
    clip_image_target_norm = nn.functional.normalize(clip_image_target.flatten(1), dim=-1)
    sims=[]
    for im in range(recons_per_sample): 
        currecon = clip_extractor.embed_image(brain_recons[0,[im]].float()).to(clip_image_target_norm.device).to(clip_image_target_norm.dtype)
        currecon = nn.functional.normalize(currecon.view(len(currecon),-1),dim=-1)
        cursim = batchwise_cosine_similarity(clip_image_target_norm,currecon)
        sims.append(cursim.item())
    if verbose: print(sims)
    best_picks[0] = int(np.nanargmax(sims))   
    if verbose: print(best_picks)
    if mem_efficient:
        clip_extractor.to("cpu")
        voxel2clip.to(device)
                    
    img2img_samples = 0 if img_lowlevel is None else 1
    num_xaxis_subplots = 1+img2img_samples+recons_per_sample
    if plotting:
        fig, ax = plt.subplots(n_samples_save, num_xaxis_subplots, 
                           figsize=(num_xaxis_subplots*5,6*n_samples_save),facecolor=(1, 1, 1))
    else:
        fig = None
        recon_img = None
    
    im = 0
    if plotting:
        ax[0].set_title(f"Original Image")
        ax[0].imshow(torch_to_Image(image[im]))
        if img2img_samples == 1:
            ax[1].set_title(f"Img2img ({img2img_strength})")
            ax[1].imshow(torch_to_Image(img_lowlevel[im].clamp(0,1)))
    for ii,i in enumerate(range(num_xaxis_subplots-recons_per_sample,num_xaxis_subplots)):
        recon = brain_recons[im][ii]
        if plotting:
            if ii == best_picks[im]:
                ax[i].set_title(f"Reconstruction",fontweight='bold')
                recon_img = recon
            else:
                ax[i].set_title(f"Recon {ii+1} from brain")
            ax[i].imshow(torch_to_Image(recon))
    if plotting:
        for i in range(num_xaxis_subplots):
            ax[i].axis('off')
    
    return fig, brain_recons, best_picks, recon_img
  