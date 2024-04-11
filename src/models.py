import clip
import torchsnooper
import math
import random
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms

def add_hooks(module, parent_name=''):
    module_name = module.__class__.__name__
    if parent_name:
        module_name = f'{parent_name}.{module_name}'

    module.register_forward_hook(lambda mod, inp, out: forward_hook(mod, inp, out, module_name))
    module.register_backward_hook(lambda mod, inp, out: backward_hook(mod, inp, out, module_name))

    for name, child in module.named_children():
        add_hooks(child, parent_name=f'{module_name}.{name}')

def forward_hook(module, input, output, name):
    if output.isnan().any():
        print(f"NaN detected in forward pass in module: {name}")
        print(f"Input: {input}")
        print(f"Output: {output}")

def backward_hook(module, grad_input, grad_output, name):
    if any(tensor is not None and torch.isnan(tensor).any() for tensor in [*grad_input, *grad_output]):
        print(f"NaN detected in backward pass in module: {name}")
        print(f"Grad Input: {grad_input}")
        print(f"Grad Output: {grad_output}")

class Clipper(torch.nn.Module):
    def __init__(self, clip_variant, clamp_embs=False, norm_embs=False,
                 hidden_state=False, device=torch.device('cpu')):
        super().__init__()
        assert clip_variant in ("RN50", "ViT-L/14", "ViT-B/32", "RN50x64"), \
            "clip_variant must be one of RN50, ViT-L/14, ViT-B/32, RN50x64"
        print(clip_variant, device)
        
        if clip_variant=="ViT-L/14" and hidden_state:
            from transformers import CLIPVisionModelWithProjection, CLIPTextModelWithProjection, CLIPTokenizer
            image_encoder = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").eval()
            image_encoder = image_encoder.to(device)
            for param in image_encoder.parameters():
                param.requires_grad = False # dont need to calculate gradients
            self.image_encoder = image_encoder

            text_encoder = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").eval()
            text_encoder = text_encoder.to(device)
            for param in text_encoder.parameters():
                param.requires_grad = False # dont need to calculate gradients
            self.text_encoder = text_encoder
            self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

        elif hidden_state:
            raise Exception("hidden_state embeddings only works with ViT-L/14 right now")
        
        clip_model, preprocess = clip.load(clip_variant, device=device)
        clip_model.eval() # dont want to train model
        for param in clip_model.parameters():
            param.requires_grad = False # dont need to calculate gradients
            
        self.clip = clip_model
        self.clip_variant = clip_variant
        if clip_variant == "RN50x64":
            self.clip_size = (448,448)
        else:
            self.clip_size = (224,224)
            
        preproc = transforms.Compose([
            transforms.Resize(size=self.clip_size[0], interpolation=transforms.InterpolationMode.BICUBIC, antialias=None),
            transforms.CenterCrop(size=self.clip_size),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])
        self.preprocess = preproc
        self.hidden_state = hidden_state
        self.mean = np.array([0.48145466, 0.4578275, 0.40821073])
        self.std = np.array([0.26862954, 0.26130258, 0.27577711])
        self.normalize = transforms.Normalize(self.mean, self.std)
        self.denormalize = transforms.Normalize((-self.mean / self.std).tolist(), (1.0 / self.std).tolist())
        self.clamp_embs = clamp_embs
        self.norm_embs = norm_embs
        self.device= device
        
        def versatile_normalize_embeddings(encoder_output):
            embeds = encoder_output.last_hidden_state
            embeds = image_encoder.vision_model.post_layernorm(embeds)
            embeds = image_encoder.visual_projection(embeds)
            return embeds
        self.versatile_normalize_embeddings = versatile_normalize_embeddings

    def resize_image(self, image):
        # note: antialias should be False if planning to use Pinkney's Image Variation SD model
        return transforms.Resize(self.clip_size, antialias=None)(image.to(self.device))

    def embed_image(self, image):
        """Expects images in -1 to 1 range"""
        if self.hidden_state:
            # clip_emb = self.preprocess((image/1.5+.25).to(self.device)) # for some reason the /1.5+.25 prevents oversaturation
            clip_emb = self.preprocess((image).to(self.device))
            clip_emb = self.image_encoder(clip_emb)
            clip_emb = self.versatile_normalize_embeddings(clip_emb)
        else:
            clip_emb = self.preprocess(image.to(self.device))
            clip_emb = self.clip.encode_image(clip_emb)
        # input is now in CLIP space, but mind-reader preprint further processes embeddings:
        if self.clamp_embs:
            clip_emb = torch.clamp(clip_emb, -1.5, 1.5)
        if self.norm_embs:
            if self.hidden_state:        
                # normalize all tokens by cls token's norm
                clip_emb = clip_emb / torch.norm(clip_emb[:, 0], dim=-1).reshape(-1, 1, 1)
            else:
                clip_emb = nn.functional.normalize(clip_emb, dim=-1)
        return clip_emb
    
    def embed_text(self, prompt):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
        """

        def normalize_embeddings(encoder_output):
            embeds = self.text_encoder.text_projection(encoder_output.last_hidden_state)
            embeds_pooled = encoder_output.text_embeds
            embeds = embeds / torch.norm(embeds_pooled.unsqueeze(1), dim=-1, keepdim=True)
            return embeds

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="max_length", return_tensors="pt").input_ids
        with torch.no_grad():
            prompt_embeds = self.text_encoder(
                text_input_ids.to(self.device),
            )
        prompt_embeds = normalize_embeddings(prompt_embeds)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        # bs_embed, seq_len, _ = prompt_embeds.shape
        # prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        # prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        return prompt_embeds

    def embed_curated_annotations(self, annots):
        for i,b in enumerate(annots):
            t = ''
            while t == '':
                rand = torch.randint(5,(1,1))[0][0]
                t = b[0,rand]
            if i==0:
                txt = np.array(t)
            else:
                txt = np.vstack((txt,t))
        txt = txt.flatten()
        return self.embed_text(txt)

class Adapter_Layer(nn.Module):
    def __init__(self,
                 in_channels,
                 bottleneck=32,
                 out_channels=None,
                 dropout=0.0,
                 init_option="lora",
                 adapter_scalar="1.0",
                 adapter_layernorm_option=None):
        super().__init__()
        self.in_channels = in_channels
        self.down_size =  bottleneck
        self.out_channels = out_channels if out_channels is not None else in_channels
        # self.non_linearity = args.non_linearity  # use ReLU by default

        #_before
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.in_channels, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.out_channels)

        self.dropout = dropout

        if init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)

        up = up * self.scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output

class ResMLP(nn.Module):
    def __init__(self, h, n_blocks, dropout=0.15):
        super().__init__()
        self.n_blocks = n_blocks
        self.mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(h, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(dropout)
            ) for _ in range(n_blocks)
        ])

    def forward(self, x):
        residual = x
        for res_block in range(self.n_blocks):
            x = self.mlp[res_block](x)
            x += residual
            residual = x
        return x

class MindSingle(nn.Module):
    def __init__(self, in_dim=15724, out_dim_image=768, out_dim_text=None, 
                 h=4096, n_blocks=4, subj_list=None,):

        super().__init__()

        self.subj_list = subj_list
        self.embedder = nn.ModuleDict({
            str(subj): nn.Sequential(
                Adapter_Layer(in_dim, 128),
                nn.Linear(in_dim, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(0.5),
            ) for subj in subj_list
        })

        self.translator = ResMLP(h, n_blocks)
        self.head_image = nn.Linear(h, out_dim_image)
        self.head_text  = nn.Linear(h, out_dim_text)

    # @torchsnooper.snoop()
    def forward(self, x):
        x = self.embedder[str(self.subj_list[0])](x)
        x = self.translator(x)

        x_image = self.head_image(x)
        x_image = x_image.reshape(len(x_image), -1)

        x_text = self.head_text(x)
        x_text = x_text.reshape(len(x_text), -1)

        return x_image, x_text

class MindBridge(MindSingle):
    def __init__(self, in_dim=15724, out_dim_image=768, out_dim_text=None, 
                 h=4096, n_blocks=4, subj_list=None, adapting=False):
        
        assert len(subj_list) >= 2, "MindBridge requires at least 2 subjects"

        super().__init__(in_dim=in_dim, out_dim_image=out_dim_image, 
            out_dim_text=out_dim_text, h=h, n_blocks=n_blocks, subj_list=subj_list
        )

        self.builder = nn.ModuleDict({
            str(subj): nn.Sequential(
                nn.Linear(h, in_dim),
                nn.LayerNorm(in_dim),
                nn.GELU(),
                Adapter_Layer(in_dim, 128),
            ) for subj in subj_list
        })

        self.adapting = adapting
        self.cyc_loss = nn.MSELoss()

    # @torchsnooper.snoop()
    def forward(self, x):
        if len(x) == 2 and type(x) is tuple:
            subj_list = x[1].tolist() # (s,)
            x = x[0]         # (b,n)
        else:
            subj_list = self.subj_list

        x = x.squeeze()
        x_subj = torch.chunk(x,len(subj_list))
        x = []
        x_rec = []
        if self.adapting: # choose subj_a (source subject) and subj_b (target subject)
            subj_a, subj_b = subj_list[0], subj_list[-1]
        else: # random sample 2 subjects
            subj_a, subj_b = random.sample(subj_list, 2) 
        for i, subj_i in enumerate(subj_list):       # subj is 1-based
            x_i = self.embedder[str(subj_i)](x_subj[i])   # subj_i seman embedding
            if subj_i == subj_a: x_a = x_i                # subj_a seman embedding are choosen
            x.append(x_i)
            x_i_rec = self.builder[str(subj_i)](x_i)      # subj_i recon brain signals
            x_rec.append(x_i_rec)

        x = torch.concat(x, dim=0)
        x_rec = torch.concat(x_rec, dim=0)
        # del x_i, x_subj, x_i_rec
        
        # forward cycling
        x_b = self.builder[str(subj_b)](x_a)              # subj_b recon brain signal using subj_a seman embedding
        x_b = self.embedder[str(subj_b)](x_b)             # subj_b seman embedding (pseudo)
        loss_cyc = self.cyc_loss(x_a, x_b)

        x = self.translator(x)
        x_image = self.head_image(x)
        x_image = x_image.reshape(len(x_image), -1)

        x_text = self.head_text(x)
        x_text = x_text.reshape(len(x_text), -1)

        return x_image, x_text, x_rec, loss_cyc


from diffusers.models.vae import Decoder
class Voxel2StableDiffusionModel(torch.nn.Module):
    def __init__(self, in_dim=15724, h=4096, n_blocks=4, use_cont=False, ups_mode='4x'):
        super().__init__()
        self.lin0 = nn.Sequential(
            nn.Linear(in_dim, h, bias=False),
            nn.LayerNorm(h),
            nn.SiLU(inplace=True),
            nn.Dropout(0.5),
        )

        self.mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(h, h, bias=False),
                nn.LayerNorm(h),
                nn.SiLU(inplace=True),
                nn.Dropout(0.25)
            ) for _ in range(n_blocks)
        ])
        self.ups_mode = ups_mode
        if ups_mode=='4x':
            self.lin1 = nn.Linear(h, 16384, bias=False)
            self.norm = nn.GroupNorm(1, 64)
            
            self.upsampler = Decoder(
                in_channels=64,
                out_channels=4,
                up_block_types=["UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D"],
                block_out_channels=[64, 128, 256],
                layers_per_block=1,
            )

            if use_cont:
                self.maps_projector = nn.Sequential(
                    nn.Conv2d(64, 512, 1, bias=False),
                    nn.GroupNorm(1,512),
                    nn.ReLU(True),
                    nn.Conv2d(512, 512, 1, bias=False),
                    nn.GroupNorm(1,512),
                    nn.ReLU(True),
                    nn.Conv2d(512, 512, 1, bias=True),
                )
            else:
                self.maps_projector = nn.Identity()
        
        if ups_mode=='8x':  # prev best
            self.lin1 = nn.Linear(h, 16384, bias=False)
            self.norm = nn.GroupNorm(1, 256)
            
            self.upsampler = Decoder(
                in_channels=256,
                out_channels=4,
                up_block_types=["UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D"],
                block_out_channels=[64, 128, 256, 256],
                layers_per_block=1,
            )
            self.maps_projector = nn.Identity()
        
        if ups_mode=='16x':
            self.lin1 = nn.Linear(h, 8192, bias=False)
            self.norm = nn.GroupNorm(1, 512)
            
            self.upsampler = Decoder(
                in_channels=512,
                out_channels=4,
                up_block_types=["UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D", "UpDecoderBlock2D"],
                block_out_channels=[64, 128, 256, 256, 512],
                layers_per_block=1,
            )
            self.maps_projector = nn.Identity()

            if use_cont:
                self.maps_projector = nn.Sequential(
                    nn.Conv2d(64, 512, 1, bias=False),
                    nn.GroupNorm(1,512),
                    nn.ReLU(True),
                    nn.Conv2d(512, 512, 1, bias=False),
                    nn.GroupNorm(1,512),
                    nn.ReLU(True),
                    nn.Conv2d(512, 512, 1, bias=True),
                )
            else:
                self.maps_projector = nn.Identity()

    # @torchsnooper.snoop()
    def forward(self, x, return_transformer_feats=False):
        x = self.lin0(x)
        residual = x
        for res_block in self.mlp:
            x = res_block(x)
            x = x + residual
            residual = x
        x = x.reshape(len(x), -1)
        x = self.lin1(x)  # bs, 4096

        if self.ups_mode == '4x':
            side = 16
        if self.ups_mode == '8x':
            side = 8
        if self.ups_mode == '16x':
            side = 4
        
        # decoder
        x = self.norm(x.reshape(x.shape[0], -1, side, side).contiguous())
        if return_transformer_feats:
            return self.upsampler(x), self.maps_projector(x).flatten(2).permute(0,2,1)
        return self.upsampler(x)
    
if __name__ == "__main__":
    in_dim=8000
    h=2048
    head = nn.Sequential(
        Adapter_Layer(in_dim, 128),
        nn.Linear(in_dim, h),
        nn.LayerNorm(h),
        nn.GELU(),
        nn.Dropout(0.5),
    ) 

    def add_hooks(module, parent_name=''):
        module_name = module.__class__.__name__
        if parent_name:
            module_name = f'{parent_name}.{module_name}'

        for name, child in module.named_children():
            add_hooks(child, parent_name=f'{module_name}.{name}')

        print(module_name)

    add_hooks(head)
