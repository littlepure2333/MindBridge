import argparse

parser = argparse.ArgumentParser(description="MindBridge Configuration")
parser.add_argument(
    "--model_name", type=str, default="testing",
    help="name of model, used for ckpt saving and wandb logging (if enabled)",
)
parser.add_argument(
    "--data_path", type=str, default="../data/natural-scenes-dataset",
    help="Path to where NSD data is stored / where to download it to",
)
parser.add_argument(
    "--subj_list",type=int, default=[1], choices=[1,2,3,4,5,6,7,8], nargs='+',
    help="Subject index to train on",
)
parser.add_argument(
    "--subj_source",type=int, default=[1], choices=[1,2,3,4,5,6,7,8], nargs='+',
    help="Source subject index to be adapted from (Can be multiple subjects)",
)
parser.add_argument(
    "--subj_target",type=int, default=[1], choices=[1,2,3,4,5,6,7,8],
    help="Target subject index to be adapted to (Only one subject)",
)
parser.add_argument(
    "--adapting",action=argparse.BooleanOptionalAction,default=False,
    help="Whether to adapt from source to target subject",
)
parser.add_argument(
    "--batch_size", type=int, default=50,
    help="Batch size per GPU",
)
parser.add_argument(
    "--val_batch_size", type=int, default=50,
    help="Validation batch size per GPU",
)
parser.add_argument(
    "--clip_variant",type=str,default="ViT-L/14",choices=["RN50", "ViT-L/14", "ViT-B/32", "RN50x64"],
    help='OpenAI clip variant',
)
parser.add_argument(
    "--wandb_log",action=argparse.BooleanOptionalAction,default=True,
    help="Whether to log to wandb",
)
parser.add_argument(
    "--wandb_project",type=str,default="MindBridge",
    help="Wandb project name",
)
parser.add_argument(
    "--resume",action=argparse.BooleanOptionalAction,default=False,
    help="Resume training from latest checkpoint, can't do it with --load_from at the same time",
)
parser.add_argument(
    "--resume_id",type=str,default=None,
    help="Run id for wandb resume",
)
parser.add_argument(
    "--load_from",type=str,default=None,
    help="load model and restart, can't do it with --resume at the same time",
)
parser.add_argument(
    "--norm_embs",action=argparse.BooleanOptionalAction,default=True,
    help="Do l2-norming of CLIP embeddings",
)
parser.add_argument(
    "--use_image_aug",action=argparse.BooleanOptionalAction,default=True,
    help="whether to use image augmentation",
)
parser.add_argument(
    "--num_epochs",type=int,default=2000,
    help="number of epochs of training",
)
parser.add_argument(
    "--lr_scheduler_type",type=str,default='cycle',choices=['cycle','linear'],
    help="Type of learning rate scheduler",
)
parser.add_argument(
    "--ckpt_interval",type=int,default=10,
    help="Save backup ckpt and reconstruct every x epochs",
)
parser.add_argument(
    "--eval_interval",type=int,default=10,
    help="Evaluate the model every x epochs",
)
parser.add_argument(
    "--h_size",type=int,default=2048,
    help="Hidden size of MLP",
)
parser.add_argument(
    "--n_blocks",type=int,default=2,
    help="Number of Hidden layers in MLP",
)
parser.add_argument(
    "--seed",type=int,default=42,
    help="Seed for reproducibility",
)
parser.add_argument(
    "--num_workers",type=int,default=5,
    help="Number of workers in dataloader"
)
parser.add_argument(
    "--max_lr",type=float,default=3e-4,
    help="Max learning rate",
)
parser.add_argument(
    "--pool_num", type=int, default=8192,
    help="Number of pooling",
)
parser.add_argument(
    "--pool_type", type=str, default='max',
    help="Type of pooling: avg, max",
)
parser.add_argument(
    "--mse_mult", type=float, default=1e4,
    help="The weight of mse loss",
)
parser.add_argument(
    "--rec_mult", type=float, default=0,
    help="The weight of brain reconstruction loss",
)
parser.add_argument(
    "--cyc_mult", type=float, default=0,
    help="The weight of cycle loss",
)
parser.add_argument(
    # "--length", type=int, default=8559,
    "--length", type=int, default=None,
    help="Indicate dataset length",
)
parser.add_argument(
    "--autoencoder_name", type=str, default=None,
    help="name of trained autoencoder model",
)
parser.add_argument(
    "--subj_load",type=int, default=None, choices=[1,2,3,4,5,6,7,8], nargs='+',
    help="subj want to be load in the model",
)
parser.add_argument(
    "--subj_test",type=int, default=1, choices=[1,2,3,4,5,6,7,8],
    help="subj to test",
)
parser.add_argument(
    "--samples",type=int, default=None, nargs='+',
    help="Specify sample indice to reconstruction"
)
parser.add_argument(
    "--img2img_strength",type=float, default=.85,
    help="How much img2img (1=no img2img; 0=outputting the low-level image itself)",
)
parser.add_argument(
    "--guidance_scale",type=float, default=3.5,
    help="Guidance scale for diffusion model.",
)
parser.add_argument(
    "-num_inference_steps",type=int, default=20,
    help="Number of inference steps for diffusion model.",
)
parser.add_argument(
    "--recons_per_sample", type=int, default=16,
    help="How many recons to output, to then automatically pick the best one (MindEye uses 16)",
)
parser.add_argument(
    "--plotting", action=argparse.BooleanOptionalAction, default=True,
    help="plotting all the results",
)
parser.add_argument(
    "--vd_cache_dir", type=str, default='../weights',
    help="Where is cached Versatile Diffusion model; if not cached will download to this path",
)
parser.add_argument(
    "--gpu_id", type=int, default=0,
    help="ID of the GPU to be used",
)
parser.add_argument(
    "--ckpt_from", type=str, default='last',
    help="ckpt_from ['last', 'best']",
)
parser.add_argument(
    "--text_image_ratio", type=float, default=0.5,
    help="text_image_ratio in Versatile Diffusion. Only valid when use_text=True. 0.5 means equally weight text and image, 0 means use only image",
)
parser.add_argument(
    "--test_start", type=int, default=0,
    help="test range start index",
)
parser.add_argument(
    "--test_end", type=int, default=None,
    help="test range end index, the total length of test data is 982, so max index is 981",
)
parser.add_argument(
    "--only_embeddings", action=argparse.BooleanOptionalAction, default=False,
    help="only return semantic embeddings of networks",
)
parser.add_argument(
    "--synthesis", action=argparse.BooleanOptionalAction, default=False,
    help="synthesize new fMRI signals",
)
parser.add_argument(
    "--verbose", action=argparse.BooleanOptionalAction, default=True,
    help="print more information",
)
parser.add_argument(
    "--results_path", type=str, default=None,
    help="path to reconstructed outputs",
)

args = parser.parse_args()