from abc import abstractmethod
import os
import math
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import wandb

# tf32 data type is faster than standard float32
torch.backends.cuda.matmul.allow_tf32 = True

# Custom models and functions #
import utils
import data

class Trainer:
    def __init__(self, args, accelerator, voxel2clip, clip_extractor, prompts_list, device) -> None:
        # train logs path
        self.outdir = os.path.abspath(f'../train_logs/{args.model_name}')
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir,exist_ok=True)
        
        self.args = args
        self.accelerator = accelerator
        self.voxel2clip = voxel2clip
        self.clip_extractor = clip_extractor
        self.prompts_list = prompts_list
        self.device = device
        self.num_devices = max(torch.cuda.device_count(), 1)
        self.epoch_start = 0

        self.prepare_dataloader()
        self.prepare_optimizer()
        self.prepare_scheduler()
        self.prepare_multi_gpu()

    @abstractmethod
    def prepare_dataloader(self):
        pass

    def prepare_optimizer(self,):
        # Prepare optimizer
        no_decay = ['bias', 'Norm', 'temperature']
        opt_grouped_parameters = [
            {'params': [p for n, p in self.voxel2clip.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
            {'params': [p for n, p in self.voxel2clip.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = torch.optim.AdamW(opt_grouped_parameters, lr=self.args.max_lr)

    def prepare_scheduler(self):
        # prepare lr scheduler
        one_epoch_steps = self.num_batches
        if self.accelerator.state.deepspeed_plugin is not None: # Multi GPU
            one_epoch_steps = math.ceil(one_epoch_steps // self.num_devices)
        total_steps = self.args.num_epochs * one_epoch_steps
        print("one_epoch_steps_per_gpu:",one_epoch_steps)
        print("total_steps:",total_steps)

        if self.args.lr_scheduler_type == 'linear':
            self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                total_iters=total_steps,
                last_epoch=-1
            )
        elif self.args.lr_scheduler_type == 'cycle':
            self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, 
                max_lr=self.args.max_lr,
                total_steps=total_steps,
                final_div_factor=100,
                last_epoch=-1, 
                pct_start=2/self.args.num_epochs,
            )

    def prepare_wandb(self, local_rank, args):
        ## Weights and Biases
        if local_rank==0 and args.wandb_log: # only use main process for wandb logging
            import wandb
            wandb_run = args.model_name
            wandb_notes = ''
            
            print(f"Wandb project {args.wandb_project} run {wandb_run}")
            wandb.login(host='https://api.wandb.ai')
            wandb_config = vars(args)
            print("wandb_config:\n",wandb_config)
            if args.resume: # wandb_auto_resume
                if args.resume_id is None:
                    args.resume_id = args.model_name
                print("wandb_id:", args.resume_id)
                wandb.init(
                    id = args.resume_id,
                    project=args.wandb_project,
                    name=wandb_run,
                    config=wandb_config,
                    notes=wandb_notes,
                    resume="allow",
                )
            else:
                wandb.init(
                    project=args.wandb_project,
                    name=wandb_run,
                    config=wandb_config,
                    notes=wandb_notes,
                )

    @abstractmethod
    def prepare_multi_gpu(self):
        pass

    def input(self, voxel, subj_id):
        return (voxel, subj_id)

    def train(self, local_rank):
        epoch = self.epoch_start
        self.losses, self.val_losses, self.lrs = [], [], []
        self.best_sim = 0
        self.best_epoch = 0

        self.val_voxel0 = self.val_image0 = None

        ## Main loop
        print(f"{self.args.model_name} starting with epoch {epoch} / {self.args.num_epochs}")
        progress_bar = tqdm(range(epoch, self.args.num_epochs), disable=(local_rank!=0))

        for epoch in progress_bar:
            self.voxel2clip.train()

            self.sims_image = 0.
            self.sims_text = 0.
            self.val_sims_image = 0.
            self.val_sims_text = 0.
            self.fwd_percent_correct = 0.
            self.bwd_percent_correct = 0.
            self.val_fwd_percent_correct = 0.
            self.val_bwd_percent_correct = 0.
            self.loss_clip_image_sum = 0.
            self.loss_clip_text_sum = 0.
            self.loss_mse_image_sum = 0.
            self.loss_mse_text_sum = 0.
            self.loss_rec_sum = 0.
            self.loss_cyc_sum = 0.
            self.val_loss_clip_image_sum = 0.
            self.val_loss_clip_text_sum = 0.
            self.val_loss_mse_image_sum = 0.
            self.val_loss_mse_text_sum = 0.
            self.val_loss_rec_sum = 0.
            self.val_loss_cyc_sum = 0.

            # wandb logging
            self.train_epoch(epoch)
            self.log_train()

            if epoch % self.args.eval_interval == 0:
                self.eval_epoch(epoch)
                self.log_val()
            
            if self.args.wandb_log and local_rank==0:
                wandb.log(self.logs)

            progress_dict = {
                "epoch": epoch,
                "lr": self.logs["train/lr"],
                "loss": self.logs["train/loss"],
            }
            
            progress_bar.set_postfix(progress_dict)

            # Main process
            if local_rank==0:
                # Uploading logs to wandb
                if self.args.wandb_log:
                    wandb.log(self.logs)
                # Save model
                if epoch % self.args.ckpt_interval == 0 or epoch == self.args.num_epochs-1:
                    self.save(epoch)

            # wait for other GPUs to catch up if needed
            self.accelerator.wait_for_everyone()

    @abstractmethod
    def train_epoch(self, epoch):
        pass

    def train_step(self, voxel, image, captions, subj_id):
        loss = 0.
        self.optimizer.zero_grad()

        if self.args.use_image_aug:
            image = data.img_augment(image)
        clip_image = self.clip_extractor.embed_image(image).float()   
        clip_text = self.clip_extractor.embed_text(captions).float()

        # clip_image_pred, clip_text_pred, voxel_rec, loss_cyc = self.voxel2clip((voxel, subj_id))
        results = self.voxel2clip(self.input(voxel, subj_id))

        # image clip loss
        clip_image_pred = results[0]
        clip_image_pred_norm = nn.functional.normalize(clip_image_pred.flatten(1), dim=-1)
        clip_image_norm = nn.functional.normalize(clip_image.flatten(1), dim=-1)
        loss_clip_image = utils.soft_clip_loss(
            clip_image_pred_norm,
            clip_image_norm,
        )

        utils.check_loss(loss_clip_image, "loss_clip_image")
        loss += loss_clip_image
        self.loss_clip_image_sum += loss_clip_image.item()

        # image mse loss
        if self.args.mse_mult:
            loss_mse_image = nn.MSELoss()(clip_image_pred_norm, clip_image_norm)
            utils.check_loss(loss_mse_image, "loss_mse_image")
            loss += self.args.mse_mult * loss_mse_image
            self.loss_mse_image_sum += loss_mse_image.item()

        # text clip loss
        clip_text_pred = results[1]
        clip_text_pred_norm = nn.functional.normalize(clip_text_pred.flatten(1), dim=-1)
        clip_text_norm = nn.functional.normalize(clip_text.flatten(1), dim=-1)
        loss_clip_text = utils.soft_clip_loss(
            clip_text_pred_norm,
            clip_text_norm,
        )
        utils.check_loss(loss_clip_text, "loss_clip_text")
        loss += loss_clip_text
        self.loss_clip_text_sum += loss_clip_text.item()

        # text mse loss
        if self.args.mse_mult:
            loss_mse_text = nn.MSELoss()(clip_text_pred_norm, clip_text_norm)
            utils.check_loss(loss_mse_text, "loss_mse_text")
            loss += self.args.mse_mult * loss_mse_text
            self.loss_mse_text_sum += loss_mse_text.item()

        # brain reconstruction loss
        if self.args.rec_mult:
            voxel_rec = results[2]
            loss_rec = nn.MSELoss()(voxel, voxel_rec)
            utils.check_loss(loss_rec, "loss_rec")
            loss += self.args.rec_mult * loss_rec            
            self.loss_rec_sum += loss_rec.item()
        
        # cycle loss
        if self.args.cyc_mult:
            loss_cyc = results[3]
            utils.check_loss(loss_cyc, "loss_cyc")
            loss += self.args.cyc_mult * loss_cyc
            self.loss_cyc_sum += loss_cyc.item()
        
        utils.check_loss(loss)
        self.accelerator.backward(loss)
        self.optimizer.step()

        self.losses.append(loss.item())
        self.lrs.append(self.optimizer.param_groups[0]['lr'])
        self.lr_scheduler.step()

        self.sims_image += nn.functional.cosine_similarity(clip_image_norm,clip_image_pred_norm).mean().item()
        self.sims_text += nn.functional.cosine_similarity(clip_text_norm,clip_text_pred_norm).mean().item()

        # forward and backward top 1 accuracy
        labels = torch.arange(len(clip_image_norm)).to(self.device) 
        self.fwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_image_pred_norm, clip_image_norm), labels, k=1)
        self.bwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_image_norm, clip_image_pred_norm), labels, k=1)

    @abstractmethod
    def eval_epoch(self, epoch):
        pass

    def eval_step(self, voxel, image, captions, subj_id):
        val_loss = 0.
        with torch.no_grad():
            # used for reconstruction
            if self.val_image0 is None:
                self.val_image0 = image.detach().clone()
                self.val_voxel0 = voxel.detach().clone()

            clip_image = self.clip_extractor.embed_image(image).float()
            clip_text = self.clip_extractor.embed_text(captions).float()

            # clip_image_pred, clip_text_pred, voxel_rec, loss_cyc = self.voxel2clip((voxel, subj_id))
            results = self.voxel2clip(self.input(voxel, subj_id))

            # image clip loss
            clip_image_pred = results[0]
            clip_image_pred_norm = nn.functional.normalize(clip_image_pred.flatten(1), dim=-1)
            clip_image_norm = nn.functional.normalize(clip_image.flatten(1), dim=-1)
            val_loss_clip_image = utils.soft_clip_loss(
                clip_image_pred_norm,
                clip_image_norm,
            )
            val_loss += val_loss_clip_image
            self.val_loss_clip_image_sum += val_loss_clip_image.item()

            # image mse loss
            if self.args.mse_mult:
                val_loss_mse_image = nn.MSELoss()(clip_image_pred_norm, clip_image_norm)
                val_loss += self.args.mse_mult * val_loss_mse_image
                self.val_loss_mse_image_sum += val_loss_mse_image.item()

            # text clip loss
            clip_text_pred = results[1]
            clip_text_pred_norm = nn.functional.normalize(clip_text_pred.flatten(1), dim=-1)
            clip_text_norm = nn.functional.normalize(clip_text.flatten(1), dim=-1)
            val_loss_clip_text = utils.soft_clip_loss(
                clip_text_pred_norm,
                clip_text_norm,
            )
            val_loss += val_loss_clip_text
            self.val_loss_clip_text_sum += val_loss_clip_text.item()

            # text mse loss
            if self.args.mse_mult:
                val_loss_mse_text = nn.MSELoss()(clip_text_pred_norm, clip_text_norm)
                val_loss += self.args.mse_mult * val_loss_mse_text
                self.val_loss_mse_text_sum += val_loss_mse_text.item()

            # brain reconstruction loss
            if self.args.rec_mult:
                voxel_rec = results[2]
                val_loss_rec = nn.MSELoss()(voxel, voxel_rec)
                val_loss += self.args.rec_mult * val_loss_rec
                self.val_loss_rec_sum += val_loss_rec.item()

            # cycle loss
            if self.args.cyc_mult:
                loss_cyc = results[3]
                val_loss_cyc = loss_cyc
                val_loss += self.args.cyc_mult * val_loss_cyc
                self.val_loss_cyc_sum += val_loss_cyc.item()

            utils.check_loss(val_loss)
            self.val_losses.append(val_loss.item())

            self.val_sims_image += nn.functional.cosine_similarity(clip_image_norm,clip_image_pred_norm).mean().item()
            self.val_sims_text += nn.functional.cosine_similarity(clip_text_norm,clip_text_pred_norm).mean().item()
            
            labels = torch.arange(len(clip_image_norm)).to(self.device) 
            self.val_fwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_image_pred_norm, clip_image_norm), labels, k=1)
            self.val_bwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_image_norm, clip_image_pred_norm), labels, k=1)

    def vis(self,):
        pass

    def save_ckpt(self, tag, epoch):
        ckpt_path = self.outdir+f'/{tag}.pth'
        print(f'saving {ckpt_path}',flush=True)
        unwrapped_model = self.accelerator.unwrap_model(self.voxel2clip)
        try:
            torch.save({
                'epoch': epoch,
                'model_state_dict': unwrapped_model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict(),
                'train_losses': self.losses,
                'val_losses': self.val_losses,
                'lrs': self.lrs,
                }, ckpt_path)
        except:
            print("Couldn't save... moving on to prevent crashing.")
        del unwrapped_model

    def save(self, epoch):
        self.save_ckpt(f'last', epoch)
        # save best model
        current_sim = (self.val_sims_image + self.val_sims_text) / (self.val_i + 1)
        if current_sim > self.best_sim:
            self.best_sim = current_sim
            self.best_epoch = epoch
            self.save_ckpt(f'best', epoch)
        else:
            print(f'Not best - current_similarity: {current_sim:.3f} @ epoch {epoch}, best_similarity: {self.best_sim:.3f} @ epoch {self.best_epoch}')
                
    def load(self,):
        print("\n---load from ckpt: {}---\n".format(self.args.load_from))
        checkpoint = torch.load(self.args.load_from, map_location='cpu')
        self.voxel2clip.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("loaded keys", checkpoint['model_state_dict'].keys())
        del checkpoint

    def resume(self,):
        print("\n---resuming from last.pth ckpt---\n")
        checkpoint = torch.load(self.outdir+'/last.pth', map_location='cpu')
        self.epoch_start = checkpoint['epoch']
        print("Resume at Epoch", self.epoch_start)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

        self.voxel2clip.load_state_dict(checkpoint['model_state_dict'])
        del checkpoint
    
    def log_train(self):
        self.logs = {
            "train/loss": np.mean(self.losses[-(self.train_i+1):]),
            "train/lr": self.lrs[-1],
            "train/num_steps": len(self.losses),
            "train/cosine_sim_image": self.sims_image / (self.train_i + 1),
            "train/cosine_sim_text": self.sims_text / (self.train_i + 1),
            "train/fwd_pct_correct": self.fwd_percent_correct / (self.train_i + 1),
            "train/bwd_pct_correct": self.bwd_percent_correct / (self.train_i + 1),
            "train/loss_clip_image": self.loss_clip_image_sum / (self.train_i + 1),
            "train/loss_clip_text": self.loss_clip_text_sum / (self.train_i + 1),
            "train/loss_mse_image": self.loss_mse_image_sum / (self.train_i + 1),
            "train/loss_mse_text": self.loss_mse_text_sum / (self.train_i + 1),
            "train/loss_rec": self.loss_rec_sum / (self.train_i + 1),
            "train/loss_cyc": self.loss_cyc_sum / (self.train_i + 1),
        }

    def log_val(self):
        self.logs.update({
            "val/loss": np.mean(self.val_losses[-(self.val_i+1):]),
            "val/num_steps": len(self.val_losses),
            "val/cosine_sim_image": self.val_sims_image / (self.val_i + 1),
            "val/cosine_sim_text": self.val_sims_text / (self.val_i + 1),
            "val/val_fwd_pct_correct": self.val_fwd_percent_correct / (self.val_i + 1),
            "val/val_bwd_pct_correct": self.val_bwd_percent_correct / (self.val_i + 1),
            "val/loss_clip_image": self.val_loss_clip_image_sum / (self.val_i + 1),
            "val/loss_clip_text": self.val_loss_clip_text_sum / (self.val_i + 1),
            "val/loss_mse_image": self.val_loss_mse_image_sum / (self.val_i + 1),
            "val/loss_mse_text": self.val_loss_mse_text_sum / (self.val_i + 1),
            "val/loss_rec": self.val_loss_rec_sum / (self.val_i + 1),
            "val/loss_cyc": self.val_loss_cyc_sum / (self.val_i + 1),
        })

class Trainer_single(Trainer):
    def __init__(self, args, accelerator, voxel2clip, clip_extractor, prompts_list, device) -> None:
        super().__init__(args, accelerator, voxel2clip, clip_extractor, prompts_list, device)

    def prepare_dataloader(self):
        # Prepare data and dataloader
        print("Preparing data and dataloader...")
        self.train_dl, self.val_dl = data.get_dls(
            subject=self.args.subj_list[0],
            data_path=self.args.data_path,
            batch_size=self.args.batch_size,
            val_batch_size=self.args.val_batch_size,
            num_workers=self.args.num_workers,
            pool_type=self.args.pool_type,
            pool_num=self.args.pool_num,
            length=self.args.length,
            seed=self.args.seed,
        )
        self.num_batches = len(self.train_dl)

    def prepare_multi_gpu(self):
        self.voxel2clip, self.optimizer, self.train_dl, self.val_dl, self.lr_scheduler = self.accelerator.prepare(
        self.voxel2clip, self.optimizer, self.train_dl, self.val_dl, self.lr_scheduler)
    
    def input(self, voxel, subj_id):
        # adapting need to know subj_id
        return voxel
    
    def train_epoch(self, epoch):
        # train loop
        for train_i, data_i in enumerate(self.train_dl):
            self.train_i = train_i
            repeat_index = train_i % 3 # randomly choose the one in the repeated three

            voxel, image, coco, subj_id = data_i
            voxel = voxel[:,repeat_index,...].float()
            subj_id = subj_id[[0],...]

            coco_ids = coco.squeeze().tolist()
            current_prompts_list = [self.prompts_list[coco_id] for coco_id in coco_ids]
            captions = [prompts[repeat_index]['caption'] for prompts in current_prompts_list]

            print(">>> Epoch{} | Iter{} | voxel: {}".format(epoch, train_i, voxel.shape), flush=True)
            self.train_step(voxel, image, captions, subj_id)

    def eval_epoch(self, epoch):
        print("evaluating...")
        self.voxel2clip.eval()

        for val_i, data_i in enumerate(self.val_dl): 
            self.val_i = val_i
            repeat_index = val_i % 3 # randomly choose the one in the repeated three
            voxel, image, coco, subj_id = data_i
            voxel = torch.mean(voxel,axis=1)
            subj_id = subj_id[[0],...]
        
            coco_ids = coco.squeeze().tolist()
            current_prompts_list = [self.prompts_list[coco_id] for coco_id in coco_ids]
            captions = [prompts[repeat_index]['caption'] for prompts in current_prompts_list]

            print(">>> Epoch{} | Eval{} | voxel: {}".format(epoch, val_i, voxel.shape), flush=True)
            self.eval_step(voxel, image, captions, subj_id)

class Trainer_bridge(Trainer):
    def __init__(self, args, accelerator, voxel2clip, clip_extractor, prompts_list, device) -> None:
        super().__init__(args, accelerator, voxel2clip, clip_extractor, prompts_list, device)

    def prepare_dataloader(self):
        # Prepare data and dataloader
        print("Preparing data and dataloader...")
        self.train_dls = [] # tarin_dls contains all subjects separately
        self.val_dls = [] # tarin_dls contains all subjects separately

        for subj in self.args.subj_list:
            train_dl, val_dl = data.get_dls(
                subject=subj,
                data_path=self.args.data_path,
                batch_size=self.args.batch_size,
                val_batch_size=self.args.val_batch_size,
                num_workers=self.args.num_workers,
                pool_type=self.args.pool_type,
                pool_num=self.args.pool_num,
                length=self.args.length,
                seed=self.args.seed,
            )
            self.train_dls.append(train_dl)
            self.val_dls.append(val_dl)
        
        self.num_batches = len(self.train_dls[0])

    def prepare_multi_gpu(self):
        self.voxel2clip, self.optimizer, self.lr_scheduler, _ = self.accelerator.prepare(
        self.voxel2clip, self.optimizer, self.lr_scheduler, self.train_dls[0])

        for i, dls in enumerate(zip(self.train_dls, self.val_dls)):
            train_dl, val_dl = dls
            self.train_dls[i] = self.accelerator.prepare(train_dl)
            self.val_dls[i] = self.accelerator.prepare(val_dl)

    def train_epoch(self, epoch):
        # train loop
        for train_i, datas in enumerate(zip(*self.train_dls)):
            self.train_i = train_i
            repeat_index = train_i % 3 # randomly choose the one in the repeated three
            
            # ensemble data from multiple subjects
            voxel_list, image_list, coco_list, subj_id_list = [], [], [], []
            for voxel, image, coco, subj_id in datas:
                voxel_list.append(voxel[:,repeat_index,...])
                image_list.append(image)
                coco_list.append(coco)
                subj_id_list.append(subj_id[[0],...])
            voxel = torch.cat(voxel_list, dim=0)
            image = torch.cat(image_list, dim=0)
            coco = torch.cat(coco_list, dim=0)
            subj_id = torch.cat(subj_id_list, dim=0)

            coco_ids = coco.squeeze().tolist()
            current_prompts_list = [self.prompts_list[coco_id] for coco_id in coco_ids]
            captions = [prompts[repeat_index]['caption'] for prompts in current_prompts_list]

            print(">>> Epoch{} | Iter{} | voxel: {}".format(epoch, train_i, voxel.shape), flush=True)
            self.train_step(voxel, image, captions, subj_id)

    def eval_epoch(self, epoch):
        print("Evaluating...")
        self.voxel2clip.eval()
        for val_i, datas in enumerate(zip(*self.val_dls)): 
            self.val_i = val_i
            repeat_index = val_i % 3 # randomly choose the one in the repeated three

            # ensemble data from multiple subjects
            voxel_list, image_list, coco_list, subj_id_list = [], [], [], []
            for voxel, image, coco, subj_id in datas:
                voxel_list.append(torch.mean(voxel,axis=1))
                image_list.append(image)
                coco_list.append(coco)
                subj_id_list.append(subj_id[[0],...])
            voxel = torch.cat(voxel_list, dim=0)
            image = torch.cat(image_list, dim=0)
            coco = torch.cat(coco_list, dim=0)
            subj_id = torch.cat(subj_id_list, dim=0)

            coco_ids = coco.squeeze().tolist()
            current_prompts_list = [self.prompts_list[coco_id] for coco_id in coco_ids]
            captions = [prompts[repeat_index]['caption'] for prompts in current_prompts_list]

            print(">>> Epoch{} | Eval{} | voxel: {}".format(epoch, val_i, voxel.shape), flush=True)
            self.eval_step(voxel, image, captions, subj_id)

class Trainer_adapt(Trainer):
    def __init__(self, args, accelerator, voxel2clip, clip_extractor, prompts_list, device) -> None:
        super().__init__(args, accelerator, voxel2clip, clip_extractor, prompts_list, device)

    def prepare_dataloader(self):
        # Prepare data and dataloader
        print("Preparing data and dataloader...")
        self.train_dls_source = [] # tarin_dls contains all subjects separately
        self.val_dls_source = [] # tarin_dls contains all subjects separately

        # source subjects
        for subj in self.args.subj_source:
            train_dl, val_dl = data.get_dls(
                subject=subj,
                data_path=self.args.data_path,
                batch_size=self.args.batch_size,
                val_batch_size=self.args.val_batch_size,
                num_workers=self.args.num_workers,
                pool_type=self.args.pool_type,
                pool_num=self.args.pool_num,
                length=self.args.length,
                seed=self.args.seed,
            )
            self.train_dls_source.append(train_dl)
            self.val_dls_source.append(val_dl)

        # target subjects
        self.train_dl_target, self.val_dl_target = data.get_dls(
            subject=self.args.subj_target,
            data_path=self.args.data_path,
            batch_size=self.args.batch_size,
            val_batch_size=self.args.val_batch_size,
            num_workers=self.args.num_workers,
            pool_type=self.args.pool_type,
            pool_num=self.args.pool_num,
            length=self.args.length,
            seed=self.args.seed,
        )

        self.num_batches = len(self.train_dl_target)

    def prepare_multi_gpu(self):
        self.voxel2clip, self.optimizer, self.lr_scheduler, self.train_dl_target, self.val_dl_target = self.accelerator.prepare(
        self.voxel2clip, self.optimizer, self.lr_scheduler, self.train_dl_target, self.val_dl_target)

        for i, dls in enumerate(zip(self.train_dls_source, self.val_dls_source)):
            train_dl, val_dl = dls
            self.train_dls_source[i] = self.accelerator.prepare(train_dl)
            self.val_dls_source[i] = self.accelerator.prepare(val_dl)

    def train_epoch(self, epoch):
        # enable iteratable
        train_dls_source_iter = []
        for train_dl_s in self.train_dls_source:
            train_dls_source_iter.append(iter(train_dl_s))

        # train loop
        for train_i, datas_target in enumerate(self.train_dl_target):
            self.train_i = train_i
            repeat_index = train_i % 3 # randomly choose the one in the repeated three
            voxel_target, image, coco, subj_id = datas_target
            voxel = voxel_target[:,repeat_index,...]

            source_index = train_i % len(train_dls_source_iter) # every time choose one source domain
            voxel_source, image_source, coco_source, subj_id_source = next(train_dls_source_iter[source_index])
            voxel_source = voxel_source[:,repeat_index,...]
            voxel = torch.cat((voxel_source, voxel), dim=0)
            image = torch.cat((image_source, image), dim=0)
            coco = torch.cat((coco_source, coco), dim=0)
            subj_id = torch.cat((subj_id_source[[0],...], subj_id[[0],...]), dim=0)

            coco_ids = coco.squeeze().tolist()
            current_prompts_list = [self.prompts_list[coco_id] for coco_id in coco_ids]
            captions = [prompts[repeat_index]['caption'] for prompts in current_prompts_list]

            print(">>> Epoch{} | Iter{} | source{} | voxel: {}".format(epoch, train_i, source_index, voxel.shape), flush=True)
            self.train_step(voxel, image, captions, subj_id)
    
    def eval_epoch(self, epoch):
        print("Evaluating...")
        self.voxel2clip.eval()

        # enable iteratable
        val_dls_source_iter = []
        for val_dl_s in self.val_dls_source:
            val_dls_source_iter.append(iter(val_dl_s))

        for val_i, datas_target in enumerate(self.val_dl_target): 
            self.val_i = val_i
            repeat_index = val_i % 3 # randomly choose the one in the repeated three
            voxel, image, coco, subj_id = datas_target
            voxel = torch.mean(voxel,axis=1)

            source_index = val_i % len(val_dls_source_iter) # every time choose one source domain
            print("Using source {}".format(source_index))
            voxel_source, image_source, coco_source, subj_id_source = next(val_dls_source_iter[source_index])
            voxel_source = torch.mean(voxel_source, axis=1)
            voxel = torch.cat((voxel_source, voxel), dim=0)
            image = torch.cat((image_source, image), dim=0)
            coco = torch.cat((coco_source, coco), dim=0)
            subj_id = torch.cat((subj_id_source[[0],...], subj_id[[0],...]), dim=0)

            coco_ids = coco.squeeze().tolist()
            current_prompts_list = [self.prompts_list[coco_id] for coco_id in coco_ids]
            captions = [prompts[repeat_index]['caption'] for prompts in current_prompts_list]

            print(">>> Epoch{} | Eval{} | voxel: {}".format(epoch, val_i, voxel.shape), flush=True)
            self.eval_step(voxel, image, captions, subj_id)            

