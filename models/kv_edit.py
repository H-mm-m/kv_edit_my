from dataclasses import dataclass
#tensor reshape工具
from einops import rearrange,repeat
import torch
import torch.nn.functional as F
from torch import Tensor
#flux自己的函数：get_schedule:上次时间步；unpack：token->image;denoise_kv:带kvcache的扩散；denoise_kv_inf:带kvcache的扩散，推理模式
from flux.sampling import get_schedule, unpack,denoise_kv,denoise_kv_inf
from flux.util import load_flow_model
from flux.model import Flux_kv

#加载flux模型：记得加载支持kv-cache的flux模型，huggingface上有多个版本
class only_Flux(torch.nn.Module): 
    def __init__(self, device,name='flux-dev'):
        self.device = device
        self.name = name
        super().__init__()
        self.model = load_flow_model(self.name, device=self.device,flux_cls=Flux_kv)

    #创建注意力掩码
    # ：seq_len:text+image token长度；mask_indices:mask token索引；text_len:text token长度；device:设备    
    def create_attention_mask(self,seq_len, mask_indices, text_len=512, device='cuda'):
        #,表示token_i能否查看token_j的值
        attention_mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)

        text_indices = torch.arange(0, text_len, device=device)
        #获取mask部分的索引，由于imagetoken在text后面，所以要加上text_len
        mask_token_indices = torch.tensor([idx + text_len for idx in mask_indices], device=device)
        #获取输入image的完整索引（text_len--seq_len)之间
        all_indices = torch.arange(text_len, seq_len, device=device)
        #获取背景部分的索引
        background_token_indices = torch.tensor([idx for idx in all_indices if idx not in mask_token_indices])

        # text setting
        #text_indices.unsqueeze(1)表示text_indices变成竖着的；.expand->把每个元素都扩展到seq_len长度
        #[0,1,2,3]-[[0],[1],[2],[3]]->[[0000...seqlen-1],[1000...seqlen-1],[2000...seqlen-1],[3000...seqlen-1]]
        #attention_mask[i in text_indices , :]=True
        attention_mask[text_indices.unsqueeze(1).expand(-1, seq_len)] = True
        attention_mask[text_indices.unsqueeze(1), text_indices] = True
        attention_mask[text_indices.unsqueeze(1), background_token_indices] = True 

        
        # mask setting
        # attention_mask[mask_token_indices.unsqueeze(1), background_token_indices] = True 
        attention_mask[mask_token_indices.unsqueeze(1), text_indices] = True  
        attention_mask[mask_token_indices.unsqueeze(1), mask_token_indices] = True 

        # background setting
        # attention_mask[background_token_indices.unsqueeze(1), mask_token_indices] = True  
        attention_mask[background_token_indices.unsqueeze(1), text_indices] = True 
        attention_mask[background_token_indices.unsqueeze(1), background_token_indices] = True  

        return attention_mask.unsqueeze(0)

    def create_attention_scale(self,seq_len, mask_indices, text_len=512, device='cuda',scale = 0):

        attention_scale = torch.zeros(1, seq_len, dtype=torch.bfloat16, device=device) # 相加时广播


        text_indices = torch.arange(0, text_len, device=device)
        
        mask_token_indices = torch.tensor([idx + text_len for idx in mask_indices], device=device)

        all_indices = torch.arange(text_len, seq_len, device=device)
        background_token_indices = torch.tensor([idx for idx in all_indices if idx not in mask_token_indices])
        #创建attention bias：背景部分的注意力权重
        attention_scale[0, background_token_indices] = scale #
        
        return attention_scale.unsqueeze(0)
     
class Flux_kv_edit_inf(only_Flux):
    def __init__(self, device,name):
        super().__init__(device,name)

    @torch.inference_mode()
    def forward(self,inp,inp_target,mask:Tensor,opts):
       
        info = {}
        info['feature'] = {}
        bs, L, d = inp["img"].shape
        h = opts.height // 8
        w = opts.width // 8
        mask = F.interpolate(mask, size=(h,w), mode='bilinear', align_corners=False)
        mask[mask > 0] = 1
        
        mask = repeat(mask, 'b c h w -> b (repeat c) h w', repeat=16)
        mask = rearrange(mask, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        info['mask'] = mask
        bool_mask = (mask.sum(dim=2) > 0.5)
        info['mask_indices'] = torch.nonzero(bool_mask)[:,1] 
        #单独分离inversion
        if opts.attn_mask and (~bool_mask).any():
            attention_mask = self.create_attention_mask(L+512, info['mask_indices'], device=self.device)
        else:
            attention_mask = None   
        info['attention_mask'] = attention_mask
        
        if opts.attn_scale != 0 and (~bool_mask).any():
            attention_scale = self.create_attention_scale(L+512, info['mask_indices'], device=mask.device,scale = opts.attn_scale)
        else:
            attention_scale = None
        info['attention_scale'] = attention_scale
        
        denoise_timesteps = get_schedule(opts.denoise_num_steps, inp["img"].shape[1], shift=(self.name != "flux-schnell"))
        denoise_timesteps = denoise_timesteps[opts.skip_step:]
    
        z0 = inp["img"]

        with torch.no_grad():
            info['inject'] = True
            z_fe, info = denoise_kv_inf(self.model, img=inp["img"], img_ids=inp['img_ids'], 
                                    source_txt=inp['txt'], source_txt_ids=inp['txt_ids'], source_vec=inp['vec'],
                                    target_txt=inp_target['txt'], target_txt_ids=inp_target['txt_ids'], target_vec=inp_target['vec'],
                                    timesteps=denoise_timesteps, source_guidance=opts.inversion_guidance, target_guidance=opts.denoise_guidance,
                                    info=info)
        mask_indices = info['mask_indices'] 
     
        z0[:, mask_indices,...] = z_fe

        z0 = unpack(z0.float(),  opts.height, opts.width)
        del info
        return z0

class Flux_kv_edit(only_Flux):
    def __init__(self, device,name):
        super().__init__(device,name)
    
    #推理模式装饰器，相当于with torch.no_grad():关闭梯度加速推理
    @torch.inference_mode()
    def forward(self,inp,inp_target,mask:Tensor,opts):
        #inp:输入的image；inp_target:目标表述文本text；mask:掩码；opts:选项
        z0,zt,info = self.inverse(inp,mask,opts)
        z0 = self.denoise(z0,zt,inp_target,mask,opts,info)
        return z0
    @torch.inference_mode()
    def inverse(self,inp,mask,opts):
        #info字典，用于存储，其实就是kv-cache
        info = {}
        info['feature'] = {}
        #bs=batch;L=image token长度；d=image token维度
        bs, L, d = inp["img"].shape
        h = opts.height // 8
        w = opts.width // 8

        if opts.attn_mask:
            #把mask resize到latent分辨率
            mask = F.interpolate(mask, size=(h,w), mode='bilinear', align_corners=False)
            mask[mask > 0] = 1
            #把mask重复16次，变成16个channel
            mask = repeat(mask, 'b c h w -> b (repeat c) h w', repeat=16)
            #把mask reshape到(h*w,c*ph*pw),ph,pw是patchify=2*2
            mask = rearrange(mask, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
            #把mask reshape到(h*w,c*ph*pw)后，计算每个token的mask值，如果mask值大于0.5，则认为该token被mask
            bool_mask = (mask.sum(dim=2) > 0.5)
            #把bool_mask中为True的token索引提取出来
            mask_indices = torch.nonzero(bool_mask)[:,1] 
            
            #单独分离inversion
            assert not (~bool_mask).all(), "mask is all false"
            assert not (bool_mask).all(), "mask is all true"
            #调用了上面继承的函数create_attention_mask，创建注意力掩码
            attention_mask = self.create_attention_mask(L+512, mask_indices, device=mask.device)
            info['attention_mask'] = attention_mask
    
        
        denoise_timesteps = get_schedule(opts.denoise_num_steps, inp["img"].shape[1], shift=(self.name != "flux-schnell"))
        #跳过前opts.skip_step步，因为前opts.skip_step步是预热阶段，不需要计算
        denoise_timesteps = denoise_timesteps[opts.skip_step:]
        
        # 加噪过程
        z0 = inp["img"].clone()        
        info['inverse'] = True
        zt, info = denoise_kv(self.model, **inp, timesteps=denoise_timesteps, guidance=opts.inversion_guidance, inverse=True, info=info)
        return z0,zt,info
    
    @torch.inference_mode()
    def denoise(self,z0,zt,inp_target,mask:Tensor,opts,info):
        
        h = opts.height // 8
        w = opts.width // 8
        L = h * w // 4 
        mask = F.interpolate(mask, size=(h,w), mode='bilinear', align_corners=False)
        mask[mask > 0] = 1
        
        mask = repeat(mask, 'b c h w -> b (repeat c) h w', repeat=16)
      
        mask = rearrange(mask, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        info['mask'] = mask
        bool_mask = (mask.sum(dim=2) > 0.5)
        info['mask_indices'] = torch.nonzero(bool_mask)[:,1]
        
        denoise_timesteps = get_schedule(opts.denoise_num_steps, inp_target["img"].shape[1], shift=(self.name != "flux-schnell"))
        denoise_timesteps = denoise_timesteps[opts.skip_step:]
       
        mask_indices = info['mask_indices']
        if opts.re_init:
            noise = torch.randn_like(zt)
            t  = denoise_timesteps[0]
            zt_noise = z0 *(1 - t) + noise * t
            inp_target["img"] = zt_noise[:, mask_indices,...]
        else:
            img_name = str(info['t']) + '_' + 'img'
            zt = info['feature'][img_name].to(zt.device)
            inp_target["img"] = zt[:, mask_indices,...]
            
        if opts.attn_scale != 0 and (~bool_mask).any():
            attention_scale = self.create_attention_scale(L+512, mask_indices, device=mask.device,scale = opts.attn_scale)
        else:
            attention_scale = None
        info['attention_scale'] = attention_scale

        info['inverse'] = False
        x, _ = denoise_kv(self.model, **inp_target, timesteps=denoise_timesteps, guidance=opts.denoise_guidance, inverse=False, info=info)
       
        z0[:, mask_indices,...] = z0[:, mask_indices,...] * (1 - info['mask'][:, mask_indices,...]) + x * info['mask'][:, mask_indices,...]
        
        z0 = unpack(z0.float(),  opts.height, opts.width)
        del info
        return z0