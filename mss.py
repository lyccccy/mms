from model.dac import DAC
from torch import nn
import torch
import numpy as np
from audiotools import AudioSignal
from model.demucs import Demucs


def d_axis_distill_loss(feature, target_feature):
    n = min(feature.size(1), target_feature.size(1))
    distill_loss = - torch.log(torch.sigmoid(torch.nn.functional.cosine_similarity(feature[:, :n], target_feature[:, :n], axis=1))).mean()
    return distill_loss

def t_axis_distill_loss(feature, target_feature, lambda_sim=1):
    n = min(feature.size(1), target_feature.size(1))
    l1_loss = torch.functional.l1_loss(feature[:, :n], target_feature[:, :n], reduction='mean')
    sim_loss = - torch.log(torch.sigmoid(torch.nn.functional.cosine_similarity(feature[:, :n], target_feature[:, :n], axis=-1))).mean()
    distill_loss = l1_loss + lambda_sim * sim_loss
    return distill_loss 

class MSS(nn.Module):
    def __init__(self,
                 mode : str = None,
                 audio_channel = 2,
                 sample_rate = 44100,                 
                ) -> None:
        super().__init__()

        self.audio_channel = audio_channel
        self.sample_rate = sample_rate
        self.mode = mode
        self.model_dac = DAC(sample_rate=sample_rate)
        self.hop_length = np.prod(self.model_dac.encoder_rates)


        if mode == "concat":
            self.Linear = nn.Linear(2*self.hop_length,self.hop_length)
        elif mode == "add":
            self.Linear = nn.Linear(self.hop_length,self.hop_length)



    def forward(self,x:torch.Tensor,s:torch.Tensor = None,length :int = None):
        if length == None:
            length = x.shape[-1]
        #pre_audio =  model_dac.preprocess(x,sample_rate=44100)

        encoded_dac  = self.model_dac.encoder(x)
        #print(encoded_dac.shape)
        #print(s.shape)

        
        if self.mode == "concat":
            combined = torch.concat((encoded_dac ,s),dim=1)
            combined = combined.permute(2, 0, 1)
            combined = self.Linear(combined)
            combined = combined.permute(1, 2, 0)

        elif self.mode == "add":
            encoded_dac = encoded_dac.permute(2,0,1)
            encoded_dac=self.Linear(encoded_dac)
            encoded_dac =encoded_dac.permute(1,2,0)

            s = s.permute(2,0,1)
            s=self.Linear(s)
            s =s.permute(1,2,0)
            
            combined = torch.add(encoded_dac,s)
        
        elif self.mode=='distill' or self.mode=='dac':
            combined = encoded_dac

        z , sematic,codes, latents, commitment_loss, codebook_loss = self.model_dac.quantizer(combined)
        z = self.model_dac.decode(z)

        if self.mode=='distill':
            distill_loss = d_axis_distill_loss(sematic, s)
        else:
            distill_loss = 0

        

        # 只在时间维度（最后一个维度）填充
        if self.mode=='distill':
            return {
                "audio":z[...,:length],
                "sematic":sematic,
                "codes": codes,
                "latents": latents,
                "vq/commitment_loss": commitment_loss,
                "vq/codebook_loss": codebook_loss,
                "distill_loss": distill_loss,
            }
        else:
            return {
                "audio":z[...,:length],
                "sematic":sematic,
                "codes": codes,
                "latents": latents,
                "vq/commitment_loss": commitment_loss,
                "vq/codebook_loss": codebook_loss,
            }
    


if __name__ == "__main__":
    import numpy as np
    from functools import partial
    teacher = Demucs()
    device = 'cpu'
    model = MSS(mode = 'distill').to(device)
    teacher = teacher.to(device)

    for n, m in model.named_modules():
        o = m.extra_repr()
        p = sum([np.prod(p.size()) for p in m.parameters()])
        fn = lambda o, p: o + f" {p/1e6:<.3f}M params."
        setattr(m, "extra_repr", partial(fn, o=o, p=p))
    print(model)
    print("Total # of params: ", sum([np.prod(p.size()) for p in model.parameters()]))

    length = 88200 * 2
    x = torch.randn(4, 2, length).to(device)
    x.requires_grad_(True)
    x.retain_grad()
    pre=model.model_dac.preprocess(x,sample_rate=44100)
    semetic = teacher.encode(pre)
    # Make a forward pass
    out = model(pre,semetic)["audio"]
    print("Input shape:", x.shape)
    print("Output shape:", out.shape)

    # Create gradient variable
    grad = torch.zeros_like(out)
    grad[:, :, grad.shape[-1] // 2] = 1

    # Make a backward pass
    out.backward(grad)

    # Check non-zero values
    gradmap = x.grad.squeeze(0)
    gradmap = (gradmap != 0).sum(0)  # sum across features
    rf = (gradmap != 0).sum()

    print(f"Receptive field: {rf.item()}")

    x = AudioSignal(torch.randn(2, 1, int(44100 * 1.24)), 44100)
    out = model(x.audio_data)
    # model.decompress(model.compress(x, verbose=True), verbose=True)