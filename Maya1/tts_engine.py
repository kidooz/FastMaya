import torch
import librosa
import numpy as np
from snac import SNAC
from FastAudioSR import FASR
from huggingface_hub import snapshot_download
from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig
from Maya1.utils import extract_snac_codes, unpack_snac_from_7

class TTSEngine:
    """
    Uses LMdeploy to run maya-1 with great speed
    """

    def __init__(self, memory_util = 0.3, tp = 1, enable_prefix_caching = True, quant_policy = 8):
        """
        Initializes the model configuration.

        Args:
            memory_util (float): Target fraction of GPU memory usage (0.0 to 1.0). Default: 0.3
            tp (int): Number of Tensor Parallel (TP) ranks. Use for multiple gpus. Default: 1
            enable_prefix_caching (bool): If True, cache input prefixes. Use for batching. Default: True
            quant_policy (int): KV cache quant bit-width (e.g., 8 or None). Saves vram at slight quality cost. Default: 8
        """
        backend_config = TurbomindEngineConfig(cache_max_entry_count=memory_util, tp=tp, enable_prefix_caching=enable_prefix_caching, quant_policy=quant_policy)
        upsampler_path = snapshot_download("YatharthS/FlashSR")
        self.pipe = pipeline("maya-research/maya1", backend_config=backend_config)
        self.snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to("cuda")
        self.upsampler = FASR(f"{upsampler_path}/upsampler.pth")
        _ = self.upsampler.model.half()
        self.gen_config = GenerationConfig(top_p=0.9,
                              top_k=40,
                              temperature=0.4,
                              max_new_tokens=1024,
                              repetition_penalty=1.4,
                              stop_token_ids=[128258],
                              do_sample=True,
                              min_p=0.0
                              )
        

    # 3. An instance method (operates on the object's data)
    def format_prompt(self, true_prompt, voice):
        """
        Formats the prompt for the TTS llm model

        Args:
            true_prompt (str): Text to speak. 
            voice (str): description of voice to use
        """
        
        prompt = f'<custom_token_3><|begin_of_text|><description="{voice}"> {true_prompt}<|eot_id|><custom_token_4><custom_token_5><custom_token_1>'
        return prompt
        
    # 4. Another instance method (modifies the object's state)
    def decode_audio(self, tokens, batched=False):
        """
        Decodes audio from snac tokens

        Args:
            tokens (list/str): List or str of tokens to decode
            batched (bool): To decode tokens as list or single string
        """
        audio = []
        device = 'cuda:0'
        if batched:
            for token in tokens:
                snac_tokens = extract_snac_codes(token)
                levels = unpack_snac_from_7(snac_tokens)
                codes_tensor = [
                    torch.tensor(level, dtype=torch.long, device=device).unsqueeze(0)
                    for level in levels
                ]
                with torch.inference_mode():
                    z_q = self.snac_model.quantizer.from_codes(codes_tensor)
                    audio1 = self.snac_model.decoder(z_q)[0, 0].cpu().numpy()
                    audio16k = librosa.resample(y=audio1, orig_sr=24000, target_sr=16000, res_type='soxr_hq')
                    audio16k = torch.from_numpy(audio16k).unsqueeze(0).to("cuda:0").half()
                    audio1 = self.upsampler.run(audio16k).cpu().numpy()
                    audio.append(audio1)
            audio = np.concatenate(audio)
            
        else:
            snac_tokens = extract_snac_codes(tokens)
            levels = unpack_snac_from_7(snac_tokens)
            codes_tensor = [
                torch.tensor(level, dtype=torch.long, device=device).unsqueeze(0)
                for level in levels
            ]
            with torch.inference_mode():
                z_q = self.snac_model.quantizer.from_codes(codes_tensor)
                audio = self.snac_model.decoder(z_q)[0, 0].cpu().numpy()
                audio16k = librosa.resample(y=audio, orig_sr=24000, target_sr=16000, res_type='soxr_hq')
                audio16k = torch.from_numpy(audio16k).unsqueeze(0).to("cuda:0").half()
                audio = self.upsampler.run(audio16k).cpu().numpy()

        return audio
        
    def generate(self, prompt, voice):
        """
        Generates speech from text, for single batch size

        Args:
            prompt (str): Input for tts model
            voice (str): Description of voice
        """
        formatted_prompt = self.format_prompt(prompt, voice)
        responses = self.pipe([formatted_prompt], gen_config=self.gen_config, do_preprocess=False)
        generated_tokens = responses[0].token_ids
        audio = self.decode_audio(generated_tokens)
        return audio

    def batch_generate(self, prompts, voices):
        """
        Generates speech from text, for larger batch size

        Args:
            prompt (list): Input for tts model, list of prompts
            voice (list): Description of voice, list of voices respective to prompt
        """
        formatted_prompts = []
        for prompt, voice in zip(prompts, voices):
            formatted_prompt = self.format_prompt(prompt, voice)
            formatted_prompts.append(formatted_prompt)
        
        responses = self.pipe(formatted_prompts, gen_config=self.gen_config, do_preprocess=False)
        generated_tokens = [response.token_ids for response in responses]
        audios = self.decode_audio(generated_tokens, batched=True)
        return audios
