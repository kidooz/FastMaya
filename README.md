# FastMaya1

FastMaya1 is a highly efficent repo to generate minutes of audios in just seconds!
[Maya1](https://huggingface.co/maya-research/maya1) is a great 3b llm based tts model that accepts voice descriptions to generate realistic speech. 
This repo optimizes it by using the highly efficent [LMdeploy](https://github.com/InternLM/lmdeploy.git) library and a custom fast and high quality upsampler.

## Key Improvements in this Repo
* 4x faster then raw transformers without batching and over 10x faster with batching. Roughly 2x faster then the vllm implementation. This repo can generate 50 seconds of audio in bold 1 second using batching!
* Custom AudioSR model to upsample 24khz audio to 48khz which considerably improves quality of audio
* Works directly out of the box in windows while vllm does not
* Memory efficent: works within 8gb vram gpus although might require more vram for larger paragraphs
* Works with multiple gpus using tensor parallel to further improve speed


Speed was tested on A100(40gb vram)
- Input text can be found in `test.txt`
- RTF: 0.02(50x faster then realtime)

## Usage

Installation(requires pip and git, uv is optional although recommended):

```
uv pip install git+https://github.com/ysharma3501/FastMaya1.git
```

Running the model(for single batch sizes):

```python
from IPython.display import Audio
from Maya1.tts_engine import TTSEngine

memory_util = 0.3 ## will use 30% of your gpu vram
tp = 1 ## change to how many gpus you have, will increase speed

tts_engine = TTSEngine(memory_util=memory_util, tp=tp)

text = "Wow. This place looks even better than I imagined. How did they set all this up so perfectly? The lights, the music, everything feels magical. I can't stop smiling right now."
voice = "Female, in her 30s with an American accent and is an event host, energetic, clear diction"

audio = tts_engine.generate(text, voice)
display(Audio(audio, rate=48000))
```


To run the model using batching:

```python
text = ["Wow. This place looks even better than I imagined. How did they set all this up so perfectly? The lights, the music, everything feels magical. I can't stop smiling right now.", "Welcome back to another episode of our podcast! <laugh_harder> Today we are diving into an absolutely fascinating topic!"]
voice = ["Female, in her 30s with an American accent and is an event host, energetic, clear diction", "Dark villain character, Male voice in their 40s with a British accent. low pitch, gravelly timbre, slow pacing, angry tone at high intensity."]

audio = tts_engine.batch_generate(text, voice)
display(Audio(audio, rate=48000))
```

It is important to note, larger batch sizes lead to larger speedups. Theoretically, the model could generate 120 seconds of audio in 1 second but real world speed is usually lower. The speedup is much less in single batch size scenarios hence Maya1 is only around 1.8x realtime on A100 hardware for such scenarios. This will be significantly improved on when awq quant is implemented. For large texts that are many sentences long, Maya1 is much faster, and can reach 50x realtime speeds as stated before.


## Next priorities
- [ ] torch compile the upsampler
- [ ] fast streaming generation(lmdeploy has 2-3x lower latency then vllm)
- [ ] awq quant support(would enable 2x even faster inference especially for single batch size scenarios)
- [ ] async inference and batching
- [ ] support for neutts-air(much faster) and other llm based tts models




