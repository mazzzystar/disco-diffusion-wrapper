# disco-diffusion-wrapper
My implementation of [disco diffusion](https://github.com/alembics/disco-diffusion) wrapper that could run your own GPU with a batch  of input text. 


## What this repo did
1.Separate the model loading and model inference parts of the initial code. Now you can use it like:
```python
# init and load pretrain model.
disco = DiscoDiffusion()
# infer
disco.draw(text, ...)
```

2.Use [deepL](https://www.deepl.com/translator) to preprocess the text, so that you can use any language you like to draw.
The default language pair is from `Chinese`->`English`(`ZH`->`EN-US`)，you can change it in `run.py` or `run_batch.py`
```python
def translate(text, source_lang="ZH", target_lang="EN-US"):
    res = translator.translate_text(text, source_lang="ZH", target_lang="EN-US")
    return res
```

3.Batch generating & saving.
As the loading part and the inferring part is seperated, you do not need to load pretrain model again for a new sentence. This reduces a bunch of time.
```python
cd disco-diffusion-wrapper/wrapper
# create and fill the sentence.list
python run_batch.py
```
The output images is saved by the name of the  origin text, so you can easily find it even after a deepL.


## Usage
```bash
git clone git@github.com:mazzzystar/disco-diffusion-wrapper.git
cd disco-diffusion-wrapper

# download pretrain model
python wrapper/utils.py

cd wrapper

# generated image by one sentence
python run.py "一行白鹭上青天"

# batch generation
touch sentence.list
# fill your text in, one by a line.
python batch_run.py
```

## Samples
Below are some samples generated using this repo. 
> 半亩方塘一鉴开，天光云影共徘徊，水墨画

![](samples/A_half-acre_square_pond_is_open_the_sky_and_clouds_wandering_together_ink_painting.png)
> 雪山，平原，幽鬼军队在夜里打着荧光绿色的火把前行，抽象画，Trending on artstation

![](samples/Snowy_mountains_plains_army_of_ghosts_marching_at_night_with_fluorescent_green_torches_abstract_painting_Trending_on_artstation.png)

If you use this project and produced some interesting results, submissions are welcomed.



## Problems
As it's for fun, I did not look much into the details, and deleted many features(such as VR/3D/Video ...) to make me faster and more clear on the project resonctruction. It's awesome if you're interested in restoring the original function, PR is wellcomed.



