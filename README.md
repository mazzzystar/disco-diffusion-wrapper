[English](#english) 
____

# disco-diffusion-wrapper 

本项目为[disco diffusion](https://github.com/alembics/disco-diffusion)的封装实现，使其能够在您的GPU上批量处理输入文本运行。

 基于此封装工作，我们构建了一个AI绘画网站（https://6pen.art/），欢迎您前去体验。

## 此仓库实现的功能

1. 将原始代码中的模型加载和模型推理部分分离。现在您可以像下面这样使用它：

```python
# 初始化并加载预训练模型。
disco = DiscoDiffusion()
# 进行推理
disco.draw(text, ...)
```

2. 引入了[DeepL](https://www.deepl.com/translator)对文本进行预处理，因此您可以使用任何自己喜欢的语言进行绘图。默认的语言对是从`中文`至`英语`（`ZH`到`EN-US`），您可以在`run.py`或`run_batch.py`中更改这一设置。

```python
def translate(text, source_lang="ZH", target_lang="EN-US"):
    res = translator.translate_text(text, source_lang="ZH", target_lang="EN-US")
    return res
```

同时，您也可以通过设置`USE_TRANSLATE=False`来关闭文本翻译功能，这样一来就不需要DeepL的authKey了。

3. 批量生成与保存功能。
   由于加载部分和推理部分已分离，在处理新句子时无需再次加载预训练模型，从而大大节省了时间。

```python
cd disco-diffusion-wrapper/wrapper
# 创建并填充sentence.list文件
python run_batch.py
```

输出的图像按照原始文本的名字命名，因此即使在经过DeepL翻译之后，您也能轻松找到对应的图像。


## 使用方法

```bash
# 克隆项目仓库
git clone git@github.com:mazzzystar/disco-diffusion-wrapper.git
cd disco-diffusion-wrapper/

# 创建并激活名为disco的conda环境
conda create --name disco
conda activate disco

# 安装依赖包
pip install -r requirements.txt
cd wrapper/

# 下载预训练模型检查点
# 如果这一步骤下载速度较慢，您可以手动从mutils.py代码中的URL下载，并将这些检查点放在相应的文件夹内
python mutils.py

# 获取DeepL API密钥，然后将其添加到run.py或batch_run.py中替换以下内容：
USE_TRANSLATE=True
auth_key = ""

# 通过一句话生成图像
python run.py "一行白鹭上青天"

# 批量生成
touch sentence.list
# 在该文件中逐行填入你的文本内容
python run_batch.py
```

## 示例

以下是使用本仓库生成的一些示例作品链接：

- ![samples/A_half-acre_square_pond_is_open_the_sky_and_clouds_wandering_together_ink_painting.png](samples/A_half-acre_square_pond_is_open_the_sky_and_clouds_wandering_together_ink_painting.png)

  > 描述：半亩方塘一鉴开，天光云影共徘徊，水墨画风格

- ![samples/Snowy_mountains_plains_army_of_ghosts_marching_at_night_with_fluorescent_green_torches_abstract_painting_Trending_on_artstation.png](samples/Snowy_mountains_plains_army_of_ghosts_marching_at_night_with_fluorescent_green_torches_abstract_painting_Trending_on_artstation.png)

  > 描述：雪山、平原，幽灵军队夜间手持荧光绿火把行进，抽象画风，在Artstation上热门

- ![samples/Glass_whale_lying_in_ruins_abstract_painting_Trending_on_artstation.png](samples/Glass_whale_lying_in_ruins_abstract_painting_Trending_on_artstation.png)

  > 描述：废墟中躺着玻璃鲸鱼，抽象画

- ![samples/明月松间照清泉石上流中国画.png](samples/明月松间照清泉石上流中国画.png)

  > 描述：明月松间照，清泉石上流，中国画风格

- ![samples/Effie_castle.jpeg](samples/Effie_castle.jpeg)

  > 描述：精灵城堡

- ![samples/山随平野尽江入大荒流Trending_on_artstation.png](samples/山随平野尽江入大荒流Trending_on_artstation.png)

  > 描述：山随平野尽，江入大荒流，Artstation热门作品

您可以通过[这里](https://drive.google.com/file/d/1OIsupQqMaYYWu4B0eemUWgvPfTSGyaqf/view?usp=sharing)下载我们所有基于中国古诗生成的结果。

如果您使用本项目创作出有趣的作品，我们欢迎投稿分享。

## 问题与改进

该项目初衷在于娱乐，因此并未深入研究诸多细节，为了加快理解和重构项目的进程，已删除了一些功能（如VR/3D/视频等）。若您对此感兴趣并愿意恢复原始功能，非常欢迎提交Pull Request（PR）进行改进和扩展。  



----



# English



# disco-diffusion-wrapper

Implementation of a [disco diffusion](https://github.com/alembics/disco-diffusion) wrapper that could run on your own GPU with a batch  of input text. 

Based on this work, an AI painting website (https://6pen.art/) was built, you may have a try.


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
You can also turn-off text  translation by setting `USE_TRANSLATE=False` so that you don't need the DeepL authKey. 

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
cd disco-diffusion-wrapper/
conda create --name disco
conda activate disco
pip install -r requirements.txt
cd wrapper/

# download pretrain model checkpoints
"""If this step is slow, you can download these model mannually from urls in the 
code of mutils.py, then put these checkpoints in the corresponding folder."""
python mutils.py

# Get your deepL API key, then add it to your run.py or batch_run.py to replace:
USE_TRANSLATE=True
auth_key = ""

# generated image by one sentence
python run.py "一行白鹭上青天"

# batch generation
touch sentence.list
# fill your text in, one by a line.
python run_batch.py
```

## Samples
Below are some samples generated using this repo. 
![](samples/A_half-acre_square_pond_is_open_the_sky_and_clouds_wandering_together_ink_painting.png)
> 半亩方塘一鉴开，天光云影共徘徊，水墨画

![](samples/Snowy_mountains_plains_army_of_ghosts_marching_at_night_with_fluorescent_green_torches_abstract_painting_Trending_on_artstation.png)
> 雪山，平原，幽鬼军队在夜里打着荧光绿色的火把前行，抽象画，Trending on artstation

![](samples/Glass_whale_lying_in_ruins_abstract_painting_Trending_on_artstation.png)
>躺在废墟中的玻璃鲸鱼，抽象画


![](samples/明月松间照清泉石上流中国画.png)
> 明月松间照，清泉石上流，中国画

![](samples/Effie_castle.jpeg)
> Elf Castle

![](samples/山随平野尽江入大荒流Trending_on_artstation.png)
> 山随平野尽，江入大荒流，Trending on artstation

You can download all our Chinese poem generating results from [here](https://drive.google.com/file/d/1OIsupQqMaYYWu4B0eemUWgvPfTSGyaqf/view?usp=sharing).

If you use this project and produced some interesting results, submissions are welcomed.



## Problems
As it's for fun, I did not look much into the details, and deleted many features(such as VR/3D/Video ...) to make me faster and more clear on the project resonctruction. It's awesome if you're interested in restoring the original function, PR is wellcomed.



