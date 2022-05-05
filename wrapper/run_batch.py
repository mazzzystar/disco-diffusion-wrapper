from mutils import *
from guided_diffusion.script_util import create_model_and_diffusion
import deepl
import sys
import string
import os
from zhon.hanzi import punctuation as punc_zh

auth_key = ""  # Replace with your free deepL key. See https://github.com/DeepLcom/deepl-python
translator = deepl.Translator(auth_key)
torch.backends.cudnn.benchmark = True

def translate(text, source_lang="ZH", target_lang="EN-US"):
    """
    You can change the source language by your situation.
    """
    res = translator.translate_text(text, source_lang="ZH", target_lang="EN-US")
    return res

def remove_comma(dirty_text):
    text = dirty_text.strip()
    # remove English comma.
    punc_en = string.punctuation
    for c in punc_en:
        text = text.replace(c, '')
    # turn blank into "_"
    text = text.replace(' ', '_')
    # remove Chinese comma.
    for c in punc_zh:
        text = text.replace(c, '')
    return text
        

def simple_prompts(text, image_path=None):
    text_prompts = {
        0: [f"{text}"],
    }
    image_prompts = None
    if image_path is not None and len(image_path) > 3:
        image_prompts = {
            0:[f'{image_path}:2',],
        }
    print('Text prompts only.') if image_prompts is None else print('Text & Image prompts both used.')
    return text_prompts, image_prompts
        
def load_diffusion_model(batch_name):
    model_config, clip_models, secondary_model, lpips_model = choose_diffusion_model(diffusion_model_name='512x512_diffusion_uncond_finetune_008100', use_secondary_model=True, diffusion_sampling_mode='ddim')
    set_parameters(model_config=model_config, batch_name=batch_name)
    print('Prepping model...')
#     print(model_config)
    model, diffusion = create_model_and_diffusion(**model_config)
    model.load_state_dict(torch.load(f'{model_path}/{diffusion_model}.pt', map_location='cpu'))
    model.requires_grad_(False).eval().to(device)
    for name, param in model.named_parameters():
        if 'qkv' in name or 'norm' in name or 'proj' in name:
            param.requires_grad_()
    if model_config['use_fp16']:
        model.convert_to_fp16()
    print('Finished of loading model.')
    return model, diffusion, clip_models, secondary_model, lpips_model
    
class DiscoDiffusion():
    def __init__(self, batch_name):
        self.batch_name = batch_name
        self.model, self.diffusion, self.clip_models, self.secondary_model, self.lpips_model = load_diffusion_model(batch_name)
    
    def draw(self, text_prompts, orig_text, image_prompts={}, steps=200, batch_size=1, display_rate=40):
        global args
        batchNum = args["batchNum"]
        start_frame = args["start_frame"]

        print(f'Starting Run: {self.batch_name}({batchNum}) at frame {start_frame}')
        args['prompts_series'] = split_prompts(text_prompts) if text_prompts else None
        args['image_prompts_series'] = split_prompts(image_prompts) if image_prompts else None
        args['text_prompts'] = text_prompts
        args['image_prompts'] = image_prompts
        args['batch_name'] = self.batch_name
        args['steps'] = steps
        args['batch_size'] = batch_size
        args['display_rate'] = display_rate

        args = SimpleNamespace(**args)

        gc.collect()
        torch.cuda.empty_cache()
        try:
            do_run(self.model, self.diffusion, self.clip_models, self.secondary_model, self.lpips_model)
        except Exception as e:
            print(e)
            pass
        finally:
            localfile = f'images_out/{self.batch_name}/'+text_prompts[0][0].replace(',', '').replace('.', '').replace(' ', '_') + '_200.png'
            
            newfile_name = remove_comma(orig_text)
            newfile_path = f'images_out/{self.batch_name}/'+ newfile_name +".png"
            os.rename(localfile, newfile_path)
   
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == '__main__':
    f = open('sentence.list', 'r')
    # The folder name to save all output images.
    disco = DiscoDiffusion('poem')
    # If style is not needed, then delete this line.
    style_class = ["ink painting", "中国画", "Trending on artstation", "水彩画"] 
    for line in f:
        if style_class is None or len(style_class) == 0:
            style_class = [" "]
        for style in style_class:
            try:
                del args
                from utils import *
                input_text = line.strip() + "，" + str(style)
                print(input_text)
                input_img_path = "" # If image input not needed, set None or ""
                result = translate(input_text)

                text_prompts, image_prompts = simple_prompts(result, input_img_path)
                print(text_prompts)
                disco.draw(text_prompts, input_text, steps=240, image_prompts=image_prompts, display_rate=40)
            except Exception as e:
                print(e)
    del translator
    
